from __future__ import annotations
import ast
import warnings
from collections import defaultdict, deque
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import dace
import dace.library
from dace import subsets
from dace.properties import CodeBlock
from dace.sdfg import nodes, utils as sdfg_utils
from dace.sdfg.state import AbstractControlFlowRegion, SDFGState
from dace.transformation.transformation import ExpandTransformation
from tqdm.auto import tqdm

from fp_arena.dtypes import float32sr, float64sr


# Default promotion rules for the standard float/SR types.
DEFAULT_PROMOTION_RULES: Dict[FrozenSet, dace.dtypes.typeclass] = {
    # Exact IEEE floats: widen.
    frozenset({dace.float16, dace.float32}): dace.float32,
    frozenset({dace.float16, dace.float64}): dace.float64,
    frozenset({dace.float32, dace.float64}): dace.float64,
    # Exact float mixed with an SR float: widen, result stays stochastic.
    frozenset({dace.float16, float32sr}): float32sr,
    frozenset({dace.float32, float32sr}): float32sr,
    frozenset({dace.float64, float32sr}): float64sr,
    frozenset({dace.float16, float64sr}): float64sr,
    frozenset({dace.float32, float64sr}): float64sr,
    frozenset({dace.float64, float64sr}): float64sr,
    # Two SR floats: widen.
    frozenset({float32sr, float64sr}): float64sr,
}


#: Qualified array identity: ``(id(sdfg), array_name)``.
QKey = Tuple[int, str]


# Whether *tc* is a floating-point typeclass this pass may retarget.
def _is_fp(tc: Optional[dace.dtypes.typeclass]) -> bool:
    if tc is None:
        return False
    if isinstance(tc, dace.mpfr):
        return True
    return tc in (dace.float16, dace.float32, dace.float64, float32sr, float64sr)


# Returns the promoted type of *t1* and *t2* according to *rules*.
def _promote(
    t1: Optional[dace.dtypes.typeclass],
    t2: Optional[dace.dtypes.typeclass],
    rules: Dict[FrozenSet, dace.dtypes.typeclass],
) -> Optional[dace.dtypes.typeclass]:
    if t1 is None:
        return t2
    if t2 is None:
        return t1
    if t1 == t2:
        return t1
    key = frozenset({t1, t2})
    if key not in rules:
        raise ValueError(f"No promotion rule defined for types {t1} and {t2}")
    return rules[key]


# None-safe equality for optional typeclasses (``dace.typeclass != None`` is unreliable).
def _types_equal(
    a: Optional[dace.dtypes.typeclass],
    b: Optional[dace.dtypes.typeclass],
) -> bool:
    if a is None or b is None:
        return a is b
    return bool(a == b)


# Returns states using topological_sort, visiting nested regions recursively
def _states_in_order(cfg: AbstractControlFlowRegion):
    for block in sdfg_utils.dfs_topological_sort(cfg):
        if isinstance(block, SDFGState):
            yield block
        elif isinstance(block, AbstractControlFlowRegion):
            yield from _states_in_order(block)


# Union-find over qualified array keys. Merging the two sides of every
# NestedSDFG boundary edge makes "outer array" and "inner connector array"
# a single node in the dataflow graph, so both sides always infer the same
# type.
class _UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[QKey, QKey] = {}

    def find(self, x: QKey) -> QKey:
        parent = self._parent
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != x:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: QKey, b: QKey) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


# All SDFGs in the hierarchy, root first. Rejects the two structures the
# pass cannot handle safely: a nested SDFG object shared by several
# NestedSDFG nodes (one descriptor mutation would silently retype every
# call site) and View descriptors (their dtype must track the viewed array,
# which this pass does not model).
def _collect_sdfgs(sdfg: dace.SDFG) -> List[dace.SDFG]:
    sdfgs = list(sdfg.all_sdfgs_recursive())
    seen: Set[int] = set()
    for sd in sdfgs:
        if id(sd) in seen:
            raise NotImplementedError(
                f"Nested SDFG object {sd.name!r} is referenced by more than one "
                "NestedSDFG node; retyping it once would affect every call site."
            )
        seen.add(id(sd))
        for name, desc in sd.arrays.items():
            if isinstance(desc, dace.data.View):
                raise NotImplementedError(
                    f"View descriptor {name!r} in SDFG {sd.name!r} is not supported."
                )
    return sdfgs


# Warns about float data this pass cannot retarget: float-typed nested-SDFG
# symbols (symbols are not data descriptors) and float arrays read on
# interstate edges (those reads follow the array's final precision but are
# implicitly widened to the symbol's fixed dtype; no casts are inserted).
def _warn_unlowerable(sdfgs: List[dace.SDFG]) -> None:
    fp_symbols: Set[str] = set()
    for sd in sdfgs:
        for state in sd.all_states():
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue
                for sym in node.symbol_mapping:
                    if _is_fp(node.sdfg.symbols.get(sym)):
                        fp_symbols.add(sym)
    if fp_symbols:
        warnings.warn(
            "change_and_propagate_fp_types: float-typed nested-SDFG symbols are "
            f"not retargetable (symbols are not data): {sorted(fp_symbols)}"
        )

    interstate_reads: Set[str] = set()
    for sd in sdfgs:
        fp_names = {n for n, d in sd.arrays.items() if _is_fp(d.dtype)}
        for e in sd.all_interstate_edges():
            interstate_reads |= e.data.free_symbols & fp_names
    if interstate_reads:
        warnings.warn(
            "change_and_propagate_fp_types: float arrays read on interstate edges "
            "follow their final precision but are widened to the target symbol's "
            f"dtype without explicit casts: {sorted(interstate_reads)}"
        )


# Merges the two sides of every NestedSDFG boundary edge into one class.
def _unify_nested_boundaries(sdfgs: List[dace.SDFG]) -> _UnionFind:
    uf = _UnionFind()
    for sd in sdfgs:
        sid = id(sd)
        for state in sd.all_states():
            for node in state.nodes():
                if not isinstance(node, nodes.NestedSDFG):
                    continue
                iid = id(node.sdfg)
                for e in state.in_edges(node):
                    if e.dst_conn and e.data is not None and e.data.data is not None:
                        uf.union((sid, e.data.data), (iid, e.dst_conn))
                for e in state.out_edges(node):
                    if e.src_conn and e.data is not None and e.data.data is not None:
                        uf.union((sid, e.data.data), (iid, e.src_conn))
    return uf


# Class membership and per-class original dtype.
#
# Returns:
#   members: class representative -> [(sdfg, name), ...]
#   original: class representative -> the (single) original dtype
# Raises if the members of one class disagree on their original dtype -- the
# input SDFG would already be reinterpreting memory across that boundary.
def _class_types(
    sdfgs: List[dace.SDFG],
    uf: _UnionFind,
) -> Tuple[Dict[QKey, List[Tuple[dace.SDFG, str]]], Dict[QKey, dace.dtypes.typeclass]]:
    members: Dict[QKey, List[Tuple[dace.SDFG, str]]] = defaultdict(list)
    for sd in sdfgs:
        for name in sd.arrays:
            members[uf.find((id(sd), name))].append((sd, name))

    original: Dict[QKey, dace.dtypes.typeclass] = {}
    for rep, mem in members.items():
        dtypes = {sd.arrays[name].dtype for sd, name in mem}
        if len(dtypes) > 1:
            names = ", ".join(f"{sd.name}.{name}" for sd, name in mem)
            raise ValueError(
                f"Arrays sharing a NestedSDFG boundary disagree on dtype: {names} "
                f"({sorted(t.to_string() for t in dtypes)})"
            )
        original[rep] = next(iter(dtypes))
    return members, original


# Builds the array-level dataflow graph of the SDFG.
#
# Returns:
#   producers: array -> set of arrays that feed any computation writing it
#   consumers: the reverse map (array -> arrays it feeds into)
def _build_dataflow(
    sdfgs: List[dace.SDFG],
    uf: _UnionFind,
) -> Tuple[Dict[QKey, Set[QKey]], Dict[QKey, Set[QKey]]]:
    producers: Dict[QKey, Set[QKey]] = defaultdict(set)
    consumers: Dict[QKey, Set[QKey]] = defaultdict(set)

    for sd in sdfgs:
        sid = id(sd)
        for state in _states_in_order(sd):
            for node in state.nodes():
                if isinstance(node, (nodes.Tasklet, nodes.LibraryNode)):
                    ins = {
                        uf.find((sid, e.data.data))
                        for e in state.in_edges(node)
                        if e.data is not None and e.data.data is not None
                    }
                    outs = {
                        uf.find((sid, e.data.data))
                        for e in state.out_edges(node)
                        if e.data is not None and e.data.data is not None
                    }
                    for out in outs:
                        producers[out].update(ins)
                        for inp in ins:
                            consumers[inp].add(out)

            # Direct AccessNode -> AccessNode copies.
            for e in state.edges():
                if e.data is None or e.data.data is None:
                    continue
                if isinstance(e.src, nodes.AccessNode) and isinstance(
                    e.dst, nodes.AccessNode
                ):
                    src = uf.find((sid, e.src.data))
                    dst = uf.find((sid, e.dst.data))
                    producers[dst].add(src)
                    consumers[src].add(dst)

    return producers, consumers


# Arrays reachable in the producer graph from any *seed*, following the producer -> consumer direction.
def _reachable_from(
    seeds: Set[QKey],
    consumers: Dict[QKey, Set[QKey]],
) -> Set[QKey]:
    reached = set(seeds)
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        for nxt in consumers.get(cur, ()):
            if nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)
    return reached


# Computes a fixed-point precision for every class of arrays.
#
# Each class's type is the join (widest, via *rules*) of its producers' types.
# Classes are classified once up front:
#   - pinned classes (in *initial_types*) are constants;
#   - non-floating-point classes are constants at their original dtype and
#     never contribute to a join (an int/bool input must not widen an fp output);
#   - source classes (no producers) are constants at their original dtype;
#   - classes not reachable from any pinned/source seed can never receive a type
#     from the fixpoint (uninitialized-transient cycles); they keep their
#     original dtype as constants;
#   - all remaining (reachable, derived) classes start at None (bottom) and only
#     ever widen, so demotion propagates correctly and the monotone worklist
#     terminates. A class still None afterwards (fed exclusively by non-fp
#     producers) falls back to its original dtype.
# Returns class representative -> dtype for every class.
def _infer_types(
    members: Dict[QKey, List[Tuple[dace.SDFG, str]]],
    producers: Dict[QKey, Set[QKey]],
    consumers: Dict[QKey, Set[QKey]],
    initial_types: Dict[QKey, dace.dtypes.typeclass],
    original_types: Dict[QKey, dace.dtypes.typeclass],
    rules: Dict[FrozenSet, dace.dtypes.typeclass],
) -> Dict[QKey, dace.dtypes.typeclass]:

    seeds = {
        name
        for name in members
        if name in initial_types
        or not _is_fp(original_types[name])
        or not producers.get(name)
    }
    reachable = _reachable_from(seeds, consumers)

    inferred: Dict[QKey, Optional[dace.dtypes.typeclass]] = {}
    pinned: Set[QKey] = set()
    for name in members:
        if name in initial_types:
            inferred[name] = initial_types[name]  # pin: constant
            pinned.add(name)
        elif not _is_fp(original_types[name]):
            inferred[name] = original_types[name]  # non-fp: constant
            pinned.add(name)
        elif not producers.get(name):
            inferred[name] = original_types[name]  # source: constant at original
            pinned.add(name)
        elif name not in reachable:
            inferred[name] = original_types[name]  # unreachable cycle: unchanged
            pinned.add(name)
        else:
            inferred[name] = None  # reachable derived: bottom, widened below

    worklist = deque(name for name in members if name not in pinned)
    queued = set(worklist)
    while worklist:
        name = worklist.popleft()
        queued.discard(name)

        new: Optional[dace.dtypes.typeclass] = None
        for prod in producers[name]:
            t = inferred[prod]
            if t is None or not _is_fp(t):
                continue
            new = _promote(new, t, rules)

        if not _types_equal(new, inferred[name]):
            inferred[name] = new
            for cons in consumers.get(name, ()):
                if cons in pinned or cons in queued:
                    continue
                worklist.append(cons)
                queued.add(cons)

    # Fed exclusively by non-fp producers: keep the original dtype.
    for name in members:
        if inferred[name] is None:
            inferred[name] = original_types[name]

    return inferred


# Prints a report of each array's original and final (inferred) precision
def _print_type_report(
    original_types: Dict[str, dace.dtypes.typeclass],
    inferred: Dict[str, dace.dtypes.typeclass],
) -> None:
    name_w = max((len(n) for n in original_types), default=0)
    name_w = max(name_w, len("array"))
    lines = [
        "change_and_propagate_fp_types: precision report",
        f"  {'array':<{name_w}}  {'original':<9}  {'final':<9}",
    ]
    for name in sorted(original_types):
        orig = original_types[name]
        final = inferred[name]
        marker = "" if _types_equal(final, orig) else "  (changed)"
        lines.append(
            f"  {name:<{name_w}}  {orig.to_string():<9}  {final.to_string():<9}{marker}"
        )
    tqdm.write("\n".join(lines))


def _connector_type(
    node: nodes.Node,
    desc: dace.data.Data,
    memlet: dace.Memlet,
    dtype: dace.dtypes.typeclass,
    output: bool,
) -> dace.dtypes.typeclass:
    scalar = bool(memlet.subset) and memlet.subset.num_elements() == 1
    if output:
        scalar = scalar and (not memlet.dynamic or memlet.wcr is not None)
    scalar = scalar or isinstance(desc, dace.data.Scalar)
    if isinstance(node, nodes.LibraryNode):
        scalar = scalar and desc.storage is not dace.dtypes.StorageType.GPU_Global
    return dtype if scalar else dace.dtypes.pointer(dtype)


# Writes inferred types onto tasklet/map/library-node connectors.
def _apply_connector_types(
    sdfg: dace.SDFG,
    inferred: Dict[str, dace.dtypes.typeclass],
) -> None:
    for state in _states_in_order(sdfg):
        for node in state.nodes():
            if not isinstance(
                node,
                (nodes.Tasklet, nodes.LibraryNode, nodes.EntryNode, nodes.ExitNode),
            ):
                continue
            for e in state.in_edges(node):
                if e.dst_conn is None or e.data is None or e.data.data is None:
                    continue
                if e.dst_conn not in node.in_connectors:
                    continue
                node.in_connectors[e.dst_conn] = _connector_type(
                    node,
                    sdfg.arrays[e.data.data],
                    e.data,
                    inferred[e.data.data],
                    output=False,
                )

            for e in state.out_edges(node):
                if e.src_conn is None or e.data is None or e.data.data is None:
                    continue
                if e.src_conn not in node.out_connectors:
                    continue
                node.out_connectors[e.src_conn] = _connector_type(
                    node,
                    sdfg.arrays[e.data.data],
                    e.data,
                    inferred[e.data.data],
                    output=True,
                )


def _cast_tasklet(
    state: dace.SDFGState,
    name: str,
    in_dtype: dace.dtypes.typeclass,
    out_dtype: dace.dtypes.typeclass,
) -> nodes.Tasklet:
    """A scalar ``_out = out_dtype(_in)`` cast tasklet."""

    t = state.add_tasklet(
        name=name,
        inputs={"_in": in_dtype},
        outputs={"_out": out_dtype},
        code=f"_out = dace.{out_dtype.to_string()}(_in)",
        language=dace.Language.Python,
    )
    t.in_connectors["_in"] = in_dtype
    t.out_connectors["_out"] = out_dtype
    return t


# The join of a tasklet's floating-point input types.
def _input_join(
    state: dace.SDFGState,
    node: nodes.Tasklet,
    sdfg: dace.SDFG,
    rules: dict[frozenset, dace.dtypes.typeclass],
) -> dace.dtypes.typeclass | None:
    join: dace.dtypes.typeclass | None = None
    for e in state.in_edges(node):
        if e.dst_conn is None or e.data is None or e.data.data is None:
            continue
        dtype = sdfg.arrays[e.data.data].dtype
        if not _is_fp(dtype):
            continue
        try:
            join = _promote(join, dtype, rules)
        except ValueError:
            return None
    return join


# Whether a float value crosses one of *node*'s connectors as a pointer: the
# tasklet then reads or writes a whole subset, which a scalar cast cannot mirror.
def _has_bulk_fp_connector(
    state: dace.SDFGState,
    node: nodes.Tasklet,
    sdfg: dace.SDFG,
) -> bool:
    edges = [(e.dst_conn, e, node.in_connectors) for e in state.in_edges(node)]
    edges += [(e.src_conn, e, node.out_connectors) for e in state.out_edges(node)]
    for conn, e, connectors in edges:
        if conn is None or e.data is None or e.data.data is None:
            continue
        if not _is_fp(sdfg.arrays[e.data.data].dtype):
            continue
        if isinstance(connectors.get(conn), dace.dtypes.pointer):
            return True
    return False


# Reads edge *e*'s array, converts it to *dtype*, and feeds that to the tasklet.
def _cast_before(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    e,
    dtype: dace.dtypes.typeclass,
) -> None:
    src_name = e.data.data
    tmp_name, _ = sdfg.add_scalar(
        f"{src_name}_at_{dtype.to_string()}", dtype, transient=True, find_new_name=True
    )
    cast = _cast_tasklet(
        state, f"cast_{src_name}_to_{tmp_name}", sdfg.arrays[src_name].dtype, dtype
    )
    tmp = state.add_access(tmp_name)
    state.remove_edge(e)
    e.dst.in_connectors[e.dst_conn] = dtype
    state.add_edge(e.src, e.src_conn, cast, "_in", dace.Memlet.from_memlet(e.data))
    state.add_edge(cast, "_out", tmp, None, dace.Memlet(data=tmp_name))
    state.add_edge(tmp, None, e.dst, e.dst_conn, dace.Memlet(data=tmp_name))


# Takes the tasklet's result at *dtype* and converts it into edge *e*'s array.
def _cast_after(
    state: dace.SDFGState,
    sdfg: dace.SDFG,
    e,
    dtype: dace.dtypes.typeclass,
) -> None:
    dst_name = e.data.data
    tmp_name, _ = sdfg.add_scalar(
        f"{dst_name}_at_{dtype.to_string()}", dtype, transient=True, find_new_name=True
    )
    cast = _cast_tasklet(
        state, f"cast_{tmp_name}_to_{dst_name}", dtype, sdfg.arrays[dst_name].dtype
    )
    tmp = state.add_access(tmp_name)
    state.remove_edge(e)
    e.src.out_connectors[e.src_conn] = dtype
    state.add_edge(e.src, e.src_conn, tmp, None, dace.Memlet(data=tmp_name))
    state.add_edge(tmp, None, cast, "_in", dace.Memlet(data=tmp_name))
    state.add_edge(cast, "_out", e.dst, e.dst_conn, dace.Memlet.from_memlet(e.data))


# Gives every conversion hidden inside a tasklet a node of its own: a tasklet
# reads and writes exactly one floating-point dtype -- the join of the types it
# reads -- and each conversion to or from it lives in a cast tasklet.
def _homogenize_tasklet_dtypes(
    sdfg: dace.SDFG,
    rules: dict[frozenset, dace.dtypes.typeclass],
) -> None:
    for state in _states_in_order(sdfg):
        for node in list(state.nodes()):
            if not isinstance(node, nodes.Tasklet):
                continue
            join = _input_join(state, node, sdfg, rules)
            if join is None or _has_bulk_fp_connector(state, node, sdfg):
                continue
            for e in list(state.in_edges(node)):
                if e.dst_conn is None or e.data is None or e.data.data is None:
                    continue
                dtype = sdfg.arrays[e.data.data].dtype
                if _is_fp(dtype) and not _types_equal(dtype, join):
                    _cast_before(state, sdfg, e, join)
            for e in list(state.out_edges(node)):
                if e.src_conn is None or e.data is None or e.data.data is None:
                    continue
                dtype = sdfg.arrays[e.data.data].dtype
                if _is_fp(dtype) and not _types_equal(dtype, join):
                    _cast_after(state, sdfg, e, join)


# Inserts a cast wherever an edge connects two differently-typed containers:
# map-boundary copies (AccessNode <-> MapExit/MapEntry) get a scalar cast
# tasklet spliced into the enclosing map's iteration; direct
# AccessNode -> AccessNode copies get an elementwise cast map over the
# copied subset.
def _insert_edge_casts(sdfg: dace.SDFG) -> None:
    for state in sdfg.all_states():
        for i, e in enumerate(list(state.edges())):
            if e.data is None or e.data.data is None:
                continue
            mem = e.data.data
            mem_dtype = sdfg.arrays[mem].dtype

            # Write: AccessNode X -> MapExit, memlet writes a different array.
            if isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.MapExit):
                x = e.src.data
                if mem == x or sdfg.arrays[x].dtype == mem_dtype:
                    continue
                t = _cast_tasklet(
                    state, f"cast_{x}_to_{mem}_{i}", sdfg.arrays[x].dtype, mem_dtype
                )
                read = (
                    dace.Memlet(data=x, subset=e.data.other_subset)
                    if e.data.other_subset is not None
                    else dace.Memlet(data=x)
                )
                write = dace.Memlet(data=mem, subset=e.data.subset)
                state.remove_edge(e)
                state.add_edge(e.src, None, t, "_in", read)
                state.add_edge(t, "_out", e.dst, e.dst_conn, write)

            # Read: MapEntry -> AccessNode X, memlet reads a different array.
            elif isinstance(e.src, nodes.MapEntry) and isinstance(
                e.dst, nodes.AccessNode
            ):
                x = e.dst.data
                if mem == x or sdfg.arrays[x].dtype == mem_dtype:
                    continue
                t = _cast_tasklet(
                    state, f"cast_{mem}_to_{x}_{i}", mem_dtype, sdfg.arrays[x].dtype
                )
                read = dace.Memlet(data=mem, subset=e.data.subset)
                write = (
                    dace.Memlet(data=x, subset=e.data.other_subset)
                    if e.data.other_subset is not None
                    else dace.Memlet(data=x)
                )
                state.remove_edge(e)
                state.add_edge(e.src, e.src_conn, t, "_in", read)
                state.add_edge(t, "_out", e.dst, None, write)

            # Direct copy: AccessNode -> AccessNode.
            elif isinstance(e.src, nodes.AccessNode) and isinstance(
                e.dst, nodes.AccessNode
            ):
                src_name, dst_name = e.src.data, e.dst.data
                src_desc, dst_desc = sdfg.arrays[src_name], sdfg.arrays[dst_name]
                if src_desc.dtype == dst_desc.dtype:
                    continue
                if e.data.wcr is not None:
                    raise NotImplementedError(
                        f"WCR copy between differently-typed containers "
                        f"{src_name!r} -> {dst_name!r} is not supported"
                    )

                if mem == src_name:
                    src_subset, dst_subset = e.data.subset, e.data.other_subset
                elif mem == dst_name:
                    src_subset, dst_subset = e.data.other_subset, e.data.subset
                else:
                    continue  # not a plain copy of these two containers
                if src_subset is None:
                    src_subset = subsets.Range.from_array(src_desc)
                if dst_subset is None:
                    dst_subset = subsets.Range.from_array(dst_desc)

                src_nd = [(d, sz) for d, sz in enumerate(src_subset.size()) if sz != 1]
                dst_nd = [(d, sz) for d, sz in enumerate(dst_subset.size()) if sz != 1]
                if len(src_nd) != len(dst_nd) or any(
                    s1 != s2 for (_, s1), (_, s2) in zip(src_nd, dst_nd)
                ):
                    raise NotImplementedError(
                        f"Reshaping copy between differently-typed containers "
                        f"{src_name}[{src_subset}] -> {dst_name}[{dst_subset}] is "
                        "not supported"
                    )

                t = _cast_tasklet(
                    state,
                    f"cast_copy_{src_name}_to_{dst_name}_{i}",
                    src_desc.dtype,
                    dst_desc.dtype,
                )
                map_ranges = {f"_i{k}": f"0:{sz}" for k, (_, sz) in enumerate(src_nd)}
                if not map_ranges:  # single-element copy
                    map_ranges = {"_i0": "0:1"}
                map_entry, map_exit = state.add_map(
                    name=f"cast_copy_map_{src_name}_to_{dst_name}_{i}",
                    ndrange=map_ranges,
                )
                map_entry.add_in_connector(f"IN_{src_name}")
                map_entry.add_out_connector(f"OUT_{src_name}")
                map_exit.add_in_connector(f"IN_{dst_name}")
                map_exit.add_out_connector(f"OUT_{dst_name}")

                def _elem_idx(subset, nondegenerate):
                    param_of = {d: f"_i{k}" for k, (d, _) in enumerate(nondegenerate)}
                    return ", ".join(
                        f"({rb}) + ({rs})*{param_of[d]}" if d in param_of else f"{rb}"
                        for d, (rb, _, rs) in enumerate(subset)
                    )

                src_idx = _elem_idx(src_subset, src_nd)
                dst_idx = _elem_idx(dst_subset, dst_nd)

                state.remove_edge(e)
                state.add_edge(
                    e.src,
                    e.src_conn,
                    map_entry,
                    f"IN_{src_name}",
                    dace.Memlet(data=src_name, subset=src_subset),
                )
                state.add_edge(
                    map_entry,
                    f"OUT_{src_name}",
                    t,
                    "_in",
                    dace.Memlet(expr=f"{src_name}[{src_idx}]"),
                )
                state.add_edge(
                    t,
                    "_out",
                    map_exit,
                    f"IN_{dst_name}",
                    dace.Memlet(expr=f"{dst_name}[{dst_idx}]"),
                )
                state.add_edge(
                    map_exit,
                    f"OUT_{dst_name}",
                    e.dst,
                    e.dst_conn,
                    dace.Memlet(data=dst_name, subset=dst_subset),
                )


# Adds a map that copies and casts *src_name* to *dst_name*
def _add_copy_map(
    state: dace.SDFGState,
    src_name: str,
    src_arr: dace.data.Data,
    dst_name: str,
    dst_arr: dace.data.Data,
) -> None:

    if src_arr.shape != dst_arr.shape:
        raise ValueError(
            f"Shape mismatch in _add_copy_map: "
            f"{src_name}{src_arr.shape} vs {dst_name}{dst_arr.shape}"
        )

    tasklet = _cast_tasklet(
        state, f"cast_{src_name}_to_{dst_name}", src_arr.dtype, dst_arr.dtype
    )

    if isinstance(src_arr, dace.data.Array):
        if not isinstance(dst_arr, dace.data.Array):
            raise TypeError(
                f"_add_copy_map: src '{src_name}' is an Array but dst '{dst_name}' is "
                f"{type(dst_arr).__name__}"
            )
        map_ranges = {f"_i{d}": f"0:{s}" for d, s in enumerate(src_arr.shape)}
        idx = ", ".join(map_ranges)

        map_entry, map_exit = state.add_map(
            name=f"cast_map_{src_name}_to_{dst_name}",
            ndrange=map_ranges,
        )
        map_entry.add_in_connector(f"IN_{src_name}")
        map_entry.add_out_connector(f"OUT_{src_name}")
        map_exit.add_in_connector(f"IN_{dst_name}")
        map_exit.add_out_connector(f"OUT_{dst_name}")

        src_an = state.add_access(src_name)
        dst_an = state.add_access(dst_name)
        state.add_edge(
            src_an,
            None,
            map_entry,
            f"IN_{src_name}",
            dace.memlet.Memlet.from_array(src_name, src_arr),
        )
        state.add_edge(
            map_entry,
            f"OUT_{src_name}",
            tasklet,
            "_in",
            dace.Memlet(expr=f"{src_name}[{idx}]"),
        )
        state.add_edge(
            tasklet,
            "_out",
            map_exit,
            f"IN_{dst_name}",
            dace.Memlet(expr=f"{dst_name}[{idx}]"),
        )
        state.add_edge(
            map_exit,
            f"OUT_{dst_name}",
            dst_an,
            None,
            dace.memlet.Memlet.from_array(dst_name, dst_arr),
        )

    else:
        if not isinstance(src_arr, dace.data.Scalar):
            raise TypeError(
                f"_add_copy_map: unsupported src descriptor type "
                f"{type(src_arr).__name__} for '{src_name}'"
            )
        if not isinstance(dst_arr, dace.data.Scalar):
            raise TypeError(
                f"_add_copy_map: unsupported dst descriptor type "
                f"{type(dst_arr).__name__} for '{dst_name}'"
            )
        src_an = state.add_access(src_name)
        dst_an = state.add_access(dst_name)
        state.add_edge(src_an, None, tasklet, "_in", dace.Memlet(expr=src_name))
        state.add_edge(tasklet, "_out", dst_an, None, dace.Memlet(expr=dst_name))


@dace.library.expansion
class _ExpandFusionBarrier(ExpandTransformation):
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg, **kwargs):
        return nodes.Tasklet(
            node.name,
            set(),
            set(),
            "// Fusion barrier",
            language=dace.Language.CPP,
            side_effects=True,
        )


@dace.library.node
class FusionBarrier(nodes.LibraryNode):
    """Empty, side-effecting library node used to prevent state fusion."""

    implementations = {"pure": _ExpandFusionBarrier}
    default_implementation = "pure"

    def __init__(self, name="fusion_barrier", *args, **kwargs):
        super().__init__(name, *args, inputs=set(), outputs=set(), **kwargs)

    def has_side_effects(self, sdfg) -> bool:
        return True


# Adds an empty side-effecting library node to prevent state fusion.
def _add_fusion_barrier(state: dace.SDFGState) -> None:
    state.add_node(FusionBarrier("fusion_barrier"))


# Wraps every float literal in ``expr = dace.<typename>(literal)``.
class _FloatConstantCaster(ast.NodeTransformer):
    def __init__(self, typename: str) -> None:
        self._typename = typename

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if not isinstance(node.value, float):
            return node
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="dace", ctx=ast.Load()),
                attr=self._typename,
                ctx=ast.Load(),
            ),
            args=[node],
            keywords=[],
        )


# Casts all float literals in Python tasklet bodies to *dtype*.
def _cast_float_constants(sdfg: dace.SDFG, dtype: dace.dtypes.typeclass) -> None:
    caster = _FloatConstantCaster(dtype.to_string())
    for sd in sdfg.all_sdfgs_recursive():
        for state in sd.all_states():
            for node in state.nodes():
                if not (
                    isinstance(node, nodes.Tasklet)
                    and node.language == dace.Language.Python
                ):
                    continue
                tree = ast.fix_missing_locations(
                    caster.visit(ast.parse(node.code.as_string))
                )
                node.code = CodeBlock(ast.unparse(tree), dace.Language.Python)


# Retypes every floating-point SDFG symbol and compile-time constant to *dtype*.
def _lower_symbols_and_constants(
    sdfgs: List[dace.SDFG], dtype: dace.dtypes.typeclass
) -> None:
    root = sdfgs[0]
    abi_symbols = set(map(str, root.free_symbols))
    for sd in sdfgs:
        for name, stype in list(sd.symbols.items()):
            if _is_fp(stype) and not (sd is root and name in abi_symbols):
                sd.symbols[name] = dtype
        for desc, _value in sd.constants_prop.values():
            if _is_fp(getattr(desc, "dtype", None)):
                desc.dtype = dtype
        # Symbols only assigned on interstate edges (never declared) get their C
        # type inferred from the assignment expression, whose float literals are
        # doubles; declare them explicitly at the constant precision.
        for e in sd.all_interstate_edges():
            for name, stype in e.data.new_symbols(sd, sd.symbols).items():
                if name not in sd.symbols and name not in sd.arrays and _is_fp(stype):
                    sd.add_symbol(name, dtype)


# Main entry point to change and propagate fp types through an SDFG.
def change_and_propagate_fp_types(
    sdfg: dace.SDFG,
    initial_types: Dict[str, dace.dtypes.typeclass],
    promotion_rules: Optional[
        Dict[FrozenSet[dace.dtypes.typeclass], dace.dtypes.typeclass]
    ] = None,
    constant_type: Optional[dace.dtypes.typeclass] = None,
) -> None:

    # Use default promotion rules if none are provided.
    rules = DEFAULT_PROMOTION_RULES if promotion_rules is None else promotion_rules

    sdfgs = _collect_sdfgs(sdfg)

    # Optionally cast every float literal in the computation to a fixed precision.
    if constant_type is not None:
        _cast_float_constants(sdfg, constant_type)
        _lower_symbols_and_constants(sdfgs, constant_type)
    _warn_unlowerable(sdfgs)

    # Unify the two sides of every NestedSDFG boundary, then resolve classes.
    uf = _unify_nested_boundaries(sdfgs)
    members, original_types = _class_types(sdfgs, uf)

    # Resolve pins (top-level array names) to classes.
    pins: Dict[QKey, dace.dtypes.typeclass] = {}
    for name, dtype in initial_types.items():
        if name not in sdfg.arrays:
            raise ValueError(
                f"Pinned array {name!r} not found in the top-level SDFG "
                f"{sdfg.name!r} (nested-only arrays cannot be pinned directly)"
            )
        if not _is_fp(sdfg.arrays[name].dtype):
            raise ValueError(f"Pinned array {name!r} is not a floating-point array")
        if not _is_fp(dtype):
            raise ValueError(
                f"Pinned type {dtype} for array {name!r} is not a floating-point type"
            )
        rep = uf.find((id(sdfg), name))
        if rep in pins and not _types_equal(pins[rep], dtype):
            raise ValueError(
                f"Conflicting pins for arrays sharing a NestedSDFG boundary "
                f"(class of {name!r}): {pins[rep]} vs {dtype}"
            )
        pins[rep] = dtype

    original_nontransients: Dict[str, dace.dtypes.typeclass] = {
        name: desc.dtype for name, desc in sdfg.arrays.items() if not desc.transient
    }

    # Build the array-level dataflow graph and solve the precision fixed point.
    producers, consumers = _build_dataflow(sdfgs, uf)

    inferred = _infer_types(
        members,
        producers,
        consumers,
        pins,
        original_types,
        rules,
    )

    # Report with level-qualified names (the root level stays unqualified).
    report_orig: Dict[str, dace.dtypes.typeclass] = {}
    report_final: Dict[str, dace.dtypes.typeclass] = {}
    for i, sd in enumerate(sdfgs):
        prefix = "" if i == 0 else f"{sd.name}@{i}/"
        for name, desc in sd.arrays.items():
            report_orig[prefix + name] = desc.dtype
            report_final[prefix + name] = inferred[uf.find((id(sd), name))]
    _print_type_report(report_orig, report_final)

    # Apply inferred dtypes to the arrays and connectors of every level.
    for sd in sdfgs:
        level_inferred = {name: inferred[uf.find((id(sd), name))] for name in sd.arrays}
        for name, dtype in level_inferred.items():
            if sd.arrays[name].dtype != dtype:
                sd.arrays[name].dtype = dtype

        _apply_connector_types(sd, level_inferred)

        # Give every conversion a node of its own: first the ones a tasklet body
        # would otherwise hide, then those on edges between differently-typed
        # containers.
        _homogenize_tasklet_dtypes(sd, rules)
        _insert_edge_casts(sd)

    # Preserve the external interface: non-transient arrays of the *root* SDFG
    # that changed type are renamed to an internal transient. Nested levels
    # need no interface preservation: Inner names and connectors are untouched
    # by the root-level rename.
    changed_interface = {
        name
        for name, orig_dtype in original_nontransients.items()
        if sdfg.arrays[name].dtype != orig_dtype
    }
    if not changed_interface:
        return

    repl_dict = {
        name: f"fp_casted_{name}_{sdfg.arrays[name].dtype.to_string()}"
        for name in changed_interface
    }

    # Snapshot read/write sets before renaming so orig_name is still meaningful.
    sdfg_inputs, sdfg_outputs = sdfg.read_and_write_sets()
    sdfg.replace_dict(repl_dict)

    # Reconstruct original-typed descriptors and add them back to the SDFG.
    orig_descs: Dict[str, dace.data.Data] = {}
    for orig_name, casted_name in repl_dict.items():
        casted_desc = sdfg.arrays[casted_name]
        casted_desc.transient = True

        orig_desc = casted_desc.clone()
        orig_desc.transient = False
        orig_desc.dtype = original_nontransients[orig_name]
        orig_descs[orig_name] = orig_desc

        sdfg.add_datadesc(name=orig_name, datadesc=orig_desc)

    # TODO: This is currently inefficient, copies all changed inputs for each sink state.
    for sink in sdfg.sink_nodes():
        copy_out_state = None
        for orig_name, casted_name in repl_dict.items():
            if orig_name in sdfg_outputs:
                if copy_out_state is None:
                    copy_out_state = sdfg.add_state_after(
                        state=sink, label=f"copy_out_{sink.label}"
                    )
                    _add_fusion_barrier(copy_out_state)
                _add_copy_map(
                    copy_out_state,
                    casted_name,
                    sdfg.arrays[casted_name],
                    orig_name,
                    orig_descs[orig_name],
                )

    # Copy in every renamed array -- even pure outputs may be overwritten only partially.
    copy_in_state = None
    for orig_name, casted_name in repl_dict.items():
        if copy_in_state is None:
            copy_in_state = sdfg.add_state_before(
                state=sdfg.start_block, label="copy_in"
            )
            _add_fusion_barrier(copy_in_state)
        _add_copy_map(
            copy_in_state,
            orig_name,
            orig_descs[orig_name],
            casted_name,
            sdfg.arrays[casted_name],
        )
