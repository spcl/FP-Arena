from collections import defaultdict, deque
from functools import reduce
from typing import Dict, FrozenSet, Optional, Set, Tuple

import dace
from dace.sdfg import nodes, utils as sdfg_utils
from dace.sdfg.state import AbstractControlFlowRegion, SDFGState

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


# Builds the array-level dataflow graph of the SDFG.
#
# Returns:
#   producers: array -> set of arrays that feed any computation writing it
#   consumers: the reverse map (array -> arrays it feeds into)
def _build_dataflow(
    sdfg: dace.SDFG,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    producers: Dict[str, Set[str]] = defaultdict(set)
    consumers: Dict[str, Set[str]] = defaultdict(set)

    for state in _states_in_order(sdfg):
        for node in state.nodes():
            if isinstance(node, nodes.NestedSDFG):
                raise NotImplementedError(
                    "Type propagation does not currently support NestedSDFG nodes. "
                    f"Found '{node.label}' in state '{state.label}'."
                )

            if isinstance(node, (nodes.Tasklet, nodes.LibraryNode)):
                ins = {
                    e.data.data
                    for e in state.in_edges(node)
                    if e.data is not None and e.data.data is not None
                }
                outs = {
                    e.data.data
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
                producers[e.dst.data].add(e.src.data)
                consumers[e.src.data].add(e.dst.data)

    return producers, consumers


# Arrays reachable in the producer graph from any *seed*, following the producer -> consumer direction.
def _reachable_from(
    seeds: Set[str],
    consumers: Dict[str, Set[str]],
) -> Set[str]:
    reached = set(seeds)
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        for nxt in consumers.get(cur, ()):
            if nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)
    return reached


# Computes a fixed-point precision for every array.
#
# Each array's type is the join (widest, via *rules*) of its producers' types.
# Arrays are classified once up front:
#   - pinned arrays (in *initial_types*) are constants;
#   - source arrays (no producers) are constants at their original dtype;
#   - arrays not reachable from any pinned/source seed can never receive a type
#     from the fixpoint (uninitialized-transient cycles); they keep their
#     original dtype as constants;
#   - all remaining (reachable, derived) arrays start at None (bottom) and only
#     ever widen, so demotion propagates correctly and the monotone worklist
#     terminates. Every such array is reachable from a seed, so none stays None.
# Returns name -> dtype for every array.
def _infer_types(
    sdfg: dace.SDFG,
    producers: Dict[str, Set[str]],
    consumers: Dict[str, Set[str]],
    initial_types: Dict[str, dace.dtypes.typeclass],
    original_types: Dict[str, dace.dtypes.typeclass],
    rules: Dict[FrozenSet, dace.dtypes.typeclass],
) -> Dict[str, dace.dtypes.typeclass]:

    seeds = {
        name for name in sdfg.arrays if name in initial_types or not producers.get(name)
    }
    reachable = _reachable_from(seeds, consumers)

    inferred: Dict[str, Optional[dace.dtypes.typeclass]] = {}
    pinned: Set[str] = set()
    for name in sdfg.arrays:
        if name in initial_types:
            inferred[name] = initial_types[name]  # pin: constant
            pinned.add(name)
        elif not producers.get(name):
            inferred[name] = original_types[name]  # source: constant at original
            pinned.add(name)
        elif name not in reachable:
            inferred[name] = original_types[name]  # unreachable cycle: unchanged
            pinned.add(name)
        else:
            inferred[name] = None  # reachable derived: bottom, widened below

    worklist = deque(name for name in sdfg.arrays if name not in pinned)
    queued = set(worklist)
    while worklist:
        name = worklist.popleft()
        queued.discard(name)

        new: Optional[dace.dtypes.typeclass] = None
        for prod in producers[name]:
            t = inferred[prod]
            if t is None:
                continue
            new = _promote(new, t, rules)

        if not _types_equal(new, inferred[name]):
            inferred[name] = new
            for cons in consumers.get(name, ()):
                if cons in pinned or cons in queued:
                    continue
                worklist.append(cons)
                queued.add(cons)

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
    print("\n".join(lines))


# Writes inferred types onto tasklet/map/library-node connectors.
def _apply_connector_types(
    sdfg: dace.SDFG,
    inferred: Dict[str, dace.dtypes.typeclass],
    rules: Dict[FrozenSet, dace.dtypes.typeclass],
) -> None:
    for state in _states_in_order(sdfg):
        for node in state.nodes():
            if isinstance(node, (nodes.Tasklet, nodes.LibraryNode)):
                in_types = []
                for e in state.in_edges(node):
                    if e.dst_conn is None or e.data is None or e.data.data is None:
                        continue
                    t = inferred[e.data.data]
                    node.in_connectors[e.dst_conn] = t
                    in_types.append(t)

                if not in_types:
                    continue
                compute = reduce(lambda a, b: _promote(a, b, rules), in_types)
                for e in state.out_edges(node):
                    if e.src_conn is None or e.data is None or e.data.data is None:
                        continue
                    node.out_connectors[e.src_conn] = compute

            elif isinstance(node, (nodes.EntryNode, nodes.ExitNode)):
                # MapEntry/Exit connectors follow the IN_/OUT_ convention; keep both in sync.
                for e in state.in_edges(node):
                    if not (e.dst_conn and e.dst_conn.startswith("IN_")):
                        continue
                    if e.data is None or e.data.data is None:
                        continue
                    t = inferred[e.data.data]
                    out_conn = "OUT_" + e.dst_conn[3:]
                    if e.dst_conn in node.in_connectors:
                        node.in_connectors[e.dst_conn] = t
                    if out_conn in node.out_connectors:
                        node.out_connectors[out_conn] = t


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

    tasklet = state.add_tasklet(
        name=f"cast_{src_name}_to_{dst_name}",
        inputs={"_in"},
        outputs={"_out"},
        code=f"_out = static_cast<{dst_arr.dtype.ctype}>(_in);",
        language=dace.Language.CPP,
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


# Main entry point to change and propagate fp types through an SDFG.
# When *instrument* is set, the generated copy_in/copy_out cast states are tagged
# with DaCe's Timer instrumentation so the runner can separate cast time from compute time.
def change_and_propagate_fp_types(
    sdfg: dace.SDFG,
    initial_types: Dict[str, dace.dtypes.typeclass],
    promotion_rules: Optional[
        Dict[FrozenSet[dace.dtypes.typeclass], dace.dtypes.typeclass]
    ] = None,
    instrument: bool = False,
) -> None:

    # Use default promotion rules if none are provided.
    rules = DEFAULT_PROMOTION_RULES if promotion_rules is None else promotion_rules

    original_types: Dict[str, dace.dtypes.typeclass] = {
        name: desc.dtype for name, desc in sdfg.arrays.items()
    }
    original_nontransients: Dict[str, dace.dtypes.typeclass] = {
        name: dtype
        for name, dtype in original_types.items()
        if not sdfg.arrays[name].transient
    }

    # Build the array-level dataflow graph and solve the precision fixed point.
    producers, consumers = _build_dataflow(sdfg)
    inferred = _infer_types(
        sdfg,
        producers,
        consumers,
        initial_types,
        original_types,
        rules,
    )

    _print_type_report(original_types, inferred)

    # Apply inferred dtypes to SDFG arrays.
    for name, dtype in inferred.items():
        if sdfg.arrays[name].dtype != dtype:
            sdfg.arrays[name].dtype = dtype

    _apply_connector_types(sdfg, inferred, rules)

    # Preserve the external interface: non-transient arrays that changed type are renamed to an internal transient.
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
                    if instrument:
                        copy_out_state.instrument = dace.InstrumentationType.Timer
                _add_copy_map(
                    copy_out_state,
                    casted_name,
                    sdfg.arrays[casted_name],
                    orig_name,
                    orig_descs[orig_name],
                )

    copy_in_state = None
    for orig_name, casted_name in repl_dict.items():
        if orig_name in sdfg_inputs:
            if copy_in_state is None:
                copy_in_state = sdfg.add_state_before(
                    state=sdfg.start_block, label="copy_in"
                )
                if instrument:
                    copy_in_state.instrument = dace.InstrumentationType.Timer
            _add_copy_map(
                copy_in_state,
                orig_name,
                orig_descs[orig_name],
                casted_name,
                sdfg.arrays[casted_name],
            )
