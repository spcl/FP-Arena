from functools import reduce
from typing import Dict, FrozenSet, Optional, Set

import dace
from dace.sdfg import nodes, utils as sdfg_utils
from dace.sdfg.state import AbstractControlFlowRegion, SDFGState

_MAX_FIXPOINT_ITERS = 10


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


# Returns the inferred type for the source of *edge*, or None if it cannot be determined.
def _edge_src_type(node_types: dict, edge) -> Optional[dace.dtypes.typeclass]:
    src = node_types.get(edge.src)
    if isinstance(src, dict):
        return src.get(edge.src_conn)
    return src


# Returns states using topological_sort, visiting nested regions recursively
def _states_in_order(cfg: AbstractControlFlowRegion):
    for block in sdfg_utils.dfs_topological_sort(cfg):
        if isinstance(block, SDFGState):
            yield block
        elif isinstance(block, AbstractControlFlowRegion):
            yield from _states_in_order(block)


# Runs one pass of type inference and promotion
def _propagate_state(
    state: SDFGState,
    inferred: Dict[str, Optional[dace.dtypes.typeclass]],
    initial_types: Dict[str, dace.dtypes.typeclass],
    supported: Set[dace.dtypes.typeclass],
    rules: Dict[FrozenSet, dace.dtypes.typeclass],
    original_types: Dict[str, dace.dtypes.typeclass],
) -> Dict:

    node_types: Dict = {}

    for node in sdfg_utils.dfs_topological_sort(state, state.source_nodes()):
        if isinstance(node, nodes.AccessNode):
            if node.data in initial_types:
                # User defined type: never changed by propagation.
                node_types[node] = initial_types[node.data]

            elif state.in_degree(node) == 0:
                # Source node: use inferred type, fall back to original if it is a supported fp type.
                t = inferred[node.data]
                if t is None:
                    t = original_types[node.data]
                node_types[node] = t

            else:
                # Destination node: collect all incoming supported fp types and promote them.
                incoming = [
                    t
                    for e in state.in_edges(node)
                    if (t := _edge_src_type(node_types, e)) in supported
                ]
                if not incoming:
                    node_types[node] = inferred[node.data]
                else:
                    new_type = reduce(lambda a, b: _promote(a, b, rules), incoming)
                    old = inferred[node.data]
                    merged = _promote(
                        old if (old is not None and old in supported) else None,
                        new_type,
                        rules,
                    )
                    inferred[node.data] = merged
                    node_types[node] = inferred[node.data]

        # TODO: This assumes all inputs are used for type inference. There are probalby some cases where this is not true. But this would probably need analysis of the tasklet code.
        elif isinstance(node, (nodes.Tasklet, nodes.LibraryNode)):
            in_types = [
                t
                for e in state.in_edges(node)
                if (t := _edge_src_type(node_types, e)) in supported
            ]
            promoted = (
                reduce(lambda a, b: _promote(a, b, rules), in_types)
                if in_types
                else None
            )

            out: Dict[str, Optional[dace.dtypes.typeclass]] = {}
            for e in state.out_edges(node):
                if promoted is None:
                    out[e.src_conn] = None
                    continue
                dst_name = e.data.data if e.data else None
                if dst_name:
                    dst_type = inferred.get(dst_name) or original_types.get(dst_name)
                    out[e.src_conn] = promoted if dst_type in supported else None
                else:
                    out[e.src_conn] = None
            node_types[node] = out

        elif isinstance(node, (nodes.EntryNode, nodes.ExitNode)):
            # TODO: Assumes that the MapEntry/Exit connectors follow the IN_/OUT_ convention. Is this safe to assume?
            out = {}
            for e in state.in_edges(node):
                if e.dst_conn and e.dst_conn.startswith("IN_"):
                    out_conn = "OUT_" + e.dst_conn[3:]
                    out[out_conn] = _edge_src_type(node_types, e)
            node_types[node] = out

        elif isinstance(node, nodes.NestedSDFG):
            raise NotImplementedError(
                "Type propagation does not currently support NestedSDFG nodes. "
                f"Found '{node.label}' in state '{state.label}'."
            )

        else:
            raise NotImplementedError(
                f"Unsupported node type {type(node).__name__} in state '{state.label}'"
            )

    return node_types


# Write inferred connector types back onto nodes.
def _apply_connector_types(
    global_node_types: Dict[SDFGState, Dict],
    supported: Set[dace.dtypes.typeclass],
) -> None:
    for state, node_types in global_node_types.items():
        for node, types in node_types.items():
            if isinstance(node, (nodes.Tasklet, nodes.LibraryNode)):
                if not isinstance(types, dict):
                    continue
                for conn, t in types.items():
                    if conn is not None and t in supported:
                        node.out_connectors[conn] = t
                for e in state.in_edges(node):
                    src_t = node_types.get(e.src)
                    if isinstance(src_t, dict):
                        src_t = src_t.get(e.src_conn)
                    if src_t in supported and e.dst_conn is not None:
                        node.in_connectors[e.dst_conn] = src_t

            elif isinstance(node, (nodes.EntryNode, nodes.ExitNode)):
                if not isinstance(types, dict):
                    continue
                for conn, t in types.items():
                    if t not in supported:
                        continue
                    if conn in node.out_connectors:
                        node.out_connectors[conn] = t
                    # Keep in_connectors in sync (MapEntry has both IN_ and OUT_ for each data conn).
                    in_conn = "IN_" + conn[4:] if conn.startswith("OUT_") else None
                    if in_conn and in_conn in node.in_connectors:
                        node.in_connectors[in_conn] = t


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
def change_and_propagate_fp_types(
    sdfg: dace.SDFG,
    initial_types: Dict[str, dace.dtypes.typeclass],
    promotion_rules: Dict[FrozenSet[dace.dtypes.typeclass], dace.dtypes.typeclass],
) -> None:

    supported: Set[dace.dtypes.typeclass] = set()
    for pair in promotion_rules:
        supported.update(pair)

    original_types: Dict[str, dace.dtypes.typeclass] = {
        name: desc.dtype for name, desc in sdfg.arrays.items()
    }
    original_nontransients: Dict[str, dace.dtypes.typeclass] = {
        name: dtype
        for name, dtype in original_types.items()
        if not sdfg.arrays[name].transient
    }

    # Initialize the inferred types dict with None, then overwrite with initial_types where given.
    inferred: Dict[str, Optional[dace.dtypes.typeclass]] = {
        name: None for name in sdfg.arrays
    }
    for name, dtype in initial_types.items():
        inferred[name] = dtype

    # Iteratively propagate types until convergence or max iterations reached.
    global_node_types: Dict[SDFGState, Dict] = {}
    for iteration in range(_MAX_FIXPOINT_ITERS):
        changed = False
        global_node_types.clear()

        # Need to snapshot inferred types at the start of each iteration to detect convergence
        snapshot = dict(inferred)
        for state in _states_in_order(sdfg):
            global_node_types[state] = _propagate_state(
                state,
                inferred,
                initial_types,
                supported,
                promotion_rules,
                original_types,
            )
        changed = inferred != snapshot

        if not changed:
            break
    else:
        raise RuntimeError(
            f"Type propagation did not converge after {_MAX_FIXPOINT_ITERS} iterations."
        )

    # Apply inferred dtypes to SDFG arrays.
    for name, dtype in inferred.items():
        if dtype is not None and sdfg.arrays[name].dtype != dtype:
            sdfg.arrays[name].dtype = dtype

    _apply_connector_types(global_node_types, supported)

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

    copy_in_state = sdfg.add_state_before(state=sdfg.start_block, label="copy_in")

    # TODO: This is currently inefficient, copies all changed inputs for each sink state.
    for sink in sdfg.sink_nodes():
        copy_out_state = sdfg.add_state_after(
            state=sink, label=f"copy_out_{sink.label}"
        )
        for orig_name, casted_name in repl_dict.items():
            if orig_name in sdfg_outputs:
                _add_copy_map(
                    copy_out_state,
                    casted_name,
                    sdfg.arrays[casted_name],
                    orig_name,
                    orig_descs[orig_name],
                )

    for orig_name, casted_name in repl_dict.items():
        if orig_name in sdfg_inputs:
            _add_copy_map(
                copy_in_state,
                orig_name,
                orig_descs[orig_name],
                casted_name,
                sdfg.arrays[casted_name],
            )
