# Copyright 2024-2026 ETH Zurich and the FP-Arena authors. All rights reserved.
"""
In-place floating-point type conversion for SDFGs.

:func:`change_fptype` rewrites the data descriptors of an SDFG from one
floating-point type to another (e.g. ``float64`` -> ``float32`` or
``float64`` -> ``fp_arena.float64sr``), recursing into nested SDFGs and
following views. Optionally it keeps the external interface unchanged by
inserting copy-in/copy-out casting maps, so a converted SDFG remains a
drop-in replacement for the original at its boundary.

This is the core "FP typecast transformation" used to retarget an existing
pipeline onto a new FP type for rapid prototyping.
"""

import copy
from typing import Dict, Set, Union

import dace


def _is_read(sdfg: dace.SDFG, name: str) -> bool:
    """Whether ``sdfg`` reads ``name``: an access node it leaves, or a reduction (``wcr``) into it."""
    return any(
        (node.data == name and state.out_degree(node) > 0) for state in sdfg.states() for node in state.data_nodes()
    ) or any(edge.data.data == name and edge.data.wcr is not None for state in sdfg.states() for edge in state.edges())


def _is_written(sdfg: dace.SDFG, name: str) -> bool:
    """Whether ``sdfg`` writes ``name``: an access node entered by an edge."""
    return any(node.data == name and state.in_degree(node) > 0 for state in sdfg.states() for node in state.data_nodes())


def _get_array_view_paths(sdfg: dace.SDFG, arrays_to_replace: Set[str]) -> Dict[str, Set[str]]:
    """
    Find, for each array to replace, the set of views reachable from it. Views
    must adopt the new type as well, otherwise their dtype would diverge.

    :param sdfg: the SDFG to scan.
    :param arrays_to_replace: array names whose dependent views are collected.
    :returns: mapping from array name to the set of dependent view names.
    """
    # TODO: Inefficient implementation, improve it later.
    view_sets = {name: set() for name in arrays_to_replace}
    for state in sdfg.states():
        for node in state.nodes():
            for node2 in state.nodes():
                if node == node2:
                    continue
                if (isinstance(node, dace.nodes.AccessNode) and isinstance(node2, dace.nodes.AccessNode)
                        and isinstance(sdfg.arrays[node.data], dace.data.Data)
                        and isinstance(sdfg.arrays[node2.data], dace.data.View) and node.data in arrays_to_replace):
                    paths_iterator = state.all_simple_paths(node, node2)
                    if any(True for _ in paths_iterator):
                        view_sets[node.data].add(node2.data)
    return view_sets


def _change_fp_type_recursive(sdfg: dace.SDFG, src_fptype: dace.dtypes.typeclass, dst_fptype: dace.dtypes.typeclass,
                              arrays_to_replace: Union[Set[str], None]) -> Set[str]:
    """
    Recursively change the FP type of the selected arrays (or all arrays of
    ``src_fptype`` when ``arrays_to_replace`` is ``None``), descending into
    nested SDFGs through their connectors.

    :param sdfg: the SDFG to modify in place.
    :param src_fptype: the source floating-point type to replace.
    :param dst_fptype: the destination floating-point type.
    :param arrays_to_replace: array names to replace, or ``None`` for all.
    :returns: the names of ``sdfg``'s containers whose type changed (the views and copies they reach included).
    """
    # Collect all views that depend on the arrays and add them to the set.
    if arrays_to_replace is not None:
        view_sets = _get_array_view_paths(sdfg, arrays_to_replace)
        for _, view_set in view_sets.items():
            arrays_to_replace = arrays_to_replace.union(view_set)
        cur_arrays_to_replace = arrays_to_replace
    else:
        cur_arrays_to_replace = {k for k, v in sdfg.arrays.items() if v.dtype == src_fptype}

    # Change FP-types of the arrays (and views that depend on them).
    for name in cur_arrays_to_replace:
        sdfg.arrays[name].dtype = dst_fptype

    # A container copied directly to or from a replaced one (an access node to an access node, no tasklet
    # in between to cast) must take the new type too; follow such copies to a fixpoint. Only replaced
    # arrays spread the type: an unrelated array that already had ``dst_fptype`` changes nothing.
    changed = True
    while changed:
        changed = False
        for state in sdfg.states():
            for edge in state.edges():
                ends = (edge.src, edge.dst)
                if not all(isinstance(end, dace.nodes.AccessNode) for end in ends):
                    continue
                names = {end.data for end in ends}
                if names & cur_arrays_to_replace and not names <= cur_arrays_to_replace:
                    for name in names - cur_arrays_to_replace:
                        if sdfg.arrays[name].dtype == src_fptype:
                            sdfg.arrays[name].dtype = dst_fptype
                    cur_arrays_to_replace = cur_arrays_to_replace | names
                    changed = True

    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.NestedSDFG):
                # Continue changing FP-types depending on the connectors.
                if arrays_to_replace is not None:
                    new_replacements = set()
                    for ie in state.in_edges(node):
                        if ie.data is not None and ie.data.data in cur_arrays_to_replace:
                            new_replacements.add(ie.dst_conn)

                    for oe in state.out_edges(node):
                        if oe.data is not None and oe.data.data in cur_arrays_to_replace:
                            new_replacements.add(oe.src_conn)

                    _change_fp_type_recursive(node.sdfg, src_fptype, dst_fptype, new_replacements)
                else:
                    _change_fp_type_recursive(node.sdfg, src_fptype, dst_fptype, None)
    return cur_arrays_to_replace


def change_fptype(sdfg: dace.SDFG,
                  src_fptype: dace.dtypes.typeclass,
                  dst_fptype: dace.dtypes.typeclass,
                  cast_in_and_out_data: bool = False,
                  arrays_to_replace: Union[Set[str], None] = None):
    """
    Change the floating-point type of arrays in ``sdfg`` from ``src_fptype`` to
    ``dst_fptype`` in place.

    :param sdfg: the SDFG to modify.
    :param src_fptype: the source floating-point type to replace.
    :param dst_fptype: the destination floating-point type.
    :param cast_in_and_out_data: if ``True``, keep the external interface in
        ``src_fptype`` and insert copy-in/copy-out casting maps so the converted
        arrays become transients; if ``False``, convert the interface as well.
    :param arrays_to_replace: array names to replace, or ``None`` to replace
        every array of ``src_fptype``.
    :raises ValueError: if a named array does not have ``src_fptype``.
    """
    # If arrays_to_replace is None, all arrays are replaced.
    if arrays_to_replace is not None:
        # Check the types match.
        for name in arrays_to_replace:
            arr = sdfg.arrays[name]
            if arr.dtype != src_fptype:
                raise ValueError(f"Array {name} from the passed inputs has fptype {arr.dtype} but function was "
                                 f"provided src fp type ({src_fptype})")

    cur_arrays_to_replace = _change_fp_type_recursive(sdfg, src_fptype, dst_fptype, arrays_to_replace)

    # Replace all occurrences of the arrays in the SDFG with their replaced
    # counterparts, keeping the original interface via copy-in/copy-out.
    if cast_in_and_out_data is True:
        # If we cast in data, the interface stays the same, we add new arrays,
        # e.g. A -> A_<dst_fptype.to_string()>.
        array_name_mapping = {name: f"{name}_{dst_fptype.to_string()}" for name in sorted(cur_arrays_to_replace)}

        # Read / written before renaming: a read-only argument must not be copied back out (that would
        # round the caller's data), a write-only one needs no copy in unless it is only written in part.
        read = {name for name in array_name_mapping if _is_read(sdfg, name)}
        written = {name for name in array_name_mapping if _is_written(sdfg, name)}

        # Names are local to an SDFG: a nested SDFG keeps its connectors and inner arrays, only the
        # outer containers are renamed.
        sdfg.replace_dict(array_name_mapping)

        # Copy-in and copy-out extensions. For all arrays that we made
        # transient, ensure we add the appropriate copy-in and copy-out nodes.
        copy_in_state = sdfg.add_state_before(state=sdfg.start_block, label="copy_in")
        last_blocks = [node for node in sdfg.nodes() if sdfg.out_degree(node) == 0]  # Only last block has no successors
        assert len(last_blocks) == 1
        last_block = last_blocks[0]
        copy_out_state = sdfg.add_state_after(state=last_block, label="copy_out")

        # Add a new array descriptor for each transient array, and add copy-in or copy-out.
        src_dst_pairs = []
        for original_arr_name, casted_arr_name in array_name_mapping.items():
            arr = sdfg.arrays[casted_arr_name]
            original_arr_desc = copy.deepcopy(arr)  # Transientness does not change for this array
            arr.transient = True  # Always transient due to copy-in
            original_arr_desc.dtype = src_fptype  # Change back to the original fp type
            sdfg.add_datadesc(name=original_arr_name, datadesc=original_arr_desc, find_new_name=False)
            src_dst_pairs.append(((casted_arr_name, arr), (original_arr_name, original_arr_desc)))

        def _add_copy_map(state: dace.SDFGState, src_arr_name: str, src_arr: dace.data.Data, dst_arr_name: str,
                          dst_arr: dace.data.Data):
            """
            Add a map (or scalar tasklet) to ``state`` that casts every element
            of ``src_arr`` into ``dst_arr``.

            :param state: the state to add the copy into.
            :param src_arr_name: name of the source array.
            :param src_arr: source data descriptor.
            :param dst_arr_name: name of the destination array.
            :param dst_arr: destination data descriptor.
            """
            assert src_arr.shape == dst_arr.shape, "Source and destination arrays must have the same shape."
            # Add a tasklet that performs the type cast.
            tasklet = state.add_tasklet(name=f"copy_{src_arr_name}_to_{dst_arr_name}",
                                        inputs={"in"},
                                        outputs={"out"},
                                        code=f"out = static_cast<{dst_arr.dtype.ctype}>(in);",
                                        language=dace.Language.CPP)

            if isinstance(src_arr, dace.data.Array):
                assert isinstance(dst_arr, dace.data.Array)
                # Create a new map node.
                map_ranges = dict()
                for dim, size in enumerate(src_arr.shape):
                    map_ranges[f"i{dim}"] = f"0:{size}"

                map_entry, map_exit = state.add_map(name=f"copy_map_{src_arr_name}_to_{dst_arr_name}",
                                                    ndrange=map_ranges)

                # Add access nodes for source and destination arrays.
                src_access = state.add_access(src_arr_name)
                dst_access = state.add_access(dst_arr_name)

                # Add edges from the map to the access nodes, caring about the connector.
                state.add_edge(src_access, None, map_entry, f"IN_{src_arr_name}",
                               dace.memlet.Memlet.from_array(src_arr_name, src_arr))
                state.add_edge(map_exit, f"OUT_{dst_arr_name}", dst_access, None,
                               dace.memlet.Memlet.from_array(dst_arr_name, dst_arr))
                map_entry.add_in_connector(f"IN_{src_arr_name}")
                map_entry.add_out_connector(f"OUT_{src_arr_name}")
                map_exit.add_in_connector(f"IN_{dst_arr_name}")
                map_exit.add_out_connector(f"OUT_{dst_arr_name}")
                access_str = ", ".join([str(s) for s in map_ranges.keys()])
                state.add_edge(map_entry, f"OUT_{src_arr_name}", tasklet, "in",
                               dace.Memlet(expr=f"{src_arr_name}[{access_str}]"))
                state.add_edge(tasklet, "out", map_exit, f"IN_{dst_arr_name}",
                               dace.Memlet(expr=f"{dst_arr_name}[{access_str}]"))
            else:
                assert isinstance(src_arr, dace.data.Scalar)
                assert isinstance(dst_arr, dace.data.Scalar)
                src_access = state.add_access(src_arr_name)
                dst_access = state.add_access(dst_arr_name)
                state.add_edge(src_access, None, tasklet, "in", dace.Memlet(expr=f"{src_arr_name}[0]"))
                state.add_edge(tasklet, "out", dst_access, None, dace.Memlet(expr=f"{dst_arr_name}[0]"))

        for (transient_arr_name, transient_arr), (nontransient_arr_name, nontransient_arr) in src_dst_pairs:
            # No need to copy from transient to transient.
            if transient_arr.transient and nontransient_arr.transient:
                continue
            if nontransient_arr_name in read or nontransient_arr_name in written:
                # A write-only array is copied in too: its writes may cover it only in part.
                _add_copy_map(state=copy_in_state,
                              src_arr_name=nontransient_arr_name,
                              src_arr=nontransient_arr,
                              dst_arr_name=transient_arr_name,
                              dst_arr=transient_arr)
            if nontransient_arr_name in written:
                _add_copy_map(state=copy_out_state,
                              src_arr_name=transient_arr_name,
                              src_arr=transient_arr,
                              dst_arr_name=nontransient_arr_name,
                              dst_arr=nontransient_arr)
