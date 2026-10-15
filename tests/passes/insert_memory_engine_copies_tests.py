# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the ``InsertMemoryEngineBulkCopies`` pass."""
import copy as _copy
import os
import sys

import dace
import numpy as np
import pytest
from dace import nodes
from dace.memlet import Memlet
from dace.sdfg import utils as sdutils
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.transformation.dataflow import WarpTiling
from dace.transformation.dataflow.add_threadblock_map import AddThreadBlockMap
from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import AddThreadBlockMaps
from dace.transformation.passes.insert_memory_engine_bulk_copies import InsertMemoryEngineBulkCopiesPipeline

import tests.polybench

# The polybench programs import their ``polybench`` harness as a top-level module, which only
# resolves when run as scripts (own directory on sys.path); importing them as a package needs it too.
sys.path.append(os.path.dirname(tests.polybench.__file__))

from tests.polybench.gemm import gemm, init_array as _gemm_init_array


def _wcr_edges(sdfg):
    """``(state, edge)`` for every edge still carrying a WCR."""
    return [(st, e) for st in sdfg.all_states() for e in st.edges() if e.data.wcr is not None]


def _count_copy_nodes(sdfg):
    """Count CopyLibraryNode instances across all states (recursive)."""
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, CopyLibraryNode))


def _count_direct_copy_edges(sdfg):
    """Count AccessNode -> AccessNode non-empty edges (recursive)."""
    count = 0
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for e in state.edges():
                if (isinstance(e.src, nodes.AccessNode) and isinstance(e.dst, nodes.AccessNode)
                        and not e.data.is_empty()):
                    count += 1
    return count


def _assert_no_other_subset(sdfg: dace.SDFG) -> None:
    """Assert no data-movement memlet still carries an ``other_subset`` after copy-node insertion.

    View-defining (alias) edges are excluded: they reference the underlying buffer rather than
    moving data, so ``InsertExplicitCopies`` correctly leaves them direct with ``other_subset``
    intact (mirrors the pass's own view-edge skip); a copy would change the SDFG's semantics.
    """
    for nsdfg in sdfg.all_sdfgs_recursive():
        for state in nsdfg.states():
            for edge in state.edges():
                memlet = edge.data
                if memlet.is_empty():
                    continue
                if any(
                        isinstance(an, nodes.AccessNode) and isinstance(nsdfg.arrays[an.data], dace.data.View)
                        and sdutils.get_view_edge(state, an) is edge for an in (edge.src, edge.dst)):
                    continue
                assert memlet.other_subset is None, (
                    f"Memlet on edge {edge.src}->{edge.dst} in SDFG '{nsdfg.name}' still "
                    f"has other_subset={memlet.other_subset}; expected None after copy insertion.")


def _assert_no_copynd(sdfg: dace.SDFG) -> None:
    """Assert ``generate_code`` emits no ``dace::CopyND`` template instantiations."""
    sdfg.expand_library_nodes()
    for obj in sdfg.generate_code():
        code = obj.code if isinstance(obj.code, str) else getattr(obj.code, 'code', str(obj.code))
        assert 'CopyND<' not in code, f"unexpected CopyND in code object {obj.title}"


def _build_copy_sdfg(name, arrays, edge_memlet):
    """Build an SDFG with two AccessNodes wired by a single edge."""
    sdfg = dace.SDFG(name)
    for arr_name, shape, storage in arrays:
        sdfg.add_array(arr_name, shape, dace.float64, storage)
    st = sdfg.add_state("s")
    src = st.add_access(arrays[0][0])
    dst = st.add_access(arrays[1][0])
    st.add_edge(src, None, dst, None, edge_memlet)
    return sdfg, st, src, dst


def _assert_copy_storages(sdfg, src_storage, dst_storage):
    """Assert that every CopyLibraryNode in ``sdfg`` has the given storages."""
    found = False
    for n, parent in sdfg.all_nodes_recursive():
        if isinstance(n, CopyLibraryNode):
            assert n.src_storage(parent) == src_storage
            assert n.dst_storage(parent) == dst_storage
            found = True
    assert found, "No CopyLibraryNode found in SDFG"


def _assert_libnode_implementation(sdfg, none=False):
    """Assert that every (if none == False) or no CopyLibraryNode in ``sdfg`` has `"MemoryEngineBulk"` as implementation."""
    found = False
    for n, parent in sdfg.all_nodes_recursive():
        if isinstance(n, CopyLibraryNode):
            if none:
                assert n.implementation != "MemoryEngineBulk"
            else:
                assert n.implementation == "MemoryEngineBulk"
            found = True
    assert found, "No CopyLibraryNode found in SDFG"


def _compile_and_run(sdfg, **kwargs):
    sdfg.expand_library_nodes()
    exe = sdfg.compile()
    exe(**kwargs)


@pytest.mark.parametrize("storage",
                         [(dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared),
                          (dace.StorageType.GPU_Shared, dace.StorageType.GPU_Global)],
                         ids=["global_to_shared", "shared_to_global"])
def test_insert_gpu_1d(storage):
    """GPU_Global <-> GPU_Shared 1D copy."""
    sdfg, _, _, _ = _build_copy_sdfg("insert_gpu_1d", [("A", [100], storage[0]), ("B", [100], storage[1])],
                                     Memlet("A[10:60]", other_subset="20:70"))

    assert _count_direct_copy_edges(sdfg) == 1
    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert _count_direct_copy_edges(sdfg) == 0
    assert _count_copy_nodes(sdfg) == 1
    _assert_copy_storages(sdfg, storage[0], storage[1])
    _assert_libnode_implementation(sdfg)


@pytest.mark.parametrize("storage", set(dace.StorageType._member_map_.values()) - {dace.StorageType.GPU_Shared,
                                                                                   dace.StorageType.GPU_Global})
def test_insert_reject_cpu(storage):
    """Memory engine should only be used on GPU"""
    sdfg, _, _, _ = _build_copy_sdfg("insert_cpu_cpu_1d", [("A", [100], storage[0]), ("B", [100], storage[1])],
                                     Memlet("A[10:60]", other_subset="20:70"))

    assert _count_direct_copy_edges(sdfg) == 1
    inserted_count = InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})["InsertMemoryEngineBulkCopies"]
    assert inserted_count == 0
    assert _count_direct_copy_edges(sdfg) == 1
    assert _count_copy_nodes(sdfg) == 0
    _assert_copy_storages(sdfg, storage[0], storage[1])
    _assert_libnode_implementation(sdfg, none=True)


@pytest.mark.parametrize("storage",
                         [(dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared),
                          (dace.StorageType.GPU_Shared, dace.StorageType.GPU_Global)],
                         ids=["global_to_shared", "shared_to_global"])
def test_insert_gpu_2d_slice(storage):
    """GPU 2D slice copy with explicit other_subset."""
    sdfg, _, _, _ = _build_copy_sdfg("insert_gpu_2d", [("A", [10, 20], storage[0]), ("B", [10, 20], storage[1])],
                                     Memlet(data="A", subset="2:8, 5:15", other_subset="0:6, 0:10"))

    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert _count_direct_copy_edges(sdfg) == 0
    assert _count_copy_nodes(sdfg) == 1
    _assert_libnode_implementation(sdfg)


@pytest.mark.parametrize("sdfg_name,memlet", [
    ("insert_other_dst", Memlet(data="B", subset="0:8", other_subset="2:10")),
    ("insert_other_src", Memlet(data="A", subset="2:10", other_subset="0:8")),
],
                         ids=["data_is_dst", "data_is_src"])
@pytest.mark.parametrize("storage",
                         [(dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared),
                          (dace.StorageType.GPU_Shared, dace.StorageType.GPU_Global)],
                         ids=["global_to_shared", "shared_to_global"])
def test_insert_other_subset_data_convention(sdfg_name, memlet, storage):
    """Both memlet conventions yield the same copy ``_in=A[2:10]``, ``_out=B[0:8]`` with no ``other_subset``."""
    sdfg, st, _, _ = _build_copy_sdfg(sdfg_name, [("A", [20], storage[0]), ("B", [20], storage[1])], memlet)

    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert _count_copy_nodes(sdfg) == 1
    _assert_libnode_implementation(sdfg)

    for n in st.nodes():
        if isinstance(n, CopyLibraryNode):
            in_m = list(st.in_edges(n))[0].data
            out_m = list(st.out_edges(n))[0].data
            assert in_m.data == "A" and str(in_m.subset) == "2:10"
            assert in_m.other_subset is None
            assert out_m.data == "B" and str(out_m.subset) == "0:8"
            assert out_m.other_subset is None
            break


@pytest.mark.parametrize("storage",
                         [(dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared),
                          (dace.StorageType.GPU_Shared, dace.StorageType.GPU_Global)],
                         ids=["global_to_shared", "shared_to_global"])
def test_insert_gpu_full_array(storage):
    """Full array copy."""
    sdfg, _, _, _ = _build_copy_sdfg("insert_full", [("A", [64], storage[0]), ("B", [64], storage[1])],
                                     Memlet("A[0:64]"))

    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    _assert_libnode_implementation(sdfg)


def test_insert_multiple_copies_same_state():
    """Two copies in the same state: A->B and A->C."""
    sdfg = dace.SDFG("insert_multi")
    sdfg.add_array("A", [64], dace.float64, dace.StorageType.GPU_Global)
    for name in ("B", "C"):
        sdfg.add_array(name, [32], dace.float64, dace.StorageType.CPU_Heap)
    st = sdfg.add_state("s")
    a = st.add_access("A")
    b = st.add_access("B")
    c = st.add_access("C")
    st.add_edge(a, None, b, None, Memlet("A[0:32]"))
    st.add_edge(a, None, c, None, Memlet("A[32:64]"))

    result = InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})["InsertMemoryEngineBulkCopies"]
    _assert_no_other_subset(sdfg)
    _assert_libnode_implementation(sdfg)
    assert result == 2
    assert _count_copy_nodes(sdfg) == 2


def test_insert_no_copies_returns_none():
    """If there are no copy edges, return None."""
    sdfg = dace.SDFG("no_copies")
    sdfg.add_array("A", [10], dace.float64, dace.StorageType.CPU_Heap)
    st = sdfg.add_state("s")
    a = st.add_access("A")
    t = st.add_tasklet("noop", {"_in"}, {"_out"}, "_out = _in + 1")
    a2 = st.add_access("A")
    st.add_edge(a, None, t, "_in", Memlet("A[0]"))
    st.add_edge(t, "_out", a2, None, Memlet("A[0]"))

    result = InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    assert result["InsertMemoryEngineBulkCopies"] is None


@pytest.mark.parametrize("storage",
                         [(dace.StorageType.GPU_Global, dace.StorageType.GPU_Shared),
                          (dace.StorageType.GPU_Shared, dace.StorageType.GPU_Global)],
                         ids=["global_to_shared", "shared_to_global"])
def test_insert_validates_after_pass(storage):
    """SDFG passes validation after InsertExplicitCopies."""
    sdfg, _, _, _ = _build_copy_sdfg("validate_after", [("A", [100], storage[0]), ("B", [100], storage[1])],
                                     Memlet("A[0:100]"))

    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})
    _assert_no_other_subset(sdfg)
    _assert_libnode_implementation(sdfg)
    sdfg.validate()


def _make_view_round_trip_sdfg(name, *, dst_side=False):
    """Build a round-trip through ``A_view``, a 5x6 view of the 4x5x6 array ``A``.

    Source-side (default) flows ``A[1] -> A_view -> other``; dst-side flows
    ``other -> A_view -> A[1]`` (view aliases the write target).

    :returns: ``(sdfg, state, a, view, other)`` -- ``a`` is the 4x5x6 array, ``other`` the 5x6 one.
    """
    gpu_global = dace.StorageType.GPU_Global
    gpu_shared = dace.StorageType.GPU_Shared
    sdfg = dace.SDFG(name)
    sdfg.add_array("A", [4, 5, 6], dace.float64, storage=gpu_global)
    sdfg.add_view("A_view", [5, 6], dace.float64, storage=gpu_shared)
    sdfg.add_array("other", [5, 6], dace.float64, storage=gpu_global)
    st = sdfg.add_state("s")
    a, v, o = st.add_access("A"), st.add_access("A_view"), st.add_access("other")
    if dst_side:
        st.add_edge(o, None, v, None, Memlet("other[0:5, 0:6]"))
        st.add_edge(v, None, a, None, Memlet("A[1, 0:5, 0:6]"))
    else:
        st.add_edge(a, None, v, None, Memlet("A[1, 0:5, 0:6]"))
        st.add_edge(v, None, o, None, Memlet("A_view[0:5, 0:6]"))
    return sdfg, st, a, v, o


def test_insert_view_src_round_trip_lifts_movement_edge():
    """``A -> A_view -> sink``: alias edge kept, movement edge lifted to ``A -> A_view -> Copy -> sink``."""
    sdfg, st, a, v, out = _make_view_round_trip_sdfg("view_src_movement")
    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})
    sdfg.validate()

    assert v in st.nodes(), "the view must be preserved as a copy endpoint"
    assert _count_copy_nodes(sdfg) == 1
    a_out = list(st.out_edges(a))
    assert len(a_out) == 1 and a_out[0].dst is v, "alias edge A -> A_view must be untouched"
    v_out = list(st.out_edges(v))
    assert len(v_out) == 1 and isinstance(v_out[0].dst, CopyLibraryNode)
    assert isinstance(list(st.in_edges(out))[0].src, CopyLibraryNode)
    _assert_libnode_implementation(sdfg)


def _run_reshape_copy_test(prefix, src_shape, dst_shape):
    """Build ``A[full] -> B`` (no other_subset), lift, and assert the derived destination range spans all of ``B``."""
    sdfg, _, _, _ = _build_copy_sdfg(f"{prefix}_{len(src_shape)}_to_{len(dst_shape)}",
                                     [("A", src_shape, dace.StorageType.GPU_Global),
                                      ("B", dst_shape, dace.StorageType.GPU_Shared)],
                                     Memlet(data="A", subset=', '.join(f"0:{s}" for s in src_shape)))
    result = InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})["InsertMemoryEngineBulkCopies"]
    assert result is None, "Reshapes are not supported through memory engine"
    sdfg.validate()


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ([8, 12, 5, 3], [96, 5, 3]),  # collapse leading two: einsum_blas test_4x4 pattern
        ([8, 10, 12], [80, 12]),  # collapse leading two: einsum_blas test_3x2 pattern
        ([8, 12, 5, 3], [8, 60, 3]),  # collapse middle two
        ([2, 3, 4, 5], [6, 20]),  # double collapse: dims 0-1 and dims 2-3
        ([8, 12, 5, 3], [1440]),  # full flatten
    ])
def test_insert_consecutive_collapse_reshape(src_shape, dst_shape):
    """Collapsing contiguous source dims derives a full-destination subset, not the rank-mismatched ``src_subset``."""
    _run_reshape_copy_test("reshape_collapse", src_shape, dst_shape)


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ([80, 12], [8, 10, 12]),  # split leading dim
        ([96, 5, 3], [8, 12, 5, 3]),  # split leading dim
        ([1440], [8, 12, 5, 3]),  # full unflatten
        ([6, 20], [2, 3, 4, 5]),  # double split
    ])
def test_insert_consecutive_split_reshape(src_shape, dst_shape):
    """Inverse split case: a higher-rank destination by splitting source dims, via the same symmetric code path."""
    _run_reshape_copy_test("reshape_split", src_shape, dst_shape)


@pytest.mark.parametrize(
    "src_shape,dst_shape",
    [
        ([8, 1, 12], [8, 12]),  # squeeze a length-1 dim
        ([8, 12, 1, 5], [96, 5]),  # squeeze + collapse
        ([1, 96, 5, 3], [8, 12, 5, 3]),  # leading 1 + split
    ])
def test_insert_reshape_with_squeezed_ones(src_shape, dst_shape):
    """Unit-length dims on either side are ignored when matching a consecutive collapse or split."""
    _run_reshape_copy_test("reshape_squeeze", src_shape, dst_shape)


def test_insert_view_multiple_consumers_each_lifted():
    """Each movement edge off a multiply-consumed View is lifted; the View is kept."""
    sdfg, state, _, v, _ = _make_view_round_trip_sdfg("view_multiple_consumers")
    sdfg.add_array("also_reads", [5, 6], dace.float64, storage=dace.StorageType.GPU_Shared)
    state.add_edge(v, None, state.add_access("also_reads"), None, Memlet("A_view[0:5, 0:6]"))
    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg, {})
    assert v in state.nodes()
    assert _count_copy_nodes(sdfg) == 2
    sdfg.validate()
    _assert_libnode_implementation(sdfg)


# Polybench-derived tests: the pass must preserve numerical output on real programs.
# Init wrappers allocate the arrays and delegate to the canonical programs' ``init_array``.


def _run_and_compare(program, init_fn, check_arrays, sizes, name, backend):
    """Run a program before and after InsertExplicitCopies, asserting numerical correctness."""
    ref_arrays = init_fn(**sizes)

    sdfg_ref = program.to_sdfg(simplify=True)
    for state in sdfg_ref.states():
        for node in state.nodes():
            if isinstance(node, dace.nodes.MapEntry):
                node.schedule = dace.ScheduleType.GPU_Device
                node.gpu_block_size = (32, 32)
            elif isinstance(node, dace.nodes.AccessNode) and node.data in ref_arrays.keys():
                sdfg_ref.arrays[node.data].storage = dace.StorageType.GPU_Global

    AddThreadBlockMaps().apply_pass(sdfg_ref, {})
    sdfg_ref.save('temp.sdfg')

    ref_exe = sdfg_ref.compile()
    ref_exe(**{k: v for k, v in ref_arrays.items()}, **sizes)
    ref_values = {k: ref_arrays[k].copy() for k in check_arrays}

    sdfg_pass = _copy.deepcopy(sdfg_ref)
    InsertMemoryEngineBulkCopiesPipeline().apply_pass(sdfg_pass, {})
    sdfg_pass.save('temp2.sdfg')
    _assert_libnode_implementation(sdfg_pass)
    _assert_no_other_subset(sdfg_pass)
    sdfg_pass.expand_library_nodes()
    pass_exe = sdfg_pass.compile()
    pass_arrays = init_fn(**sizes)
    pass_exe(**{k: v for k, v in pass_arrays.items()}, **sizes)

    for arr_name in check_arrays:
        backend.testing.assert_allclose(pass_arrays[arr_name],
                                        ref_values[arr_name],
                                        rtol=1e-10,
                                        atol=1e-12,
                                        err_msg=f"{name}: array '{arr_name}' mismatch after pass")


def _init_gemm(NI, NJ, NK, backend):
    A = backend.empty((NI, NK), dtype=backend.float64)
    B = backend.empty((NK, NJ), dtype=backend.float64)
    C = backend.empty((NI, NJ), dtype=backend.float64)
    alpha = backend.empty(1, dtype=backend.float64)
    beta = backend.empty(1, dtype=backend.float64)
    _gemm_init_array(C, A, B, alpha, beta, NI, NJ, NK)
    return {"A": A, "B": B, "C": C, "alpha": alpha, "beta": beta}


def test_polybench_gemm():
    """``InsertMemoryEngineBulk`` preserves gemm output versus the untransformed reference."""
    cp = pytest.importorskip("cupy")
    _run_and_compare(gemm,
                     lambda **kwargs: _init_gemm(**kwargs, backend=cp),
                     ["C"],
                     {"NI": 32, "NJ": 64, "NK": 96},
                     "gemm",
                     cp)


if __name__ == "__main__":
    pytest.main([__file__])
