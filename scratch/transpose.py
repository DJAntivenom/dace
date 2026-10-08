import cupy as cp
import dace
from dace.sdfg.infer_types import set_default_schedule_and_storage_types
from dace.transformation.dataflow import InLocalStorage
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def transpose(A: dace.float64[M, N] @ dace.dtypes.StorageType.GPU_Global,
              B: dace.float64[N, M] @ dace.dtypes.StorageType.GPU_Global):
    for i, j in dace.map[0:M:16, 0:N:16] @ dace.dtypes.ScheduleType.GPU_Device:
        for bi, bj in dace.map[i:i + 16, j:j + 16] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
            if bj < N and bi < M:
                B[bj, bi] = A[bi, bj]


sdfg = transpose.to_sdfg()
filename = "transpose_no_copy.sdfgz"
sdfg.save(filename, compress=True)
print("Saved initial sdfg to " + filename)

sdfg.apply_transformations(InLocalStorage)

print(f"Inserted {InsertExplicitCopies().apply_pass(sdfg, {})} explicit copies")

filename = "transpose.sdfgz"
sdfg.save(filename, compress=True)
print("Saved copy-based sdfg to " + filename)

filename = "transpose_expanded.sdfgz"
set_default_schedule_and_storage_types(sdfg)
sdfg.expand_library_nodes()
sdfg.save(filename, compress=True)
print("Saved expanded sdfg to " + filename)

csdfg = sdfg.compile()

A = cp.arange(100, dtype=dace.float64.as_numpy_dtype()).reshape((20, 5))
B = cp.empty((5, 20), dtype=A.dtype)
csdfg(A=A, B=B, M=A.shape[0], N=A.shape[1])

print(cp.asnumpy(A))
print(cp.asnumpy(B))
