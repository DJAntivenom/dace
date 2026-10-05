import dace
from dace.transformation.dataflow import InLocalStorage
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def transpose(A: dace.float64[N, M] @ dace.dtypes.StorageType.GPU_Global,
              B: dace.float64[M, N] @ dace.dtypes.StorageType.GPU_Global):
    for i, j in dace.map[0:N:16, 0:M:16] @ dace.dtypes.ScheduleType.GPU_Device:
        for bi, bj in dace.map[i:i + 16, j:j + 16] @ dace.dtypes.ScheduleType.GPU_ThreadBlock:
            B[bi, bj] = A[bj, bi]


sdfg = transpose.to_sdfg()
filename = "transpose_no_copy.sdfgz"
sdfg.save(filename, compress=True)
print("Saved initial sdfg to " + filename)

sdfg.apply_transformations(InLocalStorage)

print(f"Inserted {InsertExplicitCopies().apply_pass(sdfg, {})} explicit copies")

filename = "transpose.sdfgz"
sdfg.save(filename, compress=True)
print("Saved copy-based sdfg to " + filename)

csdfg = sdfg.compile()
