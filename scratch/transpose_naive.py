import dace

from dace.transformation.dataflow.tiling import MapTiling

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def transpose(A: dace.float64[N, M] @ dace.dtypes.StorageType.CPU_Heap,
              B: dace.float64[M, N]):
    for i, j in dace.map[0:N, 0:M] @ dace.dtypes.ScheduleType.CPU_Multicore:
        B[i, j] = A[j, i]


filenames = ["transpose_naive.sdfgz", "transpose_naive_tiled.sdfgz"]

sdfg = transpose.to_sdfg()

sdfg.apply_transformations(MapTiling, options={"tile_sizes": [16, 16]}, print_report=True)
sdfg.save(filenames[1], compress=True)
print("Saved file to " + filenames[1])
