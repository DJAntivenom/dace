// Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

#ifndef __DACE_MEMORYENGINE_CUH
#define __DACE_MEMORYENGINE_CUH

#ifdef __CUDACC__
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#error "Not implemented for AMD GPUs"
#endif

#include <cassert>

#include "../reduction.h"
#include "../types.h"
#include "../vector.h"
#include "cudacommon.cuh"

namespace dace {
namespace cuda {
template <typename State>
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled(State* __state) {
  static constexpr unsigned int MIN_DRIVER_VERSION = 12000;
  static constexpr auto FUNCTION_NAME = "cuTensorMapEncodeTiled";
  int version;
  DACE_GPU_CHECK(cudaDriverGetVersion(&version));
  if (version < MIN_DRIVER_VERSION) {
    printf("Driver version (%d) too old for TMA, requires at least %d\n",
           version, MIN_DRIVER_VERSION);
    assert(false);
    return nullptr;
  }

  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  DACE_GPU_CHECK(cudaGetDriverEntryPointByVersion(
      FUNCTION_NAME, &cuTensorMapEncodeTiled_ptr, MIN_DRIVER_VERSION,
      cudaEnableDefault, &driver_status));
  switch (driver_status) {
    case cudaDriverEntryPointSuccess:
      break;
    case cudaDriverEntryPointSymbolNotFound:
      printf("%s not found in driver\n", FUNCTION_NAME);
      assert(false);
      return nullptr;
    case cudaDriverEntryPointVersionNotSufficent:
      printf("Driver version insufficient for %s (got %d, need %u)\n",
             FUNCTION_NAME, version, MIN_DRIVER_VERSION);
      assert(false);
      return nullptr;
    default:
      printf("Unknown error: %d\n", static_cast<int>(driver_status));
      assert(false);
      return nullptr;
  }

  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(
      cuTensorMapEncodeTiled_ptr);
}
}  // namespace cuda
}  // namespace dace

#endif  // __DACE_CUDACOPY_CUH
