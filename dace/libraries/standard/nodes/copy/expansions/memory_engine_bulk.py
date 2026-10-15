# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Block-collective copy into shared memory.
"""
from typing import TYPE_CHECKING

import dace
from dace import library, nodes, dtypes
from dace.libraries.standard.nodes.copy.node import CopyLibraryNode
from dace.sdfg.scope import is_in_scope
from dace.transformation.transformation import ExpandTransformation
from dace.libraries.standard.nodes.copy.common import (_build_memory_engine_bulk_copy_code, INPUT_CONNECTOR_NAME,
                                                       OUTPUT_CONNECTOR_NAME, MEMORY_ENGINE_DESC_INPUT_NAME,
                                                       CUtensorMap)

if TYPE_CHECKING:
    pass


@library.register_expansion(CopyLibraryNode, "MemoryEngineBulk")
class ExpandMemoryEngineBulk(ExpandTransformation):
    """Memory engine based bulk copy.

    Currently only the TMA engine on NVIDIA Hopper+ GPUs are supported.

    Caller must place this outside any enclosing ``GPU_ThreadBlock`` map -- this expansion *is*
    the thread-block-level operation."""
    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):
        inp_name, inp, in_subset, out_name, out, out_subset = node.validate(parent_sdfg,
                                                                            parent_state,
                                                                            allow_cross_storage=True)

        valid_storages = [(dtypes.StorageType.GPU_Shared, dtypes.StorageType.GPU_Global),
                          (dtypes.StorageType.GPU_Global, dtypes.StorageType.GPU_Shared)]
        if (inp.storage, out.storage) not in valid_storages:
            raise ValueError(f"MemoryEngineBulk only supports GPU_Shared <-> GPU_Global copies"
                             f"(got {inp.storage} -> {out.storage}). Use MappedTasklet for "
                             "Shared <-> Register thread-level copies.")

        if is_in_scope(parent_sdfg, parent_state, node, [dtypes.ScheduleType.GPU_ThreadBlock]):
            raise ValueError("MemoryEngineBulk IS the thread-block-level operation "
                             "and must not be nested inside a GPU_ThreadBlock map.")

        if len(inp.shape) > 5:
            raise ValueError(f"MemoryEngineBulk only supports up to 5 dimensions (got {len(inp.shape)})")

        return nodes.Tasklet(node.name,
                             inputs={INPUT_CONNECTOR_NAME: dace.dtypes.pointer(inp.dtype),
                                     MEMORY_ENGINE_DESC_INPUT_NAME: dace.dtypes.pointer(CUtensorMap)},
                             outputs={OUTPUT_CONNECTOR_NAME: dace.dtypes.pointer(out.dtype)},
                             code=_build_memory_engine_bulk_copy_code(node, parent_state, inp, in_subset, out,
                                                                      out_subset),
                             language=dace.Language.CPP)

    @staticmethod
    def postprocessing(sdfg, state, expansion):
        # Add a method that traverses the graph backward to find first CPU scope.
        # There, insert a Tasklet that creates a TMA desc using dace::cuda::get_cuTensorMapEncodeTiled
        raise NotImplementedError("Implement postprocessing for MemoryEngineBulk")