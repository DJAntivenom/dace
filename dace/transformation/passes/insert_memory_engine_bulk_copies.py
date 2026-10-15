# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pass replacing ``Default`` and ``Auto`` implementations of ``CopyLibraryNodes`` with ``MemoryEngineBulk`` where possible."""
from typing import TYPE_CHECKING, Any, Dict, Optional

from dace import properties
from dace.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation import pass_pipeline as ppl, transformation
from dace.transformation.dataflow import InLocalStorage, OutLocalStorage
from dace.transformation.passes import PatternMatchAndApply, PatternMatchAndApplyRepeated
from dace.transformation.passes.gpu_specialization.codegen_preprocess_passes import InferDefaultSchedulesAndStorages
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.dtypes import StorageType


@properties.make_properties
@transformation.explicit_cf_compatible
class InsertMemoryEngineBulkCopies(ppl.Pass):
    """Changes the implementation of ``CopyLibraryNode`` to ``MemoryEngineBulk`` if the input/output storage types allow.

    If a ``CopyLibraryNode`` copies data between ``GPU_Global`` and ``GPU_Shared`` a memory engine can be used.
    In this case, this pass sets the implementation of said node to ``MemoryEngineBulk``.

    It is recommended to use the provided ``Insert{Input,Output,}MemoryEngineBulkCopiesPipeline`` pipelines, which
    can be instantiated like regular classes. They contain multiple steps making the copies and storage types explicit.
    """

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & (ppl.Modifies.Nodes | ppl.Modifies.Edges)

    def depends_on(self):
        return {InsertExplicitCopies, InferDefaultSchedulesAndStorages}

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """Set the implementation of every supporting ``CopyLibraryNode`` to ``MemoryEngineBulk`` and add TMA descriptor nodes.

        :param sdfg: The SDFG to transform, recursively including nested SDFGs.
        :param pipeline_results: Results of previously applied passes (unused).
        :returns: The number of copy nodes inserted, or ``None`` if none.
        """
        count = 0
        for nsdfg in sdfg.all_sdfgs_recursive():
            for state in nsdfg.states():
                for node in state.nodes():
                    if isinstance(node, CopyLibraryNode):
                        count += 1 if self._replace_node(node, state) else 0
        return count if count > 0 else None

    def _replace_node(self, node: CopyLibraryNode, state: SDFGState) -> bool:
        """Set implementation of ``node``.

        :param node: The nodes whose implementation will be changed if connections allow it.
        :param state: The state containing the copy node. The owning SDFG is ``state.sdfg``.
        :returns: ``True`` if the node's implementation was changed, ``False`` otherwise.
        """
        if not isinstance(node, CopyLibraryNode):
            return False

        src_data = node.src_storage(state)
        dst_data = node.dst_storage(state)

        _ALLOWED_COMBOS = [(StorageType.GPU_Global, StorageType.GPU_Shared),
                           (StorageType.GPU_Shared, StorageType.GPU_Global)]
        if (src_data, dst_data) not in _ALLOWED_COMBOS:
            return False

        if node.implementation is not None and node.implementation not in ["Auto", "Default"]:
            return False

        node.implementation = "MemoryEngineBulk"

        return True


if TYPE_CHECKING:
    class InsertGlobalToSharedMemoryEngineBulkCopiesPipeline(ppl.Pipeline):
        """Inserts memory engine copies from ``GPU_Global`` to ``GPU_Shared``."""
        ...


    class InsertSharedToGlobalMemoryEngineBulkCopiesPipeline(ppl.Pipeline):
        """Inserts memory engine copies from ``GPU_Shared`` to ``GPU_Global``."""
        ...


    class InsertMemoryEngineBulkCopiesPipeline(ppl.Pipeline):
        """Inserts memory engine copies in both directions."""
        ...
else:
    InsertGlobalToSharedMemoryEngineBulkCopiesPipeline = lambda: ppl.Pipeline([InLocalStorage(),
                                                                               InsertExplicitCopies(),
                                                                               InferDefaultSchedulesAndStorages(),
                                                                               InsertMemoryEngineBulkCopies()])
    InsertSharedToGlobalMemoryEngineBulkCopiesPipeline = lambda: ppl.Pipeline([OutLocalStorage(),
                                                                               InsertExplicitCopies(),
                                                                               InferDefaultSchedulesAndStorages(),
                                                                               InsertMemoryEngineBulkCopies()])
    InsertMemoryEngineBulkCopiesPipeline = lambda: ppl.Pipeline([PatternMatchAndApplyRepeated([InLocalStorage(),
                                                                                               OutLocalStorage()]),
                                                                 InsertExplicitCopies(),
                                                                 InferDefaultSchedulesAndStorages(),
                                                                 InsertMemoryEngineBulkCopies()])
