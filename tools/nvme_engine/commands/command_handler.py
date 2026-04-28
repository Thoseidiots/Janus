"""
NVMe Command Handler — orchestrates admin and I/O command processing.
"""

from __future__ import annotations

from typing import Any, Dict

from nvme_engine.backends.base import StorageBackendOps
from nvme_engine.commands.admin_commands import AdminCommandProcessor, AdminCommandResult, AdminCommandType
from nvme_engine.commands.io_commands import IoCommandProcessor
from nvme_engine.commands.namespace_manager import NamespaceManager
from nvme_engine.models.io_models import IoCompletion, IoRequest


class NvmeCommandHandler:
    """Central handler that routes admin and I/O commands."""

    def __init__(self, backend: StorageBackendOps, namespace_manager: NamespaceManager) -> None:
        self._backend = backend
        self._namespace_manager = namespace_manager
        self._admin = AdminCommandProcessor()
        self._io = IoCommandProcessor(backend)

    def handle_admin_command(
        self, cmd_type: AdminCommandType, params: Dict[str, Any]
    ) -> AdminCommandResult:
        return self._admin.process(cmd_type, params)

    def handle_io_command(self, request: IoRequest) -> IoCompletion:
        return self._io.execute(request)

    def handle_atomic_write(self, request: IoRequest) -> IoCompletion:
        data = request.buffer or b"\x00" * (request.buffer_size or 512)
        return self._io.atomic_write(request.lba, data)
