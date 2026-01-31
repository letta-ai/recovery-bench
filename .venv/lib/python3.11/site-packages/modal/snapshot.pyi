import modal._object
import modal.client
import modal.object
import typing
import typing_extensions

class _SandboxSnapshot(modal._object._Object):
    """> Sandbox memory snapshots are in **early preview**.

    A `SandboxSnapshot` object lets you interact with a stored Sandbox snapshot that was created by calling
    `._experimental_snapshot()` on a Sandbox instance. This includes both the filesystem and memory state of
    the original Sandbox at the time the snapshot was taken.
    """
    @staticmethod
    async def from_id(sandbox_snapshot_id: str, client: typing.Optional[modal.client._Client] = None):
        """Construct a `SandboxSnapshot` object from a sandbox snapshot ID."""
        ...

class SandboxSnapshot(modal.object.Object):
    """> Sandbox memory snapshots are in **early preview**.

    A `SandboxSnapshot` object lets you interact with a stored Sandbox snapshot that was created by calling
    `._experimental_snapshot()` on a Sandbox instance. This includes both the filesystem and memory state of
    the original Sandbox at the time the snapshot was taken.
    """
    def __init__(self, *args, **kwargs):
        """mdmd:hidden"""
        ...

    class __from_id_spec(typing_extensions.Protocol):
        def __call__(self, /, sandbox_snapshot_id: str, client: typing.Optional[modal.client.Client] = None):
            """Construct a `SandboxSnapshot` object from a sandbox snapshot ID."""
            ...

        async def aio(self, /, sandbox_snapshot_id: str, client: typing.Optional[modal.client.Client] = None):
            """Construct a `SandboxSnapshot` object from a sandbox snapshot ID."""
            ...

    from_id: __from_id_spec
