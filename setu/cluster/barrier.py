"""Barrier abstraction for SPMD synchronization across cluster backends.

Provides a thin ABC so that body functions can call ``barrier.wait()``
without knowing whether they run under SingleNodeCluster (shared-memory
``mp.Barrier``) or a Ray cluster (``ray.util.collective`` Gloo barrier).
"""

from abc import ABC, abstractmethod


class Barrier(ABC):
    """Cross-process barrier for SPMD synchronization."""

    @abstractmethod
    def wait(self) -> None:
        """Block until all participants have called wait()."""
        ...

    def destroy(self) -> None:
        """Optional cleanup (e.g. tear down collective groups)."""


class MultiprocessingBarrier(Barrier):
    """Wraps ``mp.Barrier`` for SingleNodeCluster (shared memory).

    All participants in the same SingleNodeCluster share the same
    underlying ``mp.Barrier`` object, which is picklable across
    ``mp.get_context("spawn")`` processes.
    """

    def __init__(self, mp_barrier) -> None:
        self._barrier = mp_barrier

    def wait(self) -> None:
        self._barrier.wait()


class RayCollectiveBarrier(Barrier):
    """Lazy-init barrier using ``ray.util.collective`` (Gloo backend).

    Stores ``(world_size, rank, group_name)`` as plain ints/strings --
    trivially serializable by Ray.  Calls ``init_collective_group()``
    on the **first** ``wait()`` inside the Ray actor process, as
    required by the Ray collective API.
    """

    def __init__(self, world_size: int, rank: int, group_name: str) -> None:
        self._world_size = world_size
        self._rank = rank
        self._group_name = group_name
        self._initialized = False

    def wait(self) -> None:
        if not self._initialized:
            import ray.util.collective as col

            col.init_collective_group(
                self._world_size,
                self._rank,
                backend="gloo",
                group_name=self._group_name,
            )
            self._initialized = True

        import ray.util.collective as col

        col.barrier(self._group_name)

    def destroy(self) -> None:
        if self._initialized:
            import ray.util.collective as col

            col.destroy_collective_group(self._group_name)
            self._initialized = False
