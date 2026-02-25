"""Barrier abstraction for SPMD synchronization across cluster backends.

Provides a thin ABC so that body functions can call ``barrier.wait()``
without knowing whether they run under a multiprocessing cluster (shared-memory
``mp.Barrier``) or a Ray cluster (centralized Ray actor barrier).
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class Barrier(ABC):
    """Cross-process barrier for SPMD synchronization."""

    @abstractmethod
    def wait(self) -> None:
        """Block until all participants have called wait()."""
        ...

    def destroy(self) -> None:
        """Optional cleanup (e.g. tear down collective groups)."""


class MultiprocessingBarrier(Barrier):
    """Wraps ``mp.Barrier`` for multiprocessing cluster (shared memory).

    All participants in the same multiprocessing cluster share the same
    underlying ``mp.Barrier`` object, which is picklable across
    ``mp.get_context("spawn")`` processes.
    """

    def __init__(self, mp_barrier) -> None:
        self._barrier = mp_barrier

    def wait(self) -> None:
        self._barrier.wait()


class RayActorBarrier(Barrier):
    """Barrier backed by a centralized Ray actor.

    Each participant holds a ``(actor_handle, rank)`` pair.  Calling
    ``wait()`` sends a remote ``arrive`` call and then polls ``check``
    until all participants have arrived at the current generation.

    Advantages over ``ray.util.collective`` (Gloo):
    - No Gloo rendezvous — works immediately, no slow init phase.
    - The parent process can query ``progress()`` on the actor to see
      how many clients have reached each barrier phase.
    """

    _POLL_INTERVAL_S = 0.005  # 5 ms between polls

    def __init__(self, actor_handle, rank: int) -> None:
        import ray

        self._actor = actor_handle
        self._rank = rank
        self._ray = ray

    def wait(self) -> None:
        gen_ref = self._actor.arrive.remote(self._rank)
        target_gen = self._ray.get(gen_ref)
        while True:
            ready = self._ray.get(self._actor.check.remote(target_gen))
            if ready:
                return
            time.sleep(self._POLL_INTERVAL_S)


# ---------------------------------------------------------------------------
# Ray actor that implements the barrier coordinator
# ---------------------------------------------------------------------------

def create_ray_barrier_actor(world_size: int):
    """Create a _BarrierActor and return its handle.

    Must be called from a process with an active ``ray.init()``.
    """
    import ray

    @ray.remote(num_cpus=0)
    class _BarrierActor:
        """Centralized barrier coordinator.

        Tracks a monotonically increasing *generation* counter.  When all
        ``world_size`` participants have called ``arrive`` for the current
        generation, the generation advances and all waiters are released.

        The parent process can call ``progress()`` to get a snapshot of
        ``(generation, n_arrived)`` for live status display.
        """

        def __init__(self, world_size: int) -> None:
            self._world_size = world_size
            self._generation = 0
            self._arrived: set = set()

        def arrive(self, rank: int) -> int:
            """Mark *rank* as arrived.  Returns the generation to wait for."""
            target = self._generation + 1
            self._arrived.add(rank)
            if len(self._arrived) >= self._world_size:
                self._generation = target
                self._arrived.clear()
            return target

        def check(self, target_generation: int) -> bool:
            """Return True if the barrier has advanced past *target_generation*."""
            return self._generation >= target_generation

        def progress(self) -> Tuple[int, int, int]:
            """Return ``(generation, n_arrived, world_size)``."""
            return (self._generation, len(self._arrived), self._world_size)

    return _BarrierActor.remote(world_size)


# Keep the old Gloo-based barrier for backwards compatibility.
class RayCollectiveBarrier(Barrier):
    """Lazy-init barrier using ``ray.util.collective`` (Gloo backend).

    .. deprecated:: Use :class:`RayActorBarrier` instead.
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
