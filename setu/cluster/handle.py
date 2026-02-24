"""ClientHandle: abstract handle to a spawned client process/actor.

The body callable passed to ``spawn_client`` may be a plain function
(returns ``T``) **or** a generator (yields intermediate ``T`` values and
optionally returns a final ``T``).

Use ``next_result()`` to consume values one at a time — both yielded
intermediates and the final return value.  Raises ``StopIteration``
(no value) only when there is nothing left to consume.

``result()`` drains any unconsumed values and returns the last one.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class ClientHandle(ABC, Generic[T]):
    """Handle to a spawned client process/actor.

    ``next_result()`` returns the next value (yielded or final return).
    Raises ``StopIteration`` when no values remain.

    ``result()`` drains all values and returns the last one.

    ``stop()`` disconnects the client and tears down the process/actor.
    """

    @abstractmethod
    def next_result(self, timeout: Optional[float] = None) -> T:
        """Return the next value from the body.

        Returns yielded values first, then the final return value.
        Raises ``StopIteration`` when nothing remains.
        """
        ...

    @abstractmethod
    def result(self, timeout: Optional[float] = None) -> T:
        """Block until the body finishes and return the last value."""
        ...

    @abstractmethod
    def stop(self) -> None: ...
