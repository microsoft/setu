"""ClientHandle: abstract handle to a spawned client process/actor.

The body callable passed to ``spawn_client`` is a plain function that
runs to completion and returns ``T``.

``result()`` blocks until the body finishes and returns its value.
``stop()`` disconnects the client and tears down the process/actor.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class ClientHandle(ABC, Generic[T]):
    """Handle to a spawned client process/actor.

    ``result()`` blocks until the body finishes and returns its value.
    ``stop()`` disconnects the client and tears down the process/actor.
    """

    @abstractmethod
    def result(self, timeout: Optional[float] = None) -> T:
        """Block until the body finishes and return the value."""
        ...

    @abstractmethod
    def stop(self) -> None: ...
