"""ClientHandle: abstract handle to a spawned client process/actor."""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class ClientHandle(ABC, Generic[T]):
    """Handle to a spawned client process/actor.

    ``result()`` blocks until the body function returns.
    ``stop()`` disconnects the client and tears down the process/actor.
    """

    @abstractmethod
    def result(self, timeout: Optional[float] = None) -> T: ...

    @abstractmethod
    def stop(self) -> None: ...
