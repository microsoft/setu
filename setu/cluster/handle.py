"""ClientHandle: abstract handle to a spawned client process/actor.

The body callable passed to ``spawn_client`` may be a plain function
(returns ``T``) **or** a generator (yields intermediate ``T`` values and
optionally returns a final ``T``).

Use ``next_result()`` to consume values one at a time — both yielded
intermediates and the final return value.  Raises ``StopIteration``
(no value) only when there is nothing left to consume.

``result()`` drains any unconsumed values and returns the last one.
"""

import inspect
import time
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, TypeVar

from setu.logger import init_logger

T = TypeVar("T")

_logger = init_logger(__name__)


def execute_client_body(
    body: Callable,
    client,
    participant,
    put_result: Callable,
    put_error: Callable[[str], None],
    put_done: Callable[[], None],
) -> None:
    """Run *body(client, participant)*, forwarding results via callbacks.

    Handles both plain functions and generators.  On any exception the
    traceback string is forwarded via *put_error*.  *put_done* is **always**
    called (even after errors) so the parent side never hangs.

    Args:
        body: The callable to execute (may be a generator function).
        client: The Client instance passed as the first arg to *body*.
        participant: The Participant passed as the second arg to *body*.
        put_result: Called with each yielded/returned value.
        put_error: Called with a traceback string on failure.
        put_done: Called exactly once when the body is finished (success or
            failure).  Must be safe to call after *put_error*.
    """
    t_start = time.monotonic()
    _logger.debug(
        "execute_client_body: starting body=%s participant=%s",
        getattr(body, "__name__", body), participant,
    )
    try:
        t_call = time.monotonic()
        ret = body(client, participant)
        is_gen = inspect.isgenerator(ret)
        _logger.debug(
            "execute_client_body: body returned in %.3fs, is_generator=%s",
            time.monotonic() - t_call, is_gen,
        )
        if is_gen:
            yield_count = 0
            try:
                while True:
                    t_next = time.monotonic()
                    value = next(ret)
                    yield_count += 1
                    _logger.debug(
                        "execute_client_body: yield #%d took %.3fs",
                        yield_count, time.monotonic() - t_next,
                    )
                    put_result(value)
            except StopIteration as e:
                _logger.debug(
                    "execute_client_body: generator exhausted after %d yields "
                    "(final next() took %.3fs), return value is %s",
                    yield_count, time.monotonic() - t_next,
                    "present" if e.value is not None else "None",
                )
                if e.value is not None:
                    put_result(e.value)
        else:
            if ret is not None:
                put_result(ret)
    except Exception:
        tb = traceback.format_exc()
        _logger.error("execute_client_body: body raised:\n%s", tb)
        put_error(tb)
    finally:
        _logger.debug(
            "execute_client_body: done, total=%.3fs", time.monotonic() - t_start
        )
        put_done()


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
