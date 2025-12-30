"""Event bus for inter-component communication.

This module provides a simple publish-subscribe event system that allows
components to communicate without tight coupling. Components subscribe to
events and receive callbacks when those events are emitted.
"""

from enum import Enum, auto
from typing import Callable


class EventType(Enum):
    """Enumeration of all event types for type safety."""

    DATA_LOADED = auto()  # mzML, features, or IDs loaded
    VIEW_CHANGED = auto()  # zoom/pan changed
    SELECTION_CHANGED = auto()  # spectrum, feature, or ID selected
    DISPLAY_OPTIONS_CHANGED = auto()  # colormap, axis swap, etc.
    LOADING_STATE_CHANGED = auto()  # loading started/finished


class EventBus:
    """Simple publish-subscribe event bus for inter-component communication.

    Usage:
        bus = EventBus()
        bus.subscribe("view_changed", lambda **kw: print("View changed!"))
        bus.emit("view_changed")  # Triggers all subscribed callbacks

    Thread Safety:
        This implementation is NOT thread-safe. All subscriptions and emissions
        should happen on the same thread (typically the UI thread).
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, callback: Callable) -> Callable:
        """Subscribe to an event type.

        Args:
            event_type: String identifier for the event (e.g., "data_loaded")
            callback: Function to call when event is emitted. Receives **kwargs.

        Returns:
            The callback function (for easy unsubscribe later)
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        return callback

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe a callback from an event type.

        Args:
            event_type: String identifier for the event
            callback: The callback function to remove
        """
        if event_type in self._subscribers:
            self._subscribers[event_type] = [cb for cb in self._subscribers[event_type] if cb != callback]

    def emit(self, event_type: str, **kwargs) -> None:
        """Emit an event to all subscribers.

        Exceptions in callbacks are caught and printed to avoid breaking
        the event chain.

        Args:
            event_type: String identifier for the event
            **kwargs: Data to pass to subscribers
        """
        if event_type not in self._subscribers:
            return
        for callback in self._subscribers[event_type]:
            try:
                callback(**kwargs)
            except Exception as e:
                print(f"Event handler error for {event_type}: {e}")

    def clear(self, event_type: str | None = None) -> None:
        """Clear all subscribers for an event type, or all events if None.

        Args:
            event_type: Event to clear, or None to clear all
        """
        if event_type is None:
            self._subscribers.clear()
        elif event_type in self._subscribers:
            self._subscribers[event_type].clear()

    def has_subscribers(self, event_type: str) -> bool:
        """Check if an event type has any subscribers.

        Args:
            event_type: Event to check

        Returns:
            True if there are subscribers
        """
        return bool(self._subscribers.get(event_type))
