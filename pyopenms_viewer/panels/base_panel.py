"""Base panel class for all UI panels.

All panels inherit from BasePanel and receive a reference to the shared
ViewerState. This ensures memory efficiency and consistent state management.
"""

from abc import ABC, abstractmethod

from nicegui import ui

from pyopenms_viewer.core.state import ViewerState


class BasePanel(ABC):
    """Abstract base class for all UI panels.

    Each panel:
    1. Receives a reference to ViewerState (shared, never copied)
    2. Creates its own UI elements in build()
    3. Subscribes to relevant events
    4. Updates its display when events fire

    Example:
        class MyPanel(BasePanel):
            def build(self, container):
                with container:
                    self.expansion = ui.expansion(self.name, icon=self.icon)
                    with self.expansion:
                        self.label = ui.label("Content here")
                self.state.on_data_loaded(self._on_data_loaded)
                return self.expansion

            def update(self):
                self.label.set_text(f"Data: {len(self.state.df)}")

            def _has_data(self):
                return self.state.df is not None
    """

    def __init__(
        self,
        state: ViewerState,
        panel_id: str,
        name: str,
        icon: str,
    ):
        """Initialize panel with state reference.

        Args:
            state: ViewerState instance (shared reference, not a copy)
            panel_id: Unique identifier for this panel
            name: Display name for the panel header
            icon: Material icon name for the panel
        """
        self.state = state
        self.panel_id = panel_id
        self.name = name
        self.icon = icon
        self.expansion: ui.expansion | None = None
        self._is_built = False

    @abstractmethod
    def build(self, container: ui.element) -> ui.expansion:
        """Build the panel UI inside the given container.

        Must:
        1. Create a ui.expansion with self.name and self.icon
        2. Subscribe to relevant events from self.state
        3. Return the expansion element

        Args:
            container: Parent element to build panel in

        Returns:
            The expansion element created
        """
        pass

    @abstractmethod
    def update(self) -> None:
        """Update the panel display based on current state.

        Called when relevant events fire. Should refresh any
        dynamic content based on self.state.
        """
        pass

    @abstractmethod
    def _has_data(self) -> bool:
        """Check if this panel has data to display.

        Used for auto-visibility mode.

        Returns:
            True if panel has displayable data
        """
        pass

    def should_be_visible(self) -> bool:
        """Determine if panel should be visible based on setting and data.

        Returns:
            True if panel should be visible
        """
        return self.state.should_panel_be_visible(self.panel_id)

    def set_visibility(self, visible: bool) -> None:
        """Set panel visibility.

        Args:
            visible: Whether panel should be visible
        """
        if self.expansion is not None:
            self.expansion.set_visibility(visible)

    def update_visibility(self) -> None:
        """Update visibility based on current state."""
        self.set_visibility(self.should_be_visible())


class PanelManager:
    """Manages panel ordering, visibility, and updates.

    Handles:
    - Registering panels
    - Updating panel order
    - Broadcasting visibility updates
    - Managing panel container
    """

    def __init__(self, state: ViewerState, container: ui.element):
        """Initialize panel manager.

        Args:
            state: ViewerState for visibility settings
            container: UI container holding all panels
        """
        self.state = state
        self.container = container
        self.panels: dict[str, BasePanel] = {}

    def register(self, panel: BasePanel) -> None:
        """Register a panel with the manager.

        Args:
            panel: Panel instance to register
        """
        self.panels[panel.panel_id] = panel
        self.state.panel_elements = {**self.state.panel_elements, panel.panel_id: panel.expansion}

    def update_visibility(self) -> None:
        """Update visibility of all registered panels."""
        for panel in self.panels.values():
            panel.update_visibility()

    def update_order(self) -> None:
        """Reorder panels according to state.panel_order."""
        for idx, panel_id in enumerate(self.state.panel_order):
            if panel_id in self.panels:
                panel = self.panels[panel_id]
                if panel.expansion is not None:
                    panel.expansion.move(target_index=idx)

    def update_all(self) -> None:
        """Update all panels."""
        for panel in self.panels.values():
            panel.update()
