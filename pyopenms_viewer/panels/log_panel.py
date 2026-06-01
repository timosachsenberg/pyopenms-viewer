"""Algorithm log panel component.

Displays captured C++ stdout/stderr output from pyOpenMS algorithms.
"""

from datetime import datetime

from nicegui import ui

from pyopenms_viewer.core.state import ViewerState
from pyopenms_viewer.panels.base_panel import BasePanel


class LogPanel(BasePanel):
    """Algorithm log panel.

    Captures and displays diagnostic output from pyOpenMS algorithms
    that write to C-level stdout/stderr.
    """

    def __init__(self, state: ViewerState):
        super().__init__(state, "log", "Algorithm Log", "terminal")
        self._log_element = None

    def build(self, container: ui.element) -> ui.expansion:
        with container:
            self.expansion = ui.expansion(self.name, icon=self.icon, value=False).classes("w-full max-w-[1700px]")

            with self.expansion:
                with ui.row().classes("w-full justify-end mb-1"):
                    ui.button("Clear", icon="delete_outline", on_click=self._clear_log).props("flat dense size=sm")

                self._log_element = ui.log(max_lines=1000).classes("w-full h-64")

        # Populate with any existing log messages
        for msg in self.state.log_messages:
            self._log_element.push(msg)

        # Subscribe to new log output and data-clear events
        self.state.on_algorithm_log(self._on_algorithm_log)
        self.state.on_data_loaded(self._on_data_loaded)

        # Start hidden (auto-visibility)
        self.update_visibility()

        return self.expansion

    def update(self) -> None:
        pass

    def _has_data(self) -> bool:
        return len(self.state.log_messages) > 0

    def _on_algorithm_log(self, algo_name: str, output: str) -> None:
        if self._log_element is None:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        header = f"--- [{timestamp}] {algo_name} ---"
        self._log_element.push(header)
        for line in output.splitlines():
            self._log_element.push(line)
        self._log_element.push("")

        # Auto-expand and show panel when new output arrives
        self.update_visibility()
        if self.expansion is not None:
            self.expansion.value = True

    def _on_data_loaded(self, data_type: str) -> None:
        """React to data-clear events: if log_messages was emptied, sync the UI."""
        if data_type == "mzml" and not self.state.log_messages:
            if self._log_element is not None:
                self._log_element.clear()
            self.update_visibility()

    def _clear_log(self) -> None:
        self.state.log_messages.clear()
        if self._log_element is not None:
            self._log_element.clear()
        self.update_visibility()
