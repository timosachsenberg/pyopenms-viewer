"""Explorer-style file picker dialog for browsing server-side files.

Provides a Windows Explorer-like dialog with breadcrumb navigation,
detail columns (size, date, type), and favorites/recent quick access.
"""

import platform
from datetime import datetime
from pathlib import Path
from typing import Optional

from nicegui import events, ui

from pyopenms_viewer.components.file_picker_storage import (
    add_favorite,
    add_recent_directory,
    get_favorites,
    get_recent_directories,
    is_favorite,
    remove_favorite,
    save_last_directory,
)

# Map of file extensions to (icon, type label)
_FILE_TYPE_INFO: dict[str, tuple[str, str]] = {
    ".mzml": ("science", "mzML File"),
    ".featurexml": ("scatter_plot", "featureXML File"),
    ".idxml": ("fingerprint", "idXML File"),
    ".xml": ("code", "XML File"),
}


def _human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            if unit == "B":
                return f"{size_bytes} {unit}"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


class LocalFilePicker(ui.dialog):
    """A dialog for picking files from the local filesystem."""

    SUPPORTED_EXTENSIONS = {".mzml", ".featurexml", ".idxml", ".xml"}

    def __init__(
        self,
        directory: str,
        *,
        upper_limit: Optional[str] = ...,
        multiple: bool = True,
        show_hidden_files: bool = False,
    ) -> None:
        """Initialize the file picker dialog.

        Args:
            directory: Starting directory path
            upper_limit: Directory to stop at (None: no limit, ...: same as starting directory)
            multiple: Allow selecting multiple files
            show_hidden_files: Show hidden files (starting with .)
        """
        super().__init__()

        self.path = Path(directory).expanduser().resolve()
        if upper_limit is None:
            self.upper_limit = None
        else:
            self.upper_limit = Path(directory if upper_limit is ... else upper_limit).expanduser()
        self.show_hidden_files = show_hidden_files
        self._multiple = multiple
        self._sort_column = "name"
        self._sort_ascending = True

        with self, ui.card().classes("w-full").style("max-width: 1100px; min-height: 600px"):
            # Breadcrumb bar
            self._breadcrumb_row = ui.row().classes("w-full items-center gap-0 px-2 py-1").style(
                "background: #f5f5f5; border-radius: 6px; min-height: 40px; flex-wrap: nowrap; overflow-x: auto"
            )

            # Main content: sidebar + file grid
            with ui.row().classes("w-full gap-0").style("flex: 1; min-height: 0"):
                # Sidebar
                self._sidebar_container = (
                    ui.column()
                    .classes("gap-0 py-1")
                    .style(
                        "width: 200px; min-width: 200px; border-right: 1px solid #e0e0e0; "
                        "overflow-y: auto; max-height: 480px"
                    )
                )

                # File grid
                self.grid = (
                    ui.aggrid(
                        {
                            "columnDefs": [
                                {
                                    "field": "name",
                                    "headerName": "Name",
                                    "flex": 3,
                                    "minWidth": 250,
                                    "sortable": False,
                                },
                                {
                                    "field": "size",
                                    "headerName": "Size",
                                    "flex": 1,
                                    "minWidth": 90,
                                    "sortable": False,
                                    "cellStyle": {"textAlign": "right"},
                                },
                                {
                                    "field": "type",
                                    "headerName": "Type",
                                    "flex": 1,
                                    "minWidth": 110,
                                    "sortable": False,
                                },
                                {
                                    "field": "modified",
                                    "headerName": "Date Modified",
                                    "flex": 1,
                                    "minWidth": 140,
                                    "sortable": False,
                                },
                            ],
                            "rowSelection": {"mode": "multiRow" if multiple else "singleRow"},
                        },
                        html_columns=[0],
                    )
                    .classes("flex-grow")
                    .style("flex: 1; min-height: 400px")
                    .on("cellDoubleClicked", self._handle_double_click)
                    .on("columnHeaderClicked", self._handle_header_click)
                )

            # Footer
            with ui.row().classes("w-full justify-between items-center px-2"):
                self._selection_label = ui.label("").classes("text-caption text-grey")
                with ui.row():
                    ui.button("Cancel", on_click=self.close).props("outline")
                    self._ok_button = ui.button("Open", on_click=self._handle_ok)

            # Track selection changes for the label
            self.grid.on("selectionChanged", self._update_selection_label)

        self._navigate_to(self.path)

    def _navigate_to(self, path: Path) -> None:
        """Central navigation: update path, rebuild all UI components."""
        self.path = path.resolve()
        add_recent_directory(str(self.path))
        self._build_breadcrumbs()
        self._update_grid()
        self._build_sidebar()

    def _build_breadcrumbs(self) -> None:
        """Rebuild the breadcrumb bar from current path."""
        self._breadcrumb_row.clear()
        parts = self.path.parts

        # On Windows, parts[0] is like "C:\\"; on Unix it's "/"
        with self._breadcrumb_row:
            max_visible = 6
            if len(parts) <= max_visible:
                # Show all parts
                for i, part in enumerate(parts):
                    if i > 0:
                        ui.icon("chevron_right", size="xs").classes("text-grey mx-0")
                    target = Path(*parts[: i + 1])
                    # Use a default arg to capture the current value
                    ui.button(part, on_click=lambda _, t=target: self._navigate_to(t)).props(
                        "flat dense no-caps"
                    ).classes("px-1 min-w-0")
            else:
                # Show first part, ..., last 4 parts
                target_root = Path(parts[0])
                ui.button(parts[0], on_click=lambda _, t=target_root: self._navigate_to(t)).props(
                    "flat dense no-caps"
                ).classes("px-1 min-w-0")
                ui.icon("chevron_right", size="xs").classes("text-grey mx-0")

                # Collapsed middle as a dropdown menu
                with ui.button("...", on_click=None).props("flat dense no-caps").classes("px-1 min-w-0"):
                    with ui.menu() as menu:
                        for j in range(1, len(parts) - 4):
                            target = Path(*parts[: j + 1])
                            ui.menu_item(parts[j], on_click=lambda _, t=target, m=menu: (m.close(), self._navigate_to(t)))

                for i in range(len(parts) - 4, len(parts)):
                    ui.icon("chevron_right", size="xs").classes("text-grey mx-0")
                    target = Path(*parts[: i + 1])
                    ui.button(parts[i], on_click=lambda _, t=target: self._navigate_to(t)).props(
                        "flat dense no-caps"
                    ).classes("px-1 min-w-0")

    def _build_sidebar(self) -> None:
        """Rebuild the sidebar with quick access, favorites, and recent."""
        self._sidebar_container.clear()
        with self._sidebar_container:
            # Quick Access section
            ui.label("Quick Access").classes("text-caption text-bold text-grey px-3 py-1")
            self._sidebar_item("home", "Home", Path.home())
            self._sidebar_item("folder", "CWD", Path.cwd())

            desktop = Path.home() / "Desktop"
            if desktop.is_dir():
                self._sidebar_item("desktop_windows", "Desktop", desktop)

            # Windows drive letters
            if platform.system() == "Windows":
                try:
                    import win32api

                    drives = win32api.GetLogicalDriveStrings().split("\000")[:-1]
                    for drive in drives:
                        self._sidebar_item("storage", drive.rstrip("\\"), Path(drive))
                except Exception:
                    pass

            ui.separator().classes("my-1")

            # Favorites section
            ui.label("Favorites").classes("text-caption text-bold text-grey px-3 py-1")
            favorites = get_favorites()
            if favorites:
                for fav in favorites:
                    fav_path = Path(fav)
                    with ui.row().classes("w-full items-center px-1 gap-0"):
                        btn = ui.button(
                            fav_path.name or str(fav_path),
                            on_click=lambda _, p=fav_path: self._navigate_to(p),
                        ).props("flat dense no-caps").classes("flex-grow text-left min-w-0").style(
                            "justify-content: flex-start; overflow: hidden; text-overflow: ellipsis"
                        )
                        btn.tooltip(str(fav_path))
                        ui.button(
                            icon="close",
                            on_click=lambda _, p=str(fav_path): self._remove_favorite_and_rebuild(p),
                        ).props("flat dense round size=xs").classes("text-grey")
            else:
                ui.label("No favorites").classes("text-caption text-grey-5 px-3")

            # Add current dir to favorites button
            current_is_fav = is_favorite(str(self.path))
            if not current_is_fav:
                ui.button(
                    "Add current folder",
                    icon="star_outline",
                    on_click=lambda: self._add_favorite_and_rebuild(str(self.path)),
                ).props("flat dense no-caps size=sm").classes("px-3 text-primary")

            ui.separator().classes("my-1")

            # Recent section
            ui.label("Recent").classes("text-caption text-bold text-grey px-3 py-1")
            recent = get_recent_directories()
            if recent:
                for rec in recent[:8]:
                    rec_path = Path(rec)
                    btn = ui.button(
                        rec_path.name or str(rec_path),
                        on_click=lambda _, p=rec_path: self._navigate_to(p),
                    ).props("flat dense no-caps").classes("w-full text-left px-3 min-w-0").style(
                        "justify-content: flex-start; overflow: hidden; text-overflow: ellipsis"
                    )
                    btn.tooltip(str(rec_path))
            else:
                ui.label("No recent folders").classes("text-caption text-grey-5 px-3")

    def _sidebar_item(self, icon: str, label: str, path: Path) -> None:
        """Add a sidebar quick-access item."""
        is_active = self.path == path.resolve()
        btn = ui.button(
            label,
            icon=icon,
            on_click=lambda _, p=path: self._navigate_to(p),
        ).props("flat dense no-caps").classes(
            f"w-full text-left px-3 min-w-0 {'bg-blue-1' if is_active else ''}"
        ).style("justify-content: flex-start")
        btn.tooltip(str(path))

    def _add_favorite_and_rebuild(self, directory: str) -> None:
        """Add a favorite and rebuild sidebar."""
        add_favorite(directory)
        self._build_sidebar()

    def _remove_favorite_and_rebuild(self, directory: str) -> None:
        """Remove a favorite and rebuild sidebar."""
        remove_favorite(directory)
        self._build_sidebar()

    def _update_grid(self) -> None:
        """Update the grid with current directory contents."""
        try:
            paths = list(self.path.glob("*"))
        except PermissionError:
            paths = []

        if not self.show_hidden_files:
            paths = [p for p in paths if not p.name.startswith(".")]

        # Filter files to supported extensions, keep all directories
        paths = [p for p in paths if p.is_dir() or p.suffix.lower() in self.SUPPORTED_EXTENSIONS]

        # Build row data with metadata
        rows = []
        for p in paths:
            try:
                stat = p.stat()
                size_bytes = stat.st_size
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            except (OSError, ValueError):
                size_bytes = 0
                mod_time = "N/A"

            if p.is_dir():
                icon_html = '<span class="material-icons" style="font-size:18px;vertical-align:middle;color:#f9a825">folder</span>'
                rows.append(
                    {
                        "name": f'{icon_html} <strong>{p.name}</strong>',
                        "size": "",
                        "size_bytes": -1,
                        "type": "Folder",
                        "modified": mod_time,
                        "path": str(p),
                        "_is_dir": True,
                        "_name_lower": p.name.lower(),
                    }
                )
            else:
                ext = p.suffix.lower()
                icon_name, type_label = _FILE_TYPE_INFO.get(ext, ("description", f"{ext} File"))
                icon_html = f'<span class="material-icons" style="font-size:18px;vertical-align:middle;color:#1976d2">{icon_name}</span>'
                rows.append(
                    {
                        "name": f"{icon_html} {p.name}",
                        "size": _human_readable_size(size_bytes),
                        "size_bytes": size_bytes,
                        "type": type_label,
                        "modified": mod_time,
                        "path": str(p),
                        "_is_dir": False,
                        "_name_lower": p.name.lower(),
                    }
                )

        # Sort: directories always first, then by selected column
        rows = self._sort_rows(rows)

        # Prepend parent directory entry
        if (self.upper_limit is None and self.path != self.path.parent) or (
            self.upper_limit is not None and self.path != self.upper_limit
        ):
            icon_html = '<span class="material-icons" style="font-size:18px;vertical-align:middle;color:#f9a825">folder</span>'
            rows.insert(
                0,
                {
                    "name": f"{icon_html} <strong>..</strong>",
                    "size": "",
                    "size_bytes": -1,
                    "type": "",
                    "modified": "",
                    "path": str(self.path.parent),
                    "_is_dir": True,
                    "_name_lower": "",
                },
            )

        self.grid.options["rowData"] = rows
        self.grid.update()

    def _sort_rows(self, rows: list[dict]) -> list[dict]:
        """Sort rows with directories always first, then by the selected column."""
        dirs = [r for r in rows if r["_is_dir"]]
        files = [r for r in rows if not r["_is_dir"]]

        key_map = {
            "name": lambda r: r["_name_lower"],
            "size": lambda r: r["size_bytes"],
            "type": lambda r: r["type"].lower(),
            "modified": lambda r: r["modified"],
        }
        key_fn = key_map.get(self._sort_column, key_map["name"])
        dirs.sort(key=key_fn, reverse=not self._sort_ascending)
        files.sort(key=key_fn, reverse=not self._sort_ascending)
        return dirs + files

    def _handle_header_click(self, e: events.GenericEventArguments) -> None:
        """Handle column header click for sorting."""
        col_id = e.args.get("column", {}).get("colId", "")
        if col_id in ("name", "size", "type", "modified"):
            if self._sort_column == col_id:
                self._sort_ascending = not self._sort_ascending
            else:
                self._sort_column = col_id
                self._sort_ascending = True
            self._update_grid()

    def _handle_double_click(self, e: events.GenericEventArguments) -> None:
        """Handle double-click on a row."""
        self.path = Path(e.args["data"]["path"])
        if self.path.is_dir():
            self._navigate_to(self.path)
        else:
            save_last_directory(str(self.path.parent))
            self.submit([str(self.path)])

    async def _handle_ok(self) -> None:
        """Handle OK button click."""
        rows = await self.grid.get_selected_rows()
        # Check if a single directory is selected — navigate into it
        if len(rows) == 1 and Path(rows[0]["path"]).is_dir():
            self._navigate_to(Path(rows[0]["path"]))
            return
        # Filter out directories
        selected = [r["path"] for r in rows if not Path(r["path"]).is_dir()]
        if selected:
            save_last_directory(str(self.path))
            self.submit(selected)
        else:
            ui.notify("Please select at least one file", type="warning")

    async def _update_selection_label(self, _e: events.GenericEventArguments = None) -> None:
        """Update the selection count label."""
        rows = await self.grid.get_selected_rows()
        file_count = sum(1 for r in rows if not Path(r["path"]).is_dir())
        if file_count:
            self._selection_label.text = f"{file_count} file{'s' if file_count != 1 else ''} selected"
            self._ok_button.text = f"Open ({file_count})"
        else:
            self._selection_label.text = ""
            self._ok_button.text = "Open"
