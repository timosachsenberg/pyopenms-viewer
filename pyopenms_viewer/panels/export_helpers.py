"""Shared download/export helpers for panels (TSV tables and PNG images).

These wrap the small amount of NiceGUI/JavaScript glue that several panels
need to trigger a browser download, so each panel only has to supply its own
data, columns, and filename.
"""

from nicegui import ui


def build_tsv(rows: list[dict], columns: list[dict]) -> str:
    """Build TSV text (a header row followed by one row per record).

    Args:
        rows: Row dicts keyed by each column's ``field``.
        columns: Column specs, each with ``field`` and ``label`` keys.

    Returns:
        Tab-separated text. ``None`` becomes empty; floats are rendered with
        ``.4f`` below 1000 (in absolute value) and ``.2e`` otherwise.
    """
    column_fields = [col["field"] for col in columns]
    column_labels = [col["label"] for col in columns]

    lines = ["\t".join(column_labels)]  # Header row
    for row in rows:
        values = []
        for field in column_fields:
            val = row.get(field, "")
            # Convert None to empty string, format numbers
            if val is None:
                val = ""
            elif isinstance(val, float):
                val = f"{val:.4f}" if abs(val) < 1000 else f"{val:.2e}"
            else:
                val = str(val)
            values.append(val)
        lines.append("\t".join(values))

    return "\n".join(lines)


def export_rows_as_tsv(data: list[dict], columns: list[dict], filename: str) -> None:
    """Build a TSV from ``data``/``columns`` and trigger a browser download."""
    if not data:
        ui.notify("No data to export", type="warning")
        return

    tsv_content = build_tsv(data, columns)

    # Escape backticks for JavaScript template literal
    escaped_content = tsv_content.replace("`", "\\`")

    # Trigger download using JavaScript
    ui.run_javascript(f"""
        const blob = new Blob([`{escaped_content}`], {{type: "text/tab-separated-values"}});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "{filename}";
        a.click();
        URL.revokeObjectURL(url);
    """)
    ui.notify(f"Exported {len(data)} rows", type="positive")


def save_image_element_as_png(image_element, filename: str) -> None:
    """Trigger a browser download of a NiceGUI image element's base64 PNG data."""
    if image_element is None:
        ui.notify("No image to save", type="warning")
        return

    # Get current image source (base64 data URL)
    src = image_element._props.get("src", "")
    if not src or not src.startswith("data:image/png;base64,"):
        ui.notify("No image data available", type="warning")
        return

    # Trigger download via JavaScript
    ui.run_javascript(f'''
        const link = document.createElement("a");
        link.href = "{src}";
        link.download = "{filename}";
        link.click();
    ''')
    ui.notify(f"Downloading {filename}", type="positive")
