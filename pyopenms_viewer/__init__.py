"""
pyopenms-viewer: Fast mzML peak map viewer using NiceGUI, Datashader, and pyOpenMS.

Designed to handle 50+ million peaks with smooth zooming and panning.
"""

__version__ = "0.2.0"

from pyopenms_viewer.cli import main
from pyopenms_viewer.core.events import EventBus
from pyopenms_viewer.core.state import ViewerState

__all__ = ["ViewerState", "EventBus", "main", "__version__"]
