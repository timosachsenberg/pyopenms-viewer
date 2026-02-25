"""Data loaders for mzML, featureXML, idXML, and related formats."""

# AlphaRaw loader - optional, requires alpharaw package (install with: uv sync --extra vendor)
try:
    from pyopenms_viewer.loaders.alpharaw_loader import AlphaRawLoader, is_vendor_file
    ALPHARAW_AVAILABLE = True
except ImportError:
    ALPHARAW_AVAILABLE = False
    AlphaRawLoader = None

    def is_vendor_file(x: str) -> bool:  # noqa: ARG001
        return False

from pyopenms_viewer.loaders.chromatogram_loader import extract_chromatograms
from pyopenms_viewer.loaders.feature_loader import FeatureLoader, extract_feature_data
from pyopenms_viewer.loaders.id_loader import IDLoader, extract_id_data, link_ids_to_spectra
from pyopenms_viewer.loaders.ion_mobility_loader import extract_ion_mobility_data
from pyopenms_viewer.loaders.mzml_loader import MzMLLoader, get_cv_from_spectrum
from pyopenms_viewer.loaders.spectrum_extractor import extract_spectrum_data

__all__ = [
    "MzMLLoader",
    "AlphaRawLoader",
    "FeatureLoader",
    "IDLoader",
    "is_vendor_file",
    "ALPHARAW_AVAILABLE",
    "get_cv_from_spectrum",
    "extract_feature_data",
    "extract_id_data",
    "link_ids_to_spectra",
    "extract_chromatograms",
    "extract_ion_mobility_data",
    "extract_spectrum_data",
]
