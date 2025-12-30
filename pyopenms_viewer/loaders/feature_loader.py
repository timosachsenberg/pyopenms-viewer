"""FeatureXML file loading and processing."""

from typing import Any

from pyopenms import FeatureMap, FeatureXMLFile

from pyopenms_viewer.core.state import ViewerState


def extract_feature_data(state: ViewerState) -> list[dict[str, Any]]:
    """Extract feature data for table display.

    Args:
        state: ViewerState with feature_map already loaded

    Returns:
        List of feature metadata dictionaries
    """
    if state.feature_map is None:
        return []

    data = []
    for idx, feature in enumerate(state.feature_map):
        rt = feature.getRT()
        mz = feature.getMZ()
        intensity = feature.getIntensity()
        charge = feature.getCharge()
        quality = feature.getOverallQuality()

        hulls = feature.getConvexHulls()
        rt_width = 0
        mz_width = 0
        if hulls:
            all_points = []
            for hull in hulls:
                points = hull.getHullPoints()
                all_points.extend([(p[0], p[1]) for p in points])
            if all_points:
                rt_coords = [p[0] for p in all_points]
                mz_coords = [p[1] for p in all_points]
                rt_width = max(rt_coords) - min(rt_coords)
                mz_width = max(mz_coords) - min(mz_coords)

        data.append(
            {
                "idx": idx,
                "rt": round(rt, 2),
                "mz": round(mz, 4),
                "intensity": f"{intensity:.2e}",
                "charge": charge if charge != 0 else "-",
                "quality": round(quality, 3) if quality > 0 else "-",
                "rt_width": round(rt_width, 2) if rt_width > 0 else "-",
                "mz_width": round(mz_width, 4) if mz_width > 0 else "-",
            }
        )

    return data


class FeatureLoader:
    """Loads featureXML files.

    Example:
        state = ViewerState()
        loader = FeatureLoader(state)
        if loader.load_sync("features.featureXML"):
            print(f"Loaded {len(state.feature_data)} features")
    """

    def __init__(self, state: ViewerState):
        """Initialize loader with state reference.

        Args:
            state: ViewerState instance to populate with data
        """
        self.state = state

    def load_sync(self, filepath: str) -> bool:
        """Load featureXML file synchronously.

        Args:
            filepath: Path to the featureXML file

        Returns:
            True if successful
        """
        try:
            self.state.feature_map = FeatureMap()
            FeatureXMLFile().load(filepath, self.state.feature_map)
            self.state.features_file = filepath
            self.state.selected_feature_idx = None
            self.state.feature_data = extract_feature_data(self.state)
            return True
        except Exception as e:
            print(f"Error loading features: {e}")
            return False
