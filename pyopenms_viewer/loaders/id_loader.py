"""idXML file loading and processing."""

from typing import Any

from pyopenms import IdXMLFile, PeptideIdentificationList

from pyopenms_viewer.core.state import ViewerState


def extract_id_data(state: ViewerState) -> list[dict[str, Any]]:
    """Extract peptide ID data for table display.

    Args:
        state: ViewerState with peptide_ids already loaded

    Returns:
        List of ID metadata dictionaries
    """
    if not state.peptide_ids:
        return []

    data = []
    idx = 0
    for pep_id in state.peptide_ids:
        rt = pep_id.getRT()
        mz = pep_id.getMZ()
        hits = pep_id.getHits()

        if hits:
            best_hit = hits[0]
            sequence = best_hit.getSequence().toString()
            score = best_hit.getScore()
            charge = best_hit.getCharge()
        else:
            sequence = "-"
            score = 0
            charge = 0

        data.append(
            {
                "idx": idx,
                "rt": round(rt, 2),
                "mz": round(mz, 4),
                "sequence": sequence[:30] + "..." if len(sequence) > 30 else sequence,
                "full_sequence": sequence,
                "charge": charge if charge != 0 else "-",
                "score": round(score, 4) if score != 0 else "-",
            }
        )
        idx += 1

    return data


def link_ids_to_spectra(state: ViewerState, rt_tolerance: float = 5.0, mz_tolerance: float = 0.5) -> None:
    """Link peptide IDs to spectra by matching RT and precursor m/z.

    Updates spectrum_data with ID info (sequence, score) for matching MS2 spectra.
    Also collects meta value keys from PeptideIdentification and PeptideHit.

    Args:
        state: ViewerState with peptide_ids and spectrum_data already loaded
        rt_tolerance: RT tolerance in seconds for matching
        mz_tolerance: m/z tolerance in Da for matching
    """
    if not state.peptide_ids or not state.spectrum_data:
        return

    # Collect unique meta value keys from all IDs
    meta_keys_set = set()
    for pep_id in state.peptide_ids:
        # Get PeptideIdentification meta values
        pid_keys = []
        pep_id.getKeys(pid_keys)
        for key in pid_keys:
            meta_keys_set.add(f"pid:{key.decode() if isinstance(key, bytes) else key}")

        # Get PeptideHit meta values
        hits = pep_id.getHits()
        if hits:
            hit_keys = []
            hits[0].getKeys(hit_keys)
            for key in hit_keys:
                meta_keys_set.add(f"hit:{key.decode() if isinstance(key, bytes) else key}")

    state.id_meta_keys = sorted(meta_keys_set)

    # Clear existing ID info from spectrum data
    for spec_row in state.spectrum_data:
        spec_row["sequence"] = "-"
        spec_row["full_sequence"] = ""
        spec_row["score"] = "-"
        spec_row["id_idx"] = None
        spec_row["hit_rank"] = "-"
        spec_row["all_hits"] = []
        # Initialize meta value fields
        for meta_key in state.id_meta_keys:
            spec_row[meta_key] = "-"

    # Build index of MS2 spectra by RT for faster matching
    ms2_spectra = []
    for spec_row in state.spectrum_data:
        if spec_row["ms_level"] > 1:
            ms2_spectra.append(spec_row)

    # For each ID, find matching spectrum
    for id_idx, pep_id in enumerate(state.peptide_ids):
        id_rt = pep_id.getRT()
        id_mz = pep_id.getMZ()

        hits = pep_id.getHits()
        if not hits:
            continue

        # Collect PeptideIdentification meta values
        pid_meta_values = {}
        pid_keys = []
        pep_id.getKeys(pid_keys)
        for key in pid_keys:
            key_str = key.decode() if isinstance(key, bytes) else key
            value = pep_id.getMetaValue(key)
            if isinstance(value, bytes):
                value = value.decode()
            elif isinstance(value, float):
                value = round(value, 4)
            pid_meta_values[f"pid:{key_str}"] = value

        # Collect data for all hits
        all_hits_data = []
        for hit_idx, hit in enumerate(hits):
            sequence = hit.getSequence().toString()
            score = hit.getScore()
            charge = hit.getCharge()

            # Collect hit-specific meta values
            hit_meta_values = dict(pid_meta_values)
            hit_keys = []
            hit.getKeys(hit_keys)
            for key in hit_keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                value = hit.getMetaValue(key)
                if isinstance(value, bytes):
                    value = value.decode()
                elif isinstance(value, float):
                    value = round(value, 4)
                hit_meta_values[f"hit:{key_str}"] = value

            all_hits_data.append(
                {
                    "sequence": sequence[:25] + "..." if len(sequence) > 25 else sequence,
                    "full_sequence": sequence,
                    "score": round(score, 4) if score != 0 else "-",
                    "charge": charge,
                    "hit_rank": hit_idx + 1,
                    "id_idx": id_idx,
                    "hit_idx": hit_idx,
                    "meta_values": hit_meta_values,
                }
            )

        # Find matching spectrum
        best_match = None
        best_rt_diff = float("inf")

        for spec_row in ms2_spectra:
            spec_rt = spec_row["rt"]
            spec_mz = spec_row.get("precursor_mz")

            # Skip if no precursor m/z
            if spec_mz == "-" or spec_mz is None:
                continue

            rt_diff = abs(spec_rt - id_rt)
            if rt_diff > rt_tolerance:
                continue

            mz_diff = abs(float(spec_mz) - id_mz)
            if mz_diff > mz_tolerance:
                continue

            # Find closest match
            if rt_diff < best_rt_diff:
                best_rt_diff = rt_diff
                best_match = spec_row

        # Update matching spectrum with ID info
        # Only update if this is a better match (closer RT) or no existing match
        if best_match is not None:
            existing_rt_diff = best_match.get("_rt_diff", float("inf"))
            if best_match["id_idx"] is None or best_rt_diff < existing_rt_diff:
                best_hit = all_hits_data[0]
                best_match["sequence"] = best_hit["sequence"]
                best_match["full_sequence"] = best_hit["full_sequence"]
                best_match["score"] = best_hit["score"]
                best_match["id_idx"] = id_idx
                best_match["hit_rank"] = best_hit["hit_rank"]
                best_match["all_hits"] = all_hits_data
                best_match["_rt_diff"] = best_rt_diff  # Track for comparison

                # Copy meta values
                for meta_key, value in best_hit["meta_values"].items():
                    best_match[meta_key] = value


class IDLoader:
    """Loads idXML files.

    Example:
        state = ViewerState()
        loader = IDLoader(state)
        if loader.load_sync("ids.idXML"):
            print(f"Loaded {len(state.id_data)} identifications")
    """

    def __init__(self, state: ViewerState):
        """Initialize loader with state reference.

        Args:
            state: ViewerState instance to populate with data
        """
        self.state = state

    def load_sync(self, filepath: str) -> bool:
        """Load idXML file synchronously.

        Args:
            filepath: Path to the idXML file

        Returns:
            True if successful
        """
        try:
            self.state.protein_ids = []
            self.state.peptide_ids = PeptideIdentificationList()
            IdXMLFile().load(filepath, self.state.protein_ids, self.state.peptide_ids)
            self.state.id_file = filepath
            self.state.selected_id_idx = None
            self.state.id_data = extract_id_data(self.state)
            # Link IDs to spectra
            link_ids_to_spectra(self.state)
            return True
        except Exception as e:
            print(f"Error loading IDs: {e}")
            return False
