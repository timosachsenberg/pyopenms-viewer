"""Theoretical spectrum generation for peptide annotation."""

from dataclasses import dataclass

from pyopenms import AASequence, MSSpectrum, TheoreticalSpectrumGenerator


@dataclass
class TheoreticalIon:
    """A single theoretical ion with m/z, name, type, and optional intensity."""

    mz: float
    name: str
    ion_type: str
    intensity: float = 1.0  # Default to 1.0 (equal intensity) if not predicted


@dataclass
class TheoreticalSpectrum:
    """Complete theoretical spectrum with all ion types."""

    ions: list[TheoreticalIon]
    sequence: str
    charge: int

    def get_ions_by_type(self, ion_type: str) -> list[TheoreticalIon]:
        """Get all ions of a specific type."""
        return [ion for ion in self.ions if ion.ion_type == ion_type]

    @property
    def b_ions(self) -> list[TheoreticalIon]:
        return self.get_ions_by_type("b")

    @property
    def y_ions(self) -> list[TheoreticalIon]:
        return self.get_ions_by_type("y")

    @property
    def a_ions(self) -> list[TheoreticalIon]:
        return self.get_ions_by_type("a")


def generate_theoretical_spectrum(
    sequence: AASequence,
    charge: int,
    add_isotopes: bool = False,
) -> TheoreticalSpectrum:
    """Generate theoretical b/y/a ion spectrum for annotation.

    Uses pyOpenMS TheoreticalSpectrumGenerator to create a theoretical
    spectrum with b, y, and a ions for the given peptide sequence.

    Args:
        sequence: AASequence object representing the peptide
        charge: Precursor charge state
        add_isotopes: Whether to add isotope peaks (default: False)

    Returns:
        TheoreticalSpectrum containing all generated ions with m/z, names,
        types, and intensities
    """
    tsg = TheoreticalSpectrumGenerator()
    spec = MSSpectrum()

    # Configure for b, y, and a ions
    params = tsg.getParameters()
    params.setValue("add_b_ions", "true")
    params.setValue("add_y_ions", "true")
    params.setValue("add_a_ions", "true")
    params.setValue("add_c_ions", "false")
    params.setValue("add_x_ions", "false")
    params.setValue("add_z_ions", "false")
    params.setValue("add_metainfo", "true")
    if add_isotopes:
        params.setValue("add_isotopes", "true")
    tsg.setParameters(params)

    # Generate spectrum with charge states up to min(charge, 2)
    tsg.getSpectrum(spec, sequence, 1, min(charge, 2))

    ions = []

    # Get ion names from string data arrays
    ion_names = []
    string_arrays = spec.getStringDataArrays()
    for arr in string_arrays:
        arr_name = arr.getName()
        if arr_name == "IonNames" or arr_name == b"IonNames":
            for name in arr:
                if isinstance(name, bytes):
                    ion_names.append(name.decode("utf-8", errors="ignore"))
                else:
                    ion_names.append(name)
            break

    for i in range(len(spec)):
        mz = spec[i].getMZ()
        intensity = spec[i].getIntensity()
        ion_name = ion_names[i] if i < len(ion_names) else ""

        if ion_name.startswith("b"):
            ion_type = "b"
        elif ion_name.startswith("y"):
            ion_type = "y"
        elif ion_name.startswith("a"):
            ion_type = "a"
        elif ion_name:
            ion_type = "other"
        else:
            continue  # Skip peaks without ion names

        ions.append(
            TheoreticalIon(
                mz=mz,
                name=ion_name,
                ion_type=ion_type,
                intensity=intensity if intensity > 0 else 1.0,
            )
        )

    return TheoreticalSpectrum(
        ions=ions,
        sequence=sequence.toString(),
        charge=charge,
    )
