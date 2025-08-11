from enum import StrEnum, auto


class AirfoilType(StrEnum):
    NACA_002 = auto()
    NACA_015 = auto()
    NACA_030 = auto()


def detect_airfoil_type(thickness_to_cord_ratio: float) -> str:
    """
    Detects the type of airfoil based on the image-detected thickness-to-chord ratio.

    Args:
        thickness_to_cord_ratio (float): The ratio of thickness to chord length of the airfoil.

    Returns:
        str: The type of airfoil ('thin', 'medium', 'thick').
    """
    if thickness_to_cord_ratio > 0.2:
        return AirfoilType.NACA_030
    elif 0.1 <= thickness_to_cord_ratio < 0.2:
        return AirfoilType.NACA_015
    else:
        return AirfoilType.NACA_002
