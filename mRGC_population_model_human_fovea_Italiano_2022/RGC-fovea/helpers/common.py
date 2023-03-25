"""Helper functions that have use across multiple different files."""

import os
import errno

# Common conversions
# Based on Watson 2014's linear fits (excellent up to ~13mm/50deg)
ECC_MM_TO_DEG = 3.731  # 3.731 deg/mm
ECC_DEG_TO_MM = 0.268  # 0.268 mm/deg

# Common waveform abbreviations
SIN = "SINUSOIDAL"
EXP = "EXPONENTIAL"
BIPHASIC = "BIPHASIC (STANDARD)"
ANODIC = "ANODIC-FIRST"
TRIANGLE = "CENTERED-TRIANGULAR"


def create_dir(sub_dir: str, get_cwd: bool = True) -> None:
    """
    Builds subdirec structure if need be.
        - `sub_dir`: path to directory of interest.
        - `get_cwd`: pre-pends current working directory to `sub_dir`.
    """
    path = f"{os.getcwd()}/{sub_dir}" if get_cwd else sub_dir
    if os.path.isdir(path) is False:
        os.makedirs(path)


def silent_remove(filename: str) -> None:
    """
    Removes a file denoted by `filename` (if present). Silences errors related
    to a non-existent file, raises other errors.
    """
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


def within_ellipse(
    x: float, y: float, x_ell: float, y_ell: float, r_x: float, r_y: float
) -> bool:
    """
    Returns a boolean as to whether the point (`x`, `y`) is within the ellipse
    defined by the mid-point (`x_ell`, `y_ell`) and radii `r_x`, `r_y`.
    """
    return ((x - x_ell) ** 2 / r_x ** 2 + (y - y_ell) ** 2 / r_y ** 2) <= 1
