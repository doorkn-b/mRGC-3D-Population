"""
Helper functions for common hexagon-based calculations.
Based on equations: https://www.omnicalculator.com/math/hexagon
@author: M.L. Italiano
"""

import numpy as np
from typing import Tuple
import itertools
import matplotlib.pyplot as plt


def hex_coords(
    col: int,
    row: int,
    width: float,
    height: float,
) -> Tuple[int, int]:
    """
    Given the column, row and dimensions of a hexagon, returns the x-y
    coordinates of so using a axial coordinate system.
    Decides on orientation of hexagon (vertex at top or 90 deg) based on
    height vs. width.
    https://www.redblobgames.com/grids/hexagons/

        - `col`: column index of current hexagon,
        - `row`: row index of current hexagon,
        - `width`: x-width of hexagon (centre-to-centre in x-direction),
        - `height`: y-height of hexagon (centre-to-centre in y-direction),

    Returns cartesian (x, y) of current hexagon's centre.
    """

    pointy = True if height > width else False  # True  for our usage

    x = col * width
    y = -row * height

    if pointy:  # hexagon point up, not flat side
        if row & 1:
            x += width / 2  # Shift odd rows by half a column
    elif col & 1:
        y += height / 2

    return (x, y)


def hex_area_to_side(area: float) -> float:
    """
    Given a hexagonal area, returns the side, s, which is equal to the
    circumcircle radius, R.
    """
    return np.sqrt(area / (3 * np.sqrt(3) / 2))


def hex_area_to_apothem(area: float) -> float:
    """Given a hexagonal area, returns the apothem, a."""
    side = hex_area_to_side(area=area)
    apo = side * np.sqrt(3) / 2
    return apo


def hex_diagonal_to_apothem(diagonal: float) -> float:
    """Given a hexagonal (long-)diagonal, returns the corresponding apothem."""
    return diagonal * np.sqrt(3) / 4


def hex_apothem_to_diagonal(apothem: float) -> float:
    """Given a hexagonal apothem, returns the corresponding (long-)diagonal."""
    return 4 * apothem / np.sqrt(3)


def hex_area_to_diagonal(area: float) -> float:
    """Given a hexagonal area, returns the diagonal length, D."""
    a = hex_area_to_apothem(area=area)
    D = hex_apothem_to_diagonal(apothem=a)
    return D


def diagonal_to_hex_area(diagonal: float) -> float:
    """
    Given the diameter [vertex-vertex diagonal distance] of the dendritic tree,
    computes the area of said mosaic/hexagon.
    """
    # Convert D to apothem, a
    apo = hex_diagonal_to_apothem(diagonal=diagonal)
    # Convert apothem to side
    side = 2 / np.sqrt(3) * apo
    # Convert to area
    return 6 * np.sqrt(3) / 4 * side**2


def plot_hex_coords(i_max: int) -> None:

    coords = []

    separation = 1
    width = hex_diagonal_to_apothem(separation) * 2
    height = separation
    x_off = 0
    z = 0

    clrs = ["k", "r", "b", "g", "gold", "gray"]
    l_clrs = len(clrs)

    fig1 = plt.figure(1235, tight_layout=True, figsize=(20, 20))
    ax = fig1.add_subplot(111)

    for i in range(i_max):

        # Create list of col, row hexagon indices
        idxes = np.arange(-i, i + 1, 1)  # Integers of [-i, i]
        idxes = itertools.product(idxes, idxes)

        if i:  # Need to avoid re-doing previous idx combos

            idxes_old = np.arange(-(i - 1), i, 1)
            idxes_old = itertools.product(idxes_old, idxes_old)
            set_old = set(idxes_old)
            set_new = set(idxes)
            idxes = list(set_new - set_old)

        # Convert indices into x-y coordinates and add this to coords

        for idx in idxes:

            col = idx[0]
            row = idx[1]

            # if col == row: continue # [x,x] is not valid for x!=0

            x, y = hex_coords(col, row, width, height)

            coords.append([x + x_off, y, z])

            coords_NP = np.array(coords)

        X = coords_NP[:, 0]
        Y = coords_NP[:, 1]
        # Z = coords_NP[:, 2]

        ax.scatter(X, Y, c=clrs[i % l_clrs], marker="x")

    ax.set_xlabel("x (microns)", fontsize=39)
    ax.set_ylabel("y (microns)", fontsize=39)
    plt.show()


if __name__ == "__main__":
    plot_hex_coords(3)
