# -*- coding: utf-8 -*-
"""

Creates patches of the (para-)foveal region as a function of eccentricity. 
Positive x direction is nasal; negative x is temporal. 
Positive y direction is superior; negative y is inferior. 

@author: M.L. Italiano
"""

import os
import sys
import copy
import pickle
import numpy as np
import shutil
import time
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import scipy.interpolate as si
from scipy.stats import skewnorm, norm
from scipy.spatial import distance
import itertools
from typing import Tuple, Union, List
from joblib import Parallel, delayed
import multiprocessing

import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "":
    print("INFO: No display found. Using non-interactive Agg backend")
    mpl.use("Agg")  # avoids issue with non-interactive displays as w/ EC2

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.ticker as mticker

from primate_cell_generator import mass_generate, create_tile
from helpers.common import create_dir, ECC_MM_TO_DEG, ECC_DEG_TO_MM, within_ellipse
from helpers.hex_calculations import (
    hex_coords,
    hex_diagonal_to_apothem,
    diagonal_to_hex_area,
)

plt.style.use("./helpers/plots.mplstyle")
plt.set_cmap("inferno")
cmap = plt.cm.get_cmap("inferno")

num_cores = multiprocessing.cpu_count()

############################## PARAMETERS #####################################

TRIAL = 0  # Trial number for record-keeping and reproducibility
JITTER = True  # Incorporate variance into RGC soma positions

ECC = -1.00  # eccentricity along horizontal meridian [mm]
ECC_Y = 0.00  # eccentricity along vertical meridian [mm]
PATCH_SIZE = 0.070  # Length of patch square [mm] (typically 0.600)
SHORT_AX = True  # Cut axons short for faster run-times
SHORT_AX_LEN = 1250  # Length to cut axons at [um]

CROP_POPULATION = False  # Cut population at defined limits to confine area/no.
CROP_X = [-500, 350]  # left, right limit
CROP_Y = [-250, 500]  # bottom, top limit

# If True, cell positions assume the origin is the foveal centre,
# If False, the input eccentricity (ECC, ECC_Y) is assumed to the origin.
# NOTE: only ever tested and used with DISPLACE=False -> use with care if True
DISPLACE = False

PLOT = True
PARALLEL = True  # perform population generation in parallel where possible
WARN_USER = True  # command-line prompt/warning if there is pre-existing data

###############################################################################
######## Anatomical parameters (should be left [relatively] unchanged) ########
###############################################################################

PIT = 0.15 / 2  # Eccentricity of pit   (radius) [mm]
SLOPE = 0.35 / 2  # Eccentricity of slope (radius) [mm]

OND_X = 1.77 / 2  # Optic nerve disc horizontal radius [mm]
OND_Y = 1.88 / 2  # Optic nerve disc vertical   radius [mm]
OND_EDGE = 3.4  # nasal (+) mm from fovea to OND edge
OND_POS = [3.4 + OND_X, 0]  # Position of OND midpoint
OND = OND_POS[0]  # X-Eccentricity of OND centre [mm]

SOMA_D = 12e-3  # Diameter of soma [mm]
BUFFER = SOMA_D * 0.33  # Buffer between soma size for Muller cells etc.
SOMA_BIN = SOMA_D * 1e3 * 1.33  # bin width for hexbin/histplots [um]

L_HIL = 40  # Length of axon hillock [um]
L_AIS = 40  # Length of AIS [um]
L_SD = 30  # No. points per soma-dendrite segment [unitless]

RES_AX_1 = 0.75  # [um]
RES_AX_2 = 1.5  # [um]

Z_BRANCH = 5  # z-height over which dendrites branch out [um]

MGC_proportion = 0.95  # human MGC:RGC for para-foveal region (Dacey 1993)
CF = 1  # Coverage factor for dendritic trees
# ON_OFF = 1 / 1.7  # no. of ON:OFF (beyond central is 1:1.7 (Dacey 1993))
# D_ON_OFF = 1.3 / 1  # Diameter of ON:OFF (beyond central is 1.3:1) (Dacey 1992/3)
ON_OFF = 1 / 1  # no. ON:OFF, ~equal in central (beyond central is 1:1.7 (Dacey 1993))
D_ON_OFF = 1 / 1  # Diameter of ON vs. OFF (beyond central is 1.3 : 1)
GCL_MAX = 6  # Max no. of GC layers

MIN_DEND = 5  # Minimum dendritic tree diameter [micron]

# for investigating morphological factors and/or reducing computational demand
AXON = True  # toggle creation of axons
TEST_SMALL_SOMA = False  # True sets .hoc soma diameter mean to 9 um (not 12)
NO_DENDRITES = False  # True reduces dendritic tree to 0.1 um diameter

###############################################################################
################## Interpolation-based limits for calculations ################
###############################################################################

# Based on interpolation limits of the various .csv distributions/profiles
VALID_RANGE = [-2.49, 2.49]  # used to validate experimental setup

###############################################################################
## Seeding random generator and establishing eccentricity-dependent profiles ##
###############################################################################

# Seed the random generator
np.random.seed(seed=111)

LYR_MAX = GCL_MAX  # edited according to eccentricity's true max
TXT_PATH = ""  # edited by main

# Create distributions based on literature
root = os.path.abspath(os.getcwd()).split("/RGC")[0]
root_distr = f"{root}/RGC-fovea/"

NF = pd.read_csv(f"{root_distr}/distributions/Nasal_foveal.csv")
NF_ecc = np.squeeze(np.array([round(NF["Eccentricity (mm)"], 4)]))
NF_GC = np.squeeze(np.array([(1000 * round(NF["GC (x1000) / mm^2"], 4))]))

TF = pd.read_csv(f"{root_distr}/distributions/Temporal_foveal.csv")
TF_ecc = -np.squeeze(np.array([round(TF["Eccentricity (mm)"], 4)]))
TF_GC = np.squeeze(np.array([(1000 * round(TF["GC (x1000) / mm^2"], 4))]))

GCL = pd.read_csv(f"{root_distr}/distributions/GCL_Thickness.csv")
GCL_ecc = np.squeeze(np.array([round(GCL["Eccentricity"], 3)]))
GCL_t = np.squeeze(np.array([round(GCL["Thickness"], 3)]))

IPL = pd.read_csv(f"{root_distr}/distributions/IPL_Thickness.csv")
IPL_ecc = np.squeeze(np.array([round(IPL["Eccentricity"], 3)]))
IPL_t = np.squeeze(np.array([round(IPL["Thickness"], 3)]))

# NOTE: MGC.csv is not for human, it is for marmoset ... this distribution
# is not used. Instead, a value of 95% has been adopted for 0-3mm based on
# Dacey 1993
MGC = pd.read_csv(f"{root_distr}/distributions/Midget_Prop.csv")
MGC_ecc = np.squeeze(np.array([round(MGC["Eccentricity (mm)"], 3)]))
MGC_p = np.squeeze(np.array([round(MGC["Proportion (%)"], 3)]))

# Create fits for these various curves
NF_int = si.interp1d(NF_ecc, NF_GC, kind="cubic")
TF_int = si.interp1d(TF_ecc, TF_GC, kind="cubic")
GCL_int = si.interp1d(GCL_ecc, GCL_t, kind="cubic")
IPL_int = si.interp1d(IPL_ecc, IPL_t, kind="cubic")
MGC_int = si.interp1d(MGC_ecc, MGC_p, kind="cubic")

###############################################################################


def pos_and_neg_formatter(x, pos) -> str:
    new_x = x * ECC_MM_TO_DEG
    formatted = f"{abs(new_x):.1f}"
    return "$-$" + formatted if new_x < 0 else formatted


def inside_foveola(x: float, y: float, ecc_hor: float, ecc_ver: float = 0) -> bool:
    """
    Compares an (x,y) position (using the population's chosen eccentricity,
    (`ecc_hor`, `ecc_ver`)), to determine if the cell is within the foveola,
    the area of the foveal pit/slope, devoid of RGCs.
    Returns True if within so.
    """
    if DISPLACE:
        x -= ecc_hor
        y -= ecc_ver
    ecc_abs = np.sqrt((x + ecc_hor) ** 2 + (y + ecc_ver) ** 2)
    if ecc_abs <= SLOPE / 2:
        return True
    return False


def still_inside_patch(x: float, y: float, ecc_hor: float, ecc_ver: float = 0) -> bool:
    """
    Compares an (x,y) position against the population's chosen eccentricity,
    (`ecc_hor`, `ecc_ver`), and patch size to determine if the cell is still
    within the boundaries set forth by said patch size.
    """
    if DISPLACE:
        x -= ecc_hor
        y -= ecc_ver
    if abs(x) > (PATCH_SIZE / 2):
        return False
    if abs(y) > (PATCH_SIZE / 2):
        return False
    return True


def xy_to_eccentricity(x: float, y: float, ecc_hor: float, ecc_ver: float = 0) -> float:
    """
    Given (x, y) (mm) and the population's eccentricity x- and y-coordinates
    `ecc_hor` and `ecc_ver` (mm), respectively, computes the absolute eccentricity
    of the position, retaininginformation about the nasal vs. temporal
    direction of so. For an eccentricity along the horizontal/x-meridian,
    `ecc_hor` <= 0, the position is considered temporal, else nasal.
    """

    cur_ecc = np.sqrt((x + ecc_hor) ** 2 + (y + ecc_ver) ** 2)
    if ecc_hor < 0:
        cur_ecc *= -1  # retain direction information, (-) for temporal
    return cur_ecc


def mRGC_spacing(ecc: float, enforce_min: bool = True) -> float:
    """
    Returns mRGC receptor field spacing in microns (um).
    Uses Watson 2014's approximation (R^2=0.9997) of midget RGC spacing
    averaged across all meridians:

        60s = 0.53 + 0.434r, where s is spacing and r is eccentricity in deg.

    We convert degrees to mm using Watson 2014's linear approximation.

        - `ecc` is the eccentricity in mm.
        - `enforce_min`: when True, enforces MIN_DEND spacing. This is to avoid
            erroneous behaviour for eccentricities close to 0.

    """
    ecc_deg = ECC_MM_TO_DEG * abs(ecc)
    s_deg = 1 / 60 * (0.53 + 0.434 * ecc_deg)
    s_mm = ECC_DEG_TO_MM * s_deg
    s_um = s_mm * 1000

    if enforce_min:
        s_um = max(s_um, MIN_DEND)  # enforce minimum to avoid errors ~0

    return s_um


def calc_axon_displacement(x_pos: float) -> float:
    """
    Returns the axon displacement (along y-axis) at the position at x=0.
    Returned in microns. x_pos should be given in um with (0, 0) representing
    the center of the retina (not center of the patch/population).
    Note that temporal and nasal interpolation uses different values given the
    need for temporal axons to divert around the foveola.
    This linear approximation would need to be updated if investigating
    populations distant from the horizontal meridian.
    """

    # how far along x-axis with respect to 1-10 degree limits (*1000 mm->um)
    one_deg_in_um = ECC_DEG_TO_MM * 1 * 1000
    ten_deg_in_um = ECC_DEG_TO_MM * 10 * 1000

    if x_pos >= 0:
        prop_along = 1 - (ten_deg_in_um - x_pos) / (ten_deg_in_um - one_deg_in_um)
        prop_along = max(0, prop_along)
        # max. displacement in y-dir at +1 deg [um]
        max_axon_displacement = ECC_DEG_TO_MM * 2.0 * 1000
        # min. displacement in y-dir at +10 deg [um] (closer to OND, more parallel)
        min_axon_displacement = ECC_DEG_TO_MM * 0.1 * 1000
        displaced = (
            -prop_along * (max_axon_displacement - min_axon_displacement)
            + max_axon_displacement
        )

    else:
        prop_along = abs((x_pos - one_deg_in_um) / (ten_deg_in_um - one_deg_in_um))
        min_axon_displacement = 0.55e3  # min. displacement in y-dir at 1 deg [um]
        max_axon_displacement = 1.7e3  # max. displacement in y-dir at 10 deg [um]
        # Linear displacement calculation between 1-10 deg limits
        displaced = (
            prop_along * (max_axon_displacement - min_axon_displacement)
            + min_axon_displacement
        )

    return displaced


def zIPL(ecc: float, ON: bool = True, terminal: bool = False) -> float:
    """
    Determines the z-value a cell takes based on the eccentricity's IPL
    thickness and cell type (ON vs OFF).

    Returns the z-value correspondent of dendrites in the IPL layer for either
    an ON or OFF cell (as per toggles). If ON is false, OFF is returned.

    The OFF/outer cell distribution is assumed to be normal and uses numpy's
    normal. The ON cell distribution is assumed to be anormal, demonstrating
    skewness, and is derived from scipy's skewnorm. `a` is skewness.

    Note: where 0% is the INL border and 100% is the GCL border...
    - 90% of ON cells are between 55-85% depth (none observed for 90-100%).
    - 90% of OFF cells are between 10-47% depth.

    Note: NO ON cells were observed to branch between 90-100% (Dacey 1993).

        - ecc: eccentricity of interest (mm).
        - ON: True if cell type is ON, False if OFF.
        - terminal: True, calculates the TERMINAL stratification values, i.e.,
            the peak stratification after branching. When False,

    """
    IPL_th = IPL_int(ecc)  # IPL thickness for this eccentricity
    branch_prct = Z_BRANCH / IPL_th * 100  # branch-height as % of IPL

    # CI table - https://www.mathsisfun.com/data/confidence-interval.html
    std_factor = 3  # for 99.7%
    std_factor = 2  # for 95%
    std_factor = 1.645  # for 90%

    if ON:
        depth_prct = np.random.normal(70, 15 / std_factor)
        depth_prct = skewnorm.rvs(a=-1, loc=75, scale=15 / std_factor)
        # ON cells do not branch in the last 10% (prevents entry into GCL too)
        depth_prct = min(90, depth_prct)
    else:
        depth_prct = np.random.normal(28.5, 18.5 / std_factor)
        depth_prct = max(branch_prct, depth_prct)  # prevent entry into INL

    z = IPL_th * (1 - depth_prct / 100)  # z=0 is GCL border; z=IPL_th is INL

    return z


def cell_type_to_diameter(cell: str, cell_no: int, author: str = "Consensus") -> float:
    """
    Returns the midget cell soma diameter of the cell type denoted by `cell`.
    Based on the paper desgnated by `author`:

        - Watanabe 1989: 14.5 +- 2.51 um.
        - Dacey 1993: ON cells 18.6 +- 2.3 um, OFF cells 17.4 +- 2.3 um.
        - Liu 2017: 11.4 +- 1.8 um (up to 7.5 deg).
        - Consensus: general consensus based on observations of human-based
            data: 12.0 +- 1.0 um.

    Cell no. is used to avoid returning the same random value when seeded.

    Used for the NEURON representation when assigning a diameter to the somatic
    points.

    """

    if TEST_SMALL_SOMA:  # used for investigating influence of soma diameter
        base = 9.0
        std = 1.0
        return np.random.normal(loc=base, scale=std, size=cell_no)[-1]

    if author == "Dacey":
        base = 18.6 if cell == "ON" else 17.4
        std = 2.3

    elif author == "Watanabe":
        base = 14.5
        std = 2.51

    elif author == "Liu":
        base = 11.4
        std = 1.8

    elif author == "Consensus":
        base = 12.0
        std = 1.0

    else:
        assert False, "ERROR: No author selected for determining soma diameter."

    # size = array[-1] -> different yet reproducible values under a given seed
    # the [-1] is actually not needed but have kept to be consistent with
    # what was used in the study
    # The alternative would be to just call size=1 as it is still reproducible
    # (and random) following the same number of np.random calls
    return np.random.normal(loc=base, scale=std, size=cell_no)[-1]


def get_ais_diameter(cell_no: int) -> float:
    """
    Returns the AIS diameter for a midget cell based on the range observed
    in Rodieck (1985) (0.6-0.8 um). Assumes a uniform distribution for this
    range.

    Cell no. is used to avoid returning the same random value when seeded.

    Used for the NEURON representation by assigning a diameter to the AIS
    points.

    """
    return np.random.uniform(low=0.60, high=0.80, size=cell_no)[-1]


def ecc_to_density(ecc: float) -> float:
    """
    Estimates (through interpolation) the RGC density of an input
    eccentricitity (mm).
    Differences between superior and inferior densities are assumed negligible.
    """
    if abs(ecc) < PIT:
        return 0  # in foveal pit, no RGCs
    density = TF_int(ecc) if ecc < 0 else NF_int(ecc)
    # Scale density by proportion of midget cell : GC for this eccentricity
    density *= MGC_proportion

    return density


def ecc_to_diameter(
    ecc: Union[float, list], cell_type: str = "OFF", enforce_min: bool = True
) -> list:
    """
    Returns the diameter of a midget RGC's dendritic tree at eccentricity/ies
    `ecc` (mm). Based on the method, equations, and results of Dacey (1992).

    Note that `ecc` refers to the position of the tree, NOT the cell.
    This is relevant when considering lateral displacement of RGCs.

    If `cell type` is "ON", the cell is treated as an ON mRGC, else OFF mRGC.

    `enforce_min` enforces that the minimum dendritic diameter is 5 microns
    (minimum is as specified in the parameters at the header of the file).

    Returns the resultant diameter(s) (in microns) as a list regardless of the
    number of eccentricities passed in.
    """

    if not isinstance(ecc, list):
        ecc = [ecc]  # Convert to list

    if len(np.shape(ecc)) > 1:
        ecc = np.squeeze(ecc)

    dend_size = []

    for x in ecc:

        # Use the Dacey 1992 fit (R = 0.94)
        size = 8.64 * abs(x) ** 1.04

        # Minimum dendritic diameter is enforced
        if enforce_min:
            size = max(MIN_DEND, size)

        if cell_type == "ON":  # scale ON cells
            size *= D_ON_OFF

        dend_size.append(size)

    return dend_size  # [um]


def ecc_to_ngcl(ecc: float, thickness_curve: bool = False) -> int:
    """
    Returns the number of GC layers at a given eccentricity. Bases the value
    on the mean density across the patch associated with this eccentricity.
        - `ecc`: eccentricity of the point in mm (negative if temporal).
    """

    # In foveal pit
    if abs(ecc) < PIT:
        return 0

    # Uses density curve to derive number of GC layers
    # Max density and thus density per layer
    # NOTE: max(Nasal) > max(Temporal)
    density_max = max(NF_GC) * MGC_proportion  # Layers ~ Density
    density_per = density_max / GCL_MAX  # Density per layer
    # must round to avoid floating point issues
    n_layer = np.ceil(np.round(ecc_to_density(ecc) / density_per, 2))
    n_layer = int(max(1, n_layer))

    return n_layer


def validate_setup(ecc_hor: float, patch_size: float, ecc_ver: float = 0) -> bool:
    """
    Determines if the input `PATCH_SIZE` and eccentricity values would extend
    into regions beyond valid interpolation ranges.
    Values within the foveola are permitted because they are ignored
    during population generation.
    Returns True if all values are valid, False otherwise.
    """

    patch_left = ecc_hor - patch_size / 2
    patch_right = ecc_hor + patch_size / 2

    patch_up = ecc_ver + patch_size / 2
    patch_down = ecc_ver - patch_size / 2

    valid = True

    # only need to check that furthest point (ecc_diagonal) of pop square is within
    # valid ranges
    # pop_extent = 1 / np.sqrt(2) * patch_size / 2  # diagonal reach from centre
    x_max = max(abs(patch_left), abs(patch_right))
    y_max = max(abs(patch_down), abs(patch_up))
    ecc_diagonal = np.sqrt(x_max**2 + y_max**2)
    if ecc_diagonal >= VALID_RANGE[1]:  # abs because valid_range[0] == [1]
        print(f"{ecc_diagonal = }")
        valid = False

    assert valid, (
        "ERROR: Patch exceeds the range for which the model is valid - "
        "revise your experimental parameters!"
    )

    return True


###############################################################################


class Population:
    def __init__(self) -> None:
        self.neurons = np.array([])
        self.size = 0

    def create_neurons(self) -> None:
        for i in range(self.size):
            self.neurons = np.append(self.neurons, Neuron())

    def add_somas(self, coords_soma: np.ndarray) -> None:
        """
        Given the soma coordinates array, iteratively assigns each 'Neuron' to
        a soma.
        """
        # reshape to conform to Nx3 expectation
        coords_soma = np.reshape(coords_soma, (-1, 3))

        if np.size(self.neurons) == 0:
            self.create_neurons()
        else:
            assert (
                np.size(self.neurons) == self.size
            ), f"Incorrect number of neurons for population size, neurons={np.size(self.neurons)}, size={self.size}"

        assert self.size == len(
            coords_soma
        ), f"Incorrect number of soma for size of Population, pop={self.size}, soma={len(coords_soma)}"

        for i in range(self.size):
            self.neurons[i].soma_loc = coords_soma[i, :]

    def project_axon(self, cell_id: int, ecc_hor: float, ecc_ver: float = 0) -> None:
        """
        Given a start (soma) and pre-determined end (optic disc location)
        position, returns a curve representative of the axon. This curve avoids
        the foveal centre. OND parameters are defined at the beginning of this
        script. `ecc_hor` and `ecc_ver` are used to convert the cell's position
        to be relative to the retinal plane, not just the population centre.

            - `cell_id`: used to define the start position (soma/origin).
            - `ecc_hor`: eccentricity along the horizontal meridian (mm).
            - `ecc_ver`: eccentricity along the vertical meridian (mm).
                *** ecc_ver NOT TESTED ***

        """

        # OND location in microns and assuming [0,0,0] is the patch centre
        ond = (OND - ecc_hor) * 1000

        node = self.neurons[cell_id].soma_loc

        # Construct initial segment (xy displacement to match dend's z) #

        # Displace node (soma) starting points by soma diameter
        node[0] += SOMA_D * 1e3  # [um]

        # Define points
        pt0 = node  # Soma (start)

        base_x = self.x_base[cell_id]
        new_x = base_x + 4.0 + (LYR_MAX - 0.5 - self.z_idx[cell_id]) / LYR_MAX * 5
        pt1 = np.array([new_x, node[1], node[2]])  # Hillock midpoint

        new_z = self.z_floor - 2.5 - self.z_idx[cell_id] * 2 / 3
        pt2 = np.array([pt1[0], node[1], new_z])  # Hillock  (end)

        # Interpolate to establish the Hillock
        x = np.array([pt0[0], pt1[0], pt2[0]])
        z = np.array([pt0[2], pt1[2], pt2[2]])
        f1 = si.interp1d(x, z, kind="slinear")

        x1 = np.arange(pt0[0], pt2[0], RES_AX_1)
        y1 = np.repeat(pt1[1], len(x1))
        z1 = np.array(f1(x1))

        # Construct second segment (to OND) #

        # Declare mid-point between soma and OND as max-displacement and 3rd-pt
        # Declare centre of OND as the fourth (and final) point
        # The z-axis is the same as the goal z as the OND is assumed to encompass
        # the height of the GCL

        if DISPLACE:  # Not validated, never used DISPLACE
            i = 1 if node[1] >= 0 else -1  # superior vs. inferior direction
            x_midpoint = (pt2[0] + OND * 1000) / 2
            y_displaced = calc_axon_displacement(pt2[0])
            if pt0[0] >= 0:  # Nasal side cell
                if abs(pt2[1]) > (0.75 * y_displaced):
                    y_displaced = 0.75 * abs(node[1])

        else:
            i = (
                1 if (node[1] + ecc_ver * 1e3) >= 0 else -1
            )  # superior vs. inferior direction
            x_midpoint = (pt2[0] + ond) / 2
            y_displaced = calc_axon_displacement(pt2[0] + ecc_hor * 1e3) + ecc_ver * 1e3
            if (pt0[0] + (ecc_hor * 1e3)) >= 0:  # Nasal side cell
                if (abs(pt2[1]) + (ecc_ver * 1e3)) > (0.75 * y_displaced):
                    y_displaced = 0.75 * abs(node[1])

        pt3 = [x_midpoint, i * y_displaced + node[1], pt2[2]]
        pt4 = [ond, 0, pt2[2]]

        # Interpolate to project axon from the AIS to the OND
        x = np.array([pt2[0], pt3[0], pt4[0]])
        y = np.array([pt2[1], pt3[1], pt4[1]])
        f2 = si.interp1d(x, y, kind="quadratic")

        x2 = np.arange(pt2[0], pt4[0], RES_AX_2)
        y2 = np.array(f2(x2))
        z2 = np.repeat(pt2[2], len(x2))

        # Update population
        self.neurons[cell_id].x_ax = np.concatenate((x1, x2))
        self.neurons[cell_id].y_ax = np.concatenate((y1, y2))
        self.neurons[cell_id].z_ax = np.concatenate((z1, z2))

    def create_cell(self, cell_id: int) -> None:
        """Convert cell representation to .txt file for cell number `cell_id`."""

        ID = 0
        D = SOMA_D * 1000  # [microns]
        tree = self.neurons[cell_id].tree_node

        # SOMA [LABEL: 0]

        x_soma = self.neurons[cell_id].soma_loc[0]
        y_soma = self.neurons[cell_id].soma_loc[1]
        z_soma = self.neurons[cell_id].soma_loc[2]

        # Using soma parent ID as -1 to avoid interpretation errors
        # with generate_hoc.py.
        cell = np.array(
            [0, x_soma, y_soma, z_soma, x_soma, y_soma, z_soma + D / 2, ID, -1, D / 2]
        )
        ID += 1

        # DENDRITE [LABEL: 1]

        # Base / initial stalk #

        x = self.neurons[cell_id].x_sd
        y = self.neurons[cell_id].y_sd
        z = self.neurons[cell_id].z_sd

        idx = 0

        # Add the first dendrite to establish 2D array
        # This is a loop to find the first pt which has moved off the soma

        while True:

            length = np.sqrt(
                (x[idx] - x_soma) ** 2 + (y[idx] - y_soma) ** 2 + (z[idx] - z_soma) ** 2
            )

            if not length:
                idx += 1
                continue  # point coincides with the soma (redundant)

            node = np.array(
                [1, x[idx], y[idx], z[idx], x_soma, y_soma, z_soma, ID, 0, length]
            )
            cell = np.vstack((cell, node))
            idx += 1

            break

        ID += 1

        for j in range(idx, len(x)):

            x_prev = cell[ID - 1][1]
            y_prev = cell[ID - 1][2]
            z_prev = cell[ID - 1][3]
            length = np.sqrt(
                (x[j] - x_prev) ** 2 + (y[j] - y_prev) ** 2 + (z[j] - z_prev) ** 2
            )

            if not length:
                continue

            node = np.array(
                [1, x[j], y[j], z[j], x_prev, y_prev, z_prev, ID, ID - 1, length],
                dtype=object,
            )
            cell = np.vstack((cell, node))
            ID += 1

        # Dendritic trees as generated previously #
        # Increment cur/prev ID of tree by ID

        tree[:, 7] += ID  # Current
        tree[:, 8] += ID  # Previous

        # Increment ID by length of tree
        ID += len(tree)

        # Add the tree to the cell file
        cell = np.vstack((cell, tree))

        # AXON - HILLOCKS [LABEL: 2], AIS [LABEL: 3], otherwise [LABEL: 4]
        # Hillocks  : first L_HIL microns,
        # AIS       : next  L_AIS microns,
        # AXON      : the rest until within OND

        x = self.neurons[cell_id].x_ax
        y = self.neurons[cell_id].y_ax
        z = self.neurons[cell_id].z_ax

        # Place the first node with the soma as its parent
        length = np.sqrt(
            (x[0] - x_soma) ** 2 + (y[0] - y_soma) ** 2 + (z[0] - z_soma) ** 2
        )
        node = np.array([2, x[0], y[0], z[0], x_soma, y_soma, z_soma, ID, 0, length])
        cell = np.vstack((cell, node))
        ID += 1

        for j in range(1, len(x)):

            # Check distance versus soma to determine compartment ID
            length = np.sqrt(
                (x[j] - x_soma) ** 2 + (y[j] - y_soma) ** 2 + (z[j] - z_soma) ** 2
            )

            if length <= L_HIL:
                c_id = 2
            elif length <= (L_HIL + L_AIS):
                c_id = 3
            else:

                if SHORT_AX and length > SHORT_AX_LEN:
                    break

                # Terminate axon extension if within OND
                c_id = 4

                # Check if within OND's elliptical region
                x_mm = x[j] / 1000  # current x pos in mm
                y_mm = y[j] / 1000  # current y pos in mm
                if within_ellipse(x_mm, y_mm, OND_POS[0], OND_POS[1], OND_X, OND_Y):
                    break

            # Determine node parameters
            x_prev = cell[ID - 1][1]
            y_prev = cell[ID - 1][2]
            z_prev = cell[ID - 1][3]
            length = np.sqrt(
                (x[j] - x_prev) ** 2 + (y[j] - y_prev) ** 2 + (z[j] - z_prev) ** 2
            )
            node = np.array(
                [c_id, x[j], y[j], z[j], x_prev, y_prev, z_prev, ID, ID - 1, length]
            )
            cell = np.vstack((cell, node))
            ID += 1

        SAVE_PATH = TXT_PATH + "cell_%.4d.txt" % cell_id
        np.savetxt(
            SAVE_PATH,
            cell,
            fmt="%i" + " %.5f" * 6 + " %i" * 2 + " %.5f",
        )

        self.neurons[cell_id].cell = cell

    def create_csv(self) -> None:
        """Create's a population .csv file detailing soma locations, cell types
        and more."""

        header = [
            "x (um)",
            "y (um)",
            "z (um)",
            "Cell type",
            "Dendritic diameter (um)",
            "Dendritic area (um^2)",
            "x-tree (um)",
            "y-tree (um)",
            "z-tree (um)",
        ]
        x_csv = np.empty(self.size, float)
        y_csv = np.empty(self.size, float)
        z_csv = np.empty(self.size, float)

        type_csv = np.empty(self.size, "<U3")  # <U3 is a str of len <= 3

        D_tree = np.empty(self.size, float)
        A_tree = np.empty(self.size, float)
        x_tree = np.empty(self.size, float)
        y_tree = np.empty(self.size, float)
        z_tree = np.empty(self.size, float)

        for i, nrn in enumerate(self.neurons):
            x_csv[i] = nrn.soma_loc[0]
            y_csv[i] = nrn.soma_loc[1]
            z_csv[i] = nrn.soma_loc[2]

            type_csv[i] = nrn.cell_type

            D_tree[i] = nrn.tree_diameter
            A_tree[i] = nrn.tree_area
            x_tree[i] = nrn.tree_loc[0]
            y_tree[i] = nrn.tree_loc[1]
            z_tree[i] = nrn.tree_loc[2]

        # Round for readability
        n_round = 3
        x_csv = np.round(x_csv, n_round)
        y_csv = np.round(y_csv, n_round)
        z_csv = np.round(z_csv, n_round)
        D_tree = np.round(D_tree, n_round)
        A_tree = np.round(A_tree, n_round)
        x_tree = np.round(x_tree, n_round)
        y_tree = np.round(y_tree, n_round)
        z_tree = np.round(z_tree, n_round)

        pop_csv = np.vstack(
            np.array(
                [
                    x_csv,
                    y_csv,
                    z_csv,
                    type_csv,
                    D_tree,
                    A_tree,
                    x_tree,
                    y_tree,
                    z_tree,
                ]
            ).T
        )
        pd.DataFrame(pop_csv).to_csv(
            f"{TXT_PATH.split('cellText/')[0]}/trial{TRIAL:03d}-population.csv",
            header=header,
            sep=",",
            index_label="Index",
        )

    def create_3d_plot_for_presentations(
        self,
        path: str,
    ) -> None:
        """
        Creates multiple 3D plots the population.

            - `ecc_hor`, `ecc_ver`: distances in mm along the horizontal and
                vertical meridians, respectively.
            - `title`: title of plot.
            - `path`: save path.
            - `colour_by_cell`: True - a cell and its components have ONE
                colour, i.e., coloured according to parent. False - coloured
                according to component, regardless of parent cell.

        """

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(projection="3d", proj_type="ortho")

        COLOUR_CYCLE = [
            "purple",
            "gold",
            "forestgreen",
            "tab:brown",
            "olive",
            "darkorchid",
            "red",
            "violet",
            "mediumvioletred",
            "slateblue",
        ]
        L_CYCLE = len(COLOUR_CYCLE)

        on_labelled, off_labelled = False, False

        soma_size = 3500  # marker size for soma, roughly to scale
        N = len(self.neurons)

        for idx, neuron in enumerate(self.neurons):

            cur_clr = COLOUR_CYCLE[idx % L_CYCLE]
            cur_clr = cmap(0.70) if neuron.cell_type == "ON" else cmap(0.30)

            cur_label = None

            if neuron.cell_type == "ON" and not on_labelled:
                cur_label = "ON"
                on_labelled = True

            elif neuron.cell_type == "OFF" and not off_labelled:
                cur_label = "OFF"
                off_labelled = True

            plt.figure(fig1.number)

            ax1.scatter(
                neuron.soma_loc[0],
                neuron.soma_loc[1],
                neuron.soma_loc[2],
                facecolors=cur_clr,
                edgecolors="black",
                linewidth=3,
                s=soma_size,
                marker="o",
                label=cur_label,
                alpha=0.9,
            )

            ax1.plot3D(
                neuron.x_sd[5:],
                neuron.y_sd[5:],
                neuron.z_sd[5:],
                color=cur_clr,
                linewidth=5,
                alpha=0.65,
            )

            ax1.scatter(
                neuron.tree_node[:, 1],
                neuron.tree_node[:, 2],
                neuron.tree_node[:, 3],
                s=30,
                edgecolors="black",
                linewidth=0.3,
                facecolors=cur_clr,
                depthshade=False,
                # zorder=100,
            )

            # Do not plot full extent of axons to keep visually concise & well-scaled
            mask = neuron.x_ax < (PATCH_SIZE / 2 * 1000 + 40)

            if AXON:
                ax1.plot3D(
                    neuron.x_ax[mask],
                    neuron.y_ax[mask],
                    neuron.z_ax[mask],
                    color=cur_clr,
                    marker="_",
                    linewidth=6,
                    alpha=0.75,
                )

        # Add 10-um scale bar
        # ax1.plot([65, 75], [80, 80], [-55, -55], c="k", lw=10)
        # scale = ax1.text(62, -80, -60, "10 $\mu m$", color="k", fontsize=22)

        # Plot 1
        plt.legend()
        plt.figure(fig1.number)

        plt.axis("off")
        plt.tick_params(
            axis="both",
            left="off",
            top="off",
            right="off",
            bottom="off",
            labelleft="off",
            labeltop="off",
            labelright="off",
            labelbottom="off",
        )

        ax1.view_init(0.5, -90)
        edge = PATCH_SIZE / 2 * 1000 + 40
        ax1.set_xlim(-edge, edge)
        ax1.set_ylim(-edge, edge)
        ax1.set_zlim(-70, 70)
        plt.savefig(path + "presentation_tile.jpg", bbox_inches="tight")
        plt.savefig(path + "presentation_tile.pdf", bbox_inches="tight")

        # Add 200-um diameter epiretinal electrode
        electrode_patch = Circle(
            (0, 0),
            radius=100,
            facecolor="lime",
            edgecolor="black",
            lw=2.5,
        )
        # dummy scatter point to get circular electrode legend label
        ax1.scatter(
            -edge + 10000,  # beyond limits of plot
            50,
            -100,
            s=soma_size,
            facecolors="lime",
            edgecolors="black",
            linewidth=3,
            label="Electrode",
            marker="o",
        )
        ax1.set_xlim(-edge, edge)
        ax1.set_zlim(-70, 70)
        ax1.add_patch(electrode_patch)
        ax1.legend()
        art3d.pathpatch_2d_to_3d(electrode_patch, z=-80, zdir="z")
        plt.savefig(path + "presentation_tile-electrode.jpg", bbox_inches="tight")
        plt.savefig(path + "presentation_tile-electrode.pdf", bbox_inches="tight")

        plt.close()

    def create_xz_mosaic(self, path: str) -> None:
        """
        Creates scatter plot of the mosaic distribution in the x-z planes
        (cross-section).
        """

        fig1 = plt.figure()
        ax1 = plt.gca()
        pt_size = 1500  # size of cell body markers
        cur_clr = "black"

        for _, neuron in enumerate(self.neurons):

            plt.figure(fig1.number)

            # mask cells to create cross-section view
            if JITTER:
                # May miss or include extra cells if jitter is involved
                mask = abs(neuron.soma_loc[1]) <= 8
            else:
                mask = neuron.soma_loc[1] == 0

            ax1.scatter(
                neuron.soma_loc[0][mask],
                neuron.soma_loc[2][mask],
                facecolors=cur_clr,
                edgecolors="black",
                linewidth=3,
                s=pt_size,
                marker="o",
            )

        ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])
        ax1.axes.xaxis.set_ticklabels([])
        ax1.axes.yaxis.set_ticklabels([])

        plt.title("Cross-section", pad=40, fontsize=100)
        ax1.set_xlabel("x", labelpad=40, fontsize=96)
        ax1.set_ylabel("z", labelpad=80, fontsize=96, rotation=0)
        plt.savefig(path + "mosaic_tile-xz.jpg", bbox_inches="tight")
        plt.savefig(path + "mosaic_tile-xz.pdf", bbox_inches="tight")
        plt.close()

    def create_xy_mosaic(self, path: str) -> None:
        """
        Creates scatter plot of the mosaic distribution in the x-y (retinal-
        surface).
        """

        fig1 = plt.figure()
        ax1 = plt.gca()

        pt_size = 1500  # size of cell body markers
        cur_clr = "black"

        for idx, neuron in enumerate(self.neurons):

            mask = self.z_idx[idx] == 0

            ax1.scatter(
                neuron.soma_loc[0][mask],
                neuron.soma_loc[1][mask],
                facecolors=cur_clr,
                edgecolors="black",
                linewidth=3,
                s=pt_size,
                marker="o",
            )

        ax1.axes.xaxis.set_ticks([])
        ax1.axes.yaxis.set_ticks([])
        ax1.axes.xaxis.set_ticklabels([])
        ax1.axes.yaxis.set_ticklabels([])
        plt.title("Retinal surface", pad=40, fontsize=100)
        ax1.set_xlabel("x", labelpad=40, fontsize=96)
        ax1.set_ylabel("y", labelpad=80, fontsize=96, rotation=0)

        plt.savefig(path + "mosaic_tile-xy.jpg", bbox_inches="tight")
        plt.savefig(path + "mosaic_tile-xy.pdf", bbox_inches="tight")
        plt.close()

    def create_simplified_mosaic_plots(self, path: str) -> None:
        """
        Creates scatter plots of the mosaic distribution in the x-y (retinal-
        surface) and x-z planes (cross-section).
        """
        self.create_xz_mosaic(path=path)
        self.create_xy_mosaic(path=path)


class Neuron:
    def __init__(self, soma_loc: np.ndarray = np.empty(3, float)):
        self.soma_loc = soma_loc


def create_trial_file(file_path: str, n_cells: int, std: float) -> None:
    """Creates a .txt file detailing the parameters used for the current trial."""
    n_off = int(np.ceil(n_cells / (1 + ON_OFF)))
    n_on = int(n_cells - n_off)

    file_path += f"/trial_{TRIAL:03d}.txt"
    trial_file = open(file_path, "w")
    trial_file.write(f"TRIAL {TRIAL:03d}\n\n")
    trial_file.write(f"Patch size of {PATCH_SIZE:.4f} mm x {PATCH_SIZE:.4f} mm.\n")
    trial_file.write(f"X Eccentricity of {ECC:.2f} mm.\n")
    trial_file.write(f"Ecc. (X, Y) of {ECC:.2f}, {ECC_Y:.2f} mm.\n")
    trial_file.write(
        f"Number of cells is {int(n_cells)} ... ({n_on} ON, {n_off} OFF).\n"
    )

    trial_file.write(f"CROP_POPULATION is {CROP_POPULATION}.\n")
    if CROP_POPULATION:
        trial_file.write(f"{CROP_X =} mm.\n")
        trial_file.write(f"{CROP_Y =} mm.\n")

    trial_file.write(
        f"Soma diameter of {SOMA_D*1000:.3f} um, BUFFER of {BUFFER/SOMA_D:.2f} (*SOMA_D).\n"
    )
    trial_file.write(f"Max no. of GCLs is {GCL_MAX}, no. of GCLs is {LYR_MAX}.\n")
    trial_file.write(f"Jitter is {JITTER}.\n")
    trial_file.write(f"Standard deviation is {std:.2f}%.\n")
    trial_file.write(f"Percent of midget to total RGCs is {MGC_proportion}.\n")
    trial_file.write(f"Coverage factor is {CF}.\n")
    trial_file.write(f"Eccentricity conversions...1 deg = {ECC_DEG_TO_MM} mm.\n")
    trial_file.write(f"Eccentricity conversions...1 mm = {ECC_MM_TO_DEG} deg.\n")

    trial_file.write(
        f"\nAxon resolutions: Segment 1) {RES_AX_1} um, Segment 2) {RES_AX_2} um.\n"
    )
    trial_file.write(f"Soma-dendrite branch uses {L_SD} points per segment.\n")
    trial_file.write(f"Short axons (SHORT_AX) is {SHORT_AX}. ")
    if SHORT_AX:
        trial_file.write(f"Axons are cut after {SHORT_AX_LEN} um.")
    trial_file.write("\n")
    trial_file.close()


def plot_diam_to_density(ecc: float) -> None:
    """
    Function to plot the density values derived from dendritic tree diameter
    measurements.
    """

    # ecc_hor samples
    ecc_hor = np.linspace(-ecc, ecc, 100)

    # Use dendritic diameter to estimate number of cells
    D = ecc_to_diameter(ecc_hor)  # diameter of dendritic tree
    D = D[0]
    A = np.zeros(np.shape(D))
    for idx, x in enumerate(D):
        A[idx] = diagonal_to_hex_area(x / 1000, soma=False)
    N = PATCH_SIZE**2 / A * 2  # n for 1 layer ON, 1 layer OFF
    density = N / A / 1000  # RGC density * 1000

    # Plot
    fig1 = plt.figure(10, tight_layout=True)
    ax = fig1.add_subplot(111)
    ax.plot(NF_ecc, NF_GC / 1000, c="orange", label="Nasal data", linestyle="-")
    ax.plot(TF_ecc, TF_GC / 1000, c="orange", label="Temporal data", linestyle="-")
    ax.plot(
        ecc_hor, density / 1000, c="k", linewidth=5, label="Dendritic-diameter derived"
    )
    ax.set_xlabel("Eccentricity (mm)")
    ax.set_ylabel("Number of RGC (x1000) (/$/mm^2$)")
    plt.show()


def ipl_stratification_distribution(
    n_cells: int, ecc: float = -1.0
) -> Tuple[List[float], List[float]]:
    """
    Given `n_cells` no. of cells at  an eccentricity (mm) `ecc`, returns a
    tuple of ON/OFF stratification depths as % (0 - INL, 100 - GCL border).
    """
    n_off = int(np.ceil(n_cells / (1 + ON_OFF)))
    n_on = n_cells - n_off
    IPL_th = IPL_int(ecc)

    z_on, z_off = np.zeros(n_on, dtype=float), np.zeros(n_off, dtype=float)

    for i in range(n_on):
        z_on[i] = zIPL(ecc, ON=True)

    for i in range(n_off):
        z_off[i] = zIPL(ecc, ON=False)

    # Convert to % instead of distance value
    z_on /= IPL_th
    z_on *= 100
    z_off /= IPL_th
    z_off *= 100

    # Note that z=0 corresponds to depth=100%
    z_on = 100 - z_on
    z_off = 100 - z_off

    return (z_on, z_off)


def plot_distance_to_electrode(
    ecc_mm: list = [], z_electrode: float = -80, dir: str = "figures"
) -> None:
    """
    Creates a plot of distance to electrode vs. eccentricity based on electrode
    depth, `z_electrode` (um) and by determining innermost RGC at each ecc. in
    the list, `ecc_mm` (mm). `dir` is where the plots are saved.
    """

    if np.size(ecc_mm) == 0:
        ecc_mm = np.linspace(-2.49, 2.49, 60)

    gc_in_column = np.zeros(np.shape(ecc_mm))
    z_cur = np.zeros(np.shape(ecc_mm))
    distance = np.ones(np.shape(ecc_mm)) * 1e6

    for i, x in enumerate(ecc_mm):

        if inside_foveola(x=x, y=0, ecc_hor=0, ecc_ver=0):
            distance[i] = np.nan

        gc_in_column[i] = ecc_to_ngcl(x)
        max_z_index = gc_in_column[i] - 1

        # Scale (offset) layer index by GCL thickness
        # Offset pushes cells into GCL, away from IPL
        z_cur[i] = (max_z_index + 0.5) * -GCL_int(x) / gc_in_column[i]

        distance[i] = abs(z_cur[i] - z_electrode)

    N_TICKS = 4
    fig1 = plt.figure(111)
    ax = fig1.add_subplot(111)
    ax.scatter(ecc_mm, distance, c="red", ec="black", marker="d", s=900, lw=10)

    ax.set_xlabel("Eccentricity (mm)")
    ax.set_ylabel("Distance from electrode \nto closest soma ($\\mu m$)")
    ax.set_xlim([-2.55, 2.55])
    # ax.set_title("Distance between innermost cell and electrode (z=-80 $\mu m$)")

    # Add secondary x-axis for ecc. in degrees
    curr_lims = plt.gca().get_xlim()
    ax3 = ax.twiny()
    ax3.set_xlabel("Eccentricity (degrees)", labelpad=20)
    ax3.set_xlim(curr_lims)
    formatter = mticker.FuncFormatter(pos_and_neg_formatter)
    ax3.xaxis.set_major_formatter(formatter)
    ax.locator_params(axis="both", nbins=N_TICKS)
    ax3.locator_params(axis="both", nbins=N_TICKS)
    plt.savefig(f"{dir}/fig1-distance-to-electrode.jpg")
    plt.savefig(f"{dir}/fig1-distance-to-electrode.pdf")
    plt.close()


def create_plots(heat: bool = False) -> None:
    """Produces plots of the various distributions and fits."""

    x_ecc = np.linspace(-2.49, 2.49, 1000)
    N_TICKS = 4

    figure_dir = "figures"
    create_dir(sub_dir=figure_dir)
    plot_distance_to_electrode(dir=figure_dir)

    ###################### IPL stratification distribution ####################

    n_cells = 500
    ipl_ecc = -1.00
    z_on, z_off = ipl_stratification_distribution(n_cells=n_cells, ecc=ipl_ecc)

    std_factor = 1.645  # for 90%
    branch_prct = Z_BRANCH / IPL_int(ipl_ecc) * 100  # branch-height as % of IPL

    ecc_distr = np.linspace(0, 90, n_cells)

    on_distr = skewnorm(a=-1, loc=75, scale=15 / std_factor)
    on_curve = on_distr.pdf(ecc_distr)

    off_distr = norm(loc=28.5, scale=18.5 / std_factor)
    off_curve = off_distr.pdf(ecc_distr)

    fig1 = plt.figure()
    ax = plt.gca()
    n_bin = int((90 - 0) / (Z_BRANCH)) + 1
    bins_on, _, _ = plt.hist(
        z_on,
        range=(0, 90),
        bins=n_bin,
        color="r",
        label="ON",
        orientation="horizontal",
        histtype="step",
        linewidth=7,
    )
    bins_off, _, _ = plt.hist(
        z_off,
        range=(0, 90),
        bins=n_bin,
        color="k",
        label="OFF",
        orientation="horizontal",
        histtype="step",
        # linestyle="--",
        linewidth=7,
    )

    # Normalise curves wrt current plot
    max_bin = max(max(bins_on), max(bins_off))
    on_curve = on_curve / max(on_curve) * max_bin
    off_curve = off_curve / max(off_curve) * max_bin

    plt.plot(on_curve, ecc_distr, color="r", linestyle=(0, (5, 2.5)), linewidth=9)
    plt.plot(off_curve, ecc_distr, color="k", linestyle=(0, (5, 2.5)), linewidth=9)

    plt.legend()
    plt.xlabel("Frequency")
    plt.ylabel("IPL straification depth (%)")
    plt.title(
        f"Dendritic depth distribution\n(ncells={n_cells}, ecc = {ipl_ecc:.2f} mm)"
    )
    plt.ylim(bottom=100, top=0)
    plt.xlim(
        0.01,
    )

    freq_formatter = mticker.FuncFormatter(lambda x, pos: f"{100 * x / n_cells:.0f}")
    ax.xaxis.set_major_formatter(freq_formatter)
    plt.xlabel("Frequency (%)")

    ax.locator_params(axis="both", nbins=N_TICKS)
    ax.set_yticks([0, 30, 60, 90])

    # Label INL and GCL at 0% and 100%, respectively
    curr_y_lims = plt.gca().get_ylim()
    ax4 = ax.twinx()
    ax4.set_yticks([0, 1])
    ipl_dict = {0: "GCL     ", 1: "INL     "}  # 0 bot of axis
    ipl_formatter = mticker.FuncFormatter(  # apply a function formatter
        lambda x, pos: ipl_dict.get(x)
    )
    ax4.yaxis.set_major_formatter(ipl_formatter)
    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/fig1-ipl-distribution-{abs(ipl_ecc):.1f}.jpg")
    plt.savefig(f"{figure_dir}/fig1-ipl-distribution-{abs(ipl_ecc):.1f}.pdf")
    plt.close()

    ################### AXON DISPLACEMENT VERSUS ECCENTRICITY #################

    n_ax = 6
    # x_ax = np.tile(np.linspace(-350, -2490, n_ax), 2)
    x_ax = np.tile(np.linspace(-350, -3490, n_ax), 2)
    x_ax = np.concatenate((x_ax, np.tile(np.linspace(300, 2600, int(n_ax / 2)), 2)))
    y_ax = np.ones(np.shape(x_ax))
    # for idx, _ in enumerate(y_ax[int(len(y_ax) * 1 / 2) : :]):
    #     y_ax[idx] *= 50 * idx
    y_ax[n_ax : int(n_ax * 2)] *= -1
    y_ax[15::] *= -1
    z_ax = np.zeros(np.shape(x_ax))

    coords = np.empty((len(x_ax), 3), dtype=float)
    coords[:, 0] = x_ax
    coords[:, 1] = y_ax
    coords[:, 2] = z_ax

    fig1, ax = plt.subplots()
    # ax = plt.gca()

    # OND location in microns and assuming [0,0,0] is the patch centre
    ond = (OND) * 1000

    # Create connection lists for x,y,z
    X_ax, Y_ax, Z_ax = np.array([]), np.array([]), np.array([])

    for _, node in enumerate(coords):

        # Define points
        pt0 = node  # Soma (start)

        base_x = node[0]
        new_x = base_x + 4.0 + (LYR_MAX - 0.5 - node[2]) / LYR_MAX * 5
        pt1 = np.array([new_x, node[1], node[2]])  # Hillock midpoint

        new_z = -40 - 2.5 - node[2] * 2 / 3
        pt2 = np.array([pt1[0], node[1], new_z])  # Hillock  (end)

        # Interpolate to establish the Hillock
        x = np.array([pt0[0], pt1[0], pt2[0]])
        z = np.array([pt0[2], pt1[2], pt2[2]])
        f1 = si.interp1d(x, z, kind="slinear")

        x1 = np.arange(pt0[0], pt2[0], RES_AX_1)
        y1 = np.repeat(pt1[1], len(x1))
        z1 = np.array(f1(x1))

        X_ax = np.append(X_ax, x1)
        Y_ax = np.append(Y_ax, y1)
        Z_ax = np.append(Z_ax, z1)

        # Construct second segment (to OND) #
        # NOTE: Assumes temporal patches, projection is not suited for nasal

        # Declare mid-point between soma and OND as max-displacement and 3rd-pt
        # Declare centre of OND as the fourth (and final) point
        # The z-axis is the same as the goal z as the OND is assumed to encompass
        # the height of the GCL
        i = 1 if node[1] >= 0 else -1  # superior vs. inferior direction

        # Temporal/nasal interpolation differ as temporal must avoid foveola
        ecc_dummy = 0
        y_displaced = calc_axon_displacement(pt2[0])

        if (node[0] + ecc_dummy * (not DISPLACE)) >= 0:
            if abs(pt2[1]) > (0.75 * y_displaced):
                y_displaced = 0.75 * abs(node[1])

        pt3 = [(ond + pt2[0]) / 2, i * y_displaced + node[1], pt2[2]]
        pt4 = [ond, 0, pt3[2]]

        # Interpolate to project axon from the AIS to the OND
        x = np.array([pt2[0], pt3[0], pt4[0]])
        y = np.array([pt2[1], pt3[1], pt4[1]])
        f2 = si.interp1d(x, y, kind="quadratic")

        x2 = np.arange(pt2[0], pt4[0], RES_AX_2)
        y2 = np.array(f2(x2))
        z2 = np.repeat(pt2[2], len(x2))
        for i in range(len(x2)):
            x_mm = x2[i] / 1000
            y_mm = y2[i] / 1000
            if within_ellipse(x_mm, y_mm, OND_POS[0], OND_POS[1], OND_X, OND_Y):
                x2 = x2[: i + 1]
                y2 = y2[: i + 1]
                z2 = z2[: i + 1]
                break

        X_ax = np.append(X_ax, x2)
        Y_ax = np.append(Y_ax, y2)
        Z_ax = np.append(Z_ax, z2)

    circle_pit = Circle(
        (0, 0),
        radius=SLOPE,
        edgecolor="black",
        facecolor="mediumslateblue",
        linewidth=5,
        alpha=0.9,
        label="Foveola",
    )
    circle_ond = Circle(
        OND_POS,
        radius=(OND_X + OND_Y) / 2,
        edgecolor="black",
        facecolor="indianred",
        linewidth=5,
        alpha=0.9,
        label="Optic nerve",
    )

    circle_five_deg = Circle(
        (0, 0),
        radius=5 * ECC_DEG_TO_MM,
        edgecolor="goldenrod",
        facecolor="None",
        linewidth=10,
        ls="--",
        alpha=0.85,
    )

    circle_ten_deg = Circle(
        (0, 0),
        radius=10 * ECC_DEG_TO_MM,
        edgecolor="goldenrod",
        facecolor="None",
        linewidth=10,
        ls="--",
        alpha=0.85,
    )

    ax.annotate("5$^o$", xy=(3.33 * ECC_DEG_TO_MM, -0.1), xycoords="data", fontsize=72)
    ax.annotate("10$^o$", xy=(7.67 * ECC_DEG_TO_MM, -0.1), xycoords="data", fontsize=72)
    ax.add_patch(circle_five_deg)
    ax.add_patch(circle_ten_deg)
    ax.add_patch(circle_ond)
    ax.add_patch(circle_pit)
    ax.scatter(X_ax / 1000, Y_ax / 1000, c="k", s=30, label="Axon")
    ax.legend(loc="upper left", ncol=1, markerscale=2, fontsize=52)
    ax.set_xlim(-2.75, 3.80)
    ax.set_xlim(-3.75, 3.80)
    # ax.set_title("Axon pathways", pad=50)
    ax.set_xlabel("x ($mm$)")
    ax.set_ylabel("y ($mm$)")
    ax.locator_params(axis="both", nbins=N_TICKS)

    # Add secondary X-axis for ecc. in degrees
    curr_lims = plt.gca().get_xlim()
    ax3 = ax.twiny()
    ax3.set_xlabel("x (degrees)", labelpad=20)
    ax3.set_xlim(curr_lims)

    formatter = mticker.FuncFormatter(pos_and_neg_formatter)
    ax3.xaxis.set_major_formatter(formatter)
    ax3.locator_params(axis="both", nbins=N_TICKS)

    # Add secondary Y-axis for ecc. in degrees
    curr_y_lims = plt.gca().get_ylim()
    ax4 = ax.twinx()
    ax4.set_ylabel("y (degrees)", labelpad=20)
    ax4.set_ylim(curr_y_lims)
    ax4.yaxis.set_major_formatter(formatter)
    ax4.locator_params(axis="both", nbins=N_TICKS)

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/fig1-Axon-wide-pathways.jpg")
    plt.savefig(f"{figure_dir}/fig1-Axon-wide-pathways.pdf")
    plt.close()

    ####################### THICKNESS VERSUS ECCENTRICITY #####################

    GCL_fit = GCL_int(GCL_ecc)
    IPL_fit = IPL_int(IPL_ecc)
    GCL_fit = GCL_int(x_ecc)
    IPL_fit = IPL_int(x_ecc)
    fig1 = plt.figure(111)
    ax = fig1.add_subplot(111)
    ax.plot(x_ecc, GCL_fit, c="k", linestyle=":", linewidth=14, zorder=-20)
    ax.scatter(GCL_ecc, GCL_t, c="orange", marker="X", s=750, ec="black", lw=5)
    ax.plot(x_ecc, IPL_fit, c="k", linewidth=14, zorder=-20)
    ax.scatter(IPL_ecc, IPL_t, c="red", marker="d", s=750, ec="black", lw=5)

    # Dummy plot for combined legend labels
    ax.plot(
        np.nan,
        np.nan,
        c="k",
        mfc="red",
        label="IPL",
        marker="d",
        ms=40,
        mec="black",
        mew=5,
    )
    ax.plot(
        np.nan,
        np.nan,
        linestyle=":",
        c="k",
        mfc="orange",
        label="GCL",
        marker="X",
        ms=40,
        mec="black",
        mew=5,
    )

    ax.set_xlabel("Eccentricity (mm)")
    ax.set_ylabel("Thickness ($\\mu m$)")
    ax.set_xlim([-2.55, 2.55])
    ax.set_ylim([0, 60])
    ax.legend(loc="lower left", fontsize=52)

    # Add secondary x-axis for ecc. in degrees
    curr_lims = plt.gca().get_xlim()
    ax3 = ax.twiny()
    ax3.set_xlabel("Eccentricity (degrees)", labelpad=20)
    ax3.set_xlim(curr_lims)
    ax3.xaxis.set_major_formatter(formatter)
    ax3.locator_params(axis="both", nbins=N_TICKS)
    ax.locator_params(axis="both", nbins=N_TICKS)
    plt.savefig(f"{figure_dir}/fig1-GCL-IPL-thickness.jpg")
    plt.savefig(f"{figure_dir}/fig1-GCL-IPL-thickness.pdf")
    plt.close()

    #################### GC DENSITY VERSUS ECCENTRICITY #######################

    x_ecc_1d = np.linspace(0, 2.49, 1000)

    NF_fit = NF_int(x_ecc_1d)
    TF_fit = TF_int(-1 * x_ecc_1d)

    fig1 = plt.figure(1337)
    ax = fig1.add_subplot(111)
    ax.plot(
        -1 * x_ecc_1d,
        TF_fit / 1000,
        c="k",
        linewidth=14,
        zorder=-20,
    )
    ax.plot(
        x_ecc_1d,
        NF_fit / 1000,
        c="k",
        linewidth=14,
        zorder=-30,
    )
    ax.scatter(
        TF_ecc,
        TF_GC / 1000,
        c="red",
        marker="d",
        s=1200,
        ec="black",
        lw=5,
    )
    ax.scatter(
        NF_ecc,
        NF_GC / 1000,
        c="orange",
        marker="X",
        s=1200,
        ec="black",
        lw=5,
    )

    # Dummy plot for combined legend labels
    ax.plot(
        np.nan,
        np.nan,
        c="k",
        mfc="red",
        label="Temporal",
        marker="d",
        ms=40,
        mec="black",
        mew=5,
    )
    ax.plot(
        np.nan,
        np.nan,
        c="k",
        mfc="orange",
        label="Nasal",
        marker="X",
        ms=40,
        mec="black",
        mew=5,
    )

    ax.set_xlabel("Eccentricity (mm)")
    ax.set_ylabel("Number of RGC $(x1000/mm^2)$")
    ax.set_xlim([-2.55, 2.55])
    ax.set_ylim([0, 35])
    ax.legend(loc="lower left", fontsize=52)
    plt.locator_params(axis="both", nbins=N_TICKS)

    # Add secondary x-axis for ecc. in degrees
    curr_lims = plt.gca().get_xlim()
    ax3 = ax.twiny()
    ax3.set_xlabel("Eccentricity (degrees)", labelpad=20)
    ax3.set_xlim(curr_lims)
    ax3.xaxis.set_major_formatter(formatter)
    ax3.locator_params(axis="both", nbins=N_TICKS)
    plt.savefig(f"{figure_dir}/fig1-GC-density.jpg")
    plt.savefig(f"{figure_dir}/fig1-GC-density.pdf")
    plt.close()

    ################## DENDRITIC FIELD VERSUS ECCENTRICITY ####################

    # Calculate sizes for each x_ecc
    x_ecc = np.linspace(PIT, 3.00, 100)
    y_size = np.array(ecc_to_diameter(x_ecc))
    y_spacing = [mRGC_spacing(ecc=i) for i in x_ecc]

    # Plot
    fig1 = plt.figure(777)
    ax = fig1.add_subplot(111)
    ax.plot(
        x_ecc,
        y_size,
        c="k",
        lw=10,
        label="Diameter $\pm  \sigma$",
    )
    ax.fill_between(
        x_ecc,
        y_size - 0.05 * y_size,
        y_size + 0.05 * y_size,
        color="lightgrey",
        zorder=-20,
    )
    ax.plot(
        x_ecc,
        y_spacing,
        c="red",
        lw=10,
        ls="--",
        label="Spacing",
    )
    ax.plot(
        -x_ecc,
        y_size,
        c="k",
        lw=10,
    )
    ax.fill_between(
        -x_ecc,
        y_size - 0.05 * y_size,
        y_size + 0.05 * y_size,
        color="lightgrey",
        zorder=-20,
    )
    ax.plot(
        -x_ecc,
        y_spacing,
        c="red",
        lw=10,
        ls="--",
    )
    ax.set_xlabel("Eccentricity (mm)")
    ax.set_ylabel("Dendritic field diameter/spacing $(\\mu m)$")
    plt.legend(loc="upper left")
    plt.locator_params(axis="both", nbins=N_TICKS)
    log_yticks = [5, 10, 50, 100]
    ax.set_yticks(log_yticks)

    # Add secondary x-axis for ecc. in degrees
    curr_lims = plt.gca().get_xlim()
    ax3 = ax.twiny()
    ax3.set_xlabel("Eccentricity (degrees)")
    ax3.set_xlim(curr_lims)
    ax3.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_ticks([5, 10, 50])
    ax.set_ylim(0.1, 1.2 * np.array(ecc_to_diameter([3])))
    ax.set_xlim(-2.55, 2.55)
    ax3.set_xlim(-2.55, 2.55)
    ax.locator_params(axis="both", nbins=N_TICKS)
    ax3.locator_params(axis="x", nbins=N_TICKS)
    plt.savefig(f"{figure_dir}/fig1-dendritic-diameter.jpg")
    plt.savefig(f"{figure_dir}/fig1-dendritic-diameter.pdf")
    plt.close()

    plt.close("all")


def create_population(
    ecc_hor: float,
    ecc_ver: float = 0,
    jitter: bool = False,
    std: float = 5,  # was 1
    AXON: bool = False,
) -> Population:
    """
    Plot the population tile for the given eccentricity and patch size.
    - ecc_hor: the eccentricity (along the horizontal meridian) of interest.
    - ecc_ver: the eccentricity (along the vertical meridian) of interest.
    - jitter: toggle to add Gaussian noise to the coordinates for realism.
    - std: associated std for the noise used alongside `jitter` [% of mean].
    - AXON: toggle to include axon segments.
    """

    ########################## Determine parameters ###########################

    population = Population()
    ecc_abs = np.sqrt(ecc_hor**2 + ecc_ver**2)
    population.ecc_hor = ecc_hor
    population.ecc_ver = ecc_ver
    population.ecc_abs = ecc_abs

    # Measurements for hexagonal mosaic's width/height
    soma_mosaic_diameter = SOMA_D + BUFFER  # mm
    height = soma_mosaic_diameter  # equivalent to long diagonal of hex.
    width = 2 * hex_diagonal_to_apothem(height)  # = 2 * sqrt(3)/4 * height

    ############################# Place somas #################################

    coords_soma = np.empty((0, 3), dtype=float)  # Initialise coordinate list
    z_indexes = np.empty((0, 1), dtype=float)  # Initialise z-index list

    z = 0  # Set z-coordinate for first layer
    n_cells = 0  # Neuron/cell counter
    max_gc_in_column = 0  # tracker of maximum GC in one layer/column

    in_patch = True  # tracker to inform whether cells are within boundaries
    i = 0  # hex index/tiling counter

    # Create hexagonal mosaic through iterative placement of soma
    while in_patch:

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

            x, y = hex_coords(col, row, width, height)

            in_patch = still_inside_patch(x=x, y=y, ecc_hor=ecc_hor, ecc_ver=ecc_ver)
            in_pit = inside_foveola(x=x, y=y, ecc_hor=ecc_hor, ecc_ver=ecc_ver)
            if (not in_patch) or (in_pit):
                continue

            cur_ecc = xy_to_eccentricity(x=x, y=y, ecc_hor=ecc_hor, ecc_ver=ecc_ver)
            gc_in_column = ecc_to_ngcl(cur_ecc)  # No. of stacked GCs here

            # Init column, adding its x- and y-values with z-placeholders
            column = np.array([[x, y, 9999]] * gc_in_column)  # 9999 overwritten

            # Determine the appropriate z/depth values for this column
            z_cur = np.arange(gc_in_column)  # layer indexes
            # Scale (offset) layer index by GCL thickness
            # Offset pushes cells into GCL, away from IPL
            z_cur = (z_cur + 0.5) * -GCL_int(cur_ecc) / gc_in_column
            column[:, 2] = z_cur  # Update the column's z-values

            # Update coordinate data structures
            coords_soma = np.append(coords_soma, column, axis=0)
            z_indexes = np.append(z_indexes, np.arange(gc_in_column))

            # Update trackers
            n_cells += gc_in_column
            max_gc_in_column = max(max_gc_in_column, gc_in_column)

        i += 1

    if DISPLACE:
        coords_soma[:, 0] += ecc_hor
        coords_soma[:, 1] += ecc_ver

    # Check if duplicates are (somehow) present
    set_nrns = np.unique(coords_soma, axis=1)
    assert len(set_nrns) == len(coords_soma), "Duplicate neurons present."

    # Displace every 2nd layer in the x-direction to create a more realistic
    # bunching/layering, thereby creating a more 3D hexagonal mosaic
    # Cells are displaced in the negative/temporal direction as they are
    # laterally displaced away from the fovea anyway - this helps emulate so
    for layer in np.arange(1, max_gc_in_column + 1, 2):
        to_shift = np.where(z_indexes == layer)
        coords_soma[to_shift, 0] -= width * 0.50

    # Convert X/Y to microns; z has been accounted for by use of GCL thickness
    coords_soma[:, 0:2] *= 1000

    # Save base x-/y-values, lowest depth of GCs, and z_indexes before jitter
    population.x_base = copy.deepcopy(coords_soma[:, 0])
    population.y_base = copy.deepcopy(coords_soma[:, 1])
    population.z_floor = np.amin(coords_soma[:, 2])
    population.z_idx = copy.deepcopy(z_indexes)

    # Add gaussian noise (mean = 0) to create a jitter effect
    if jitter:
        std_soma = SOMA_D * 1000 * std / 100  # * 1000 mm to um, / 100 % to raw
        coords_soma += np.random.normal(0, std_soma, size=np.shape(coords_soma))

    # Save this in the population
    population.size = n_cells
    population.add_somas(coords_soma=coords_soma)
    print(f"INFO: Population size is {population.size}.")
    if n_cells >= 10000:
        # assert CROP_POPULATION, (
        print(
            "Population has exceeded 9999 cells - consider "
            "changing PATCH_SIZE or introducing limits using CROP_POPULATION."
        )

    ################# Create dendrites at appropriate layers ##################

    # Compute no. of ON / OFF cells
    N_OFF = int(np.ceil(n_cells / (1 + ON_OFF)))
    N_ON = n_cells - N_OFF

    population.n_on = N_ON
    population.n_off = N_OFF

    # The eccentricity associated with the population's centre is used to
    # determine the diameters/measurements which define the ON/OFF hex. mosaics
    # However, each tree has its appropriate position (and thus eccentricity)
    # accounted for in determining its diameter and thus spanning its tree.

    # Height is equivalent to hex's long-diagonal
    # h_on_um = ecc_to_diameter(ecc=ecc_hor, cell_type="ON", enforce_min=True)[0]
    h_on_um = mRGC_spacing(ecc_abs)
    h_on = h_on_um / 1000  # convert micron to mm
    h_on /= CF  # scale by coverage factor
    w_on = 2 * hex_diagonal_to_apothem(h_on)

    # h_off_um = ecc_to_diameter(ecc=ecc_hor, cell_type="OFF", enforce_min=True)[0]
    h_off_um = mRGC_spacing(ecc_abs)
    h_off = h_off_um / 1000  # convert micron to mm
    h_off /= CF  # scale by coverage factor
    w_off = 2 * hex_diagonal_to_apothem(h_off)

    # Create ON coords #
    coords_on = np.empty((0, 3), float)
    i = 0  # Hex tiling counter

    while N_ON:

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
            x, y = hex_coords(col, row, w_on, h_on)

            cur_ecc = xy_to_eccentricity(x=x, y=y, ecc_hor=ecc_hor, ecc_ver=ecc_ver)
            z = zIPL(cur_ecc)

            coords_on = np.concatenate((coords_on, [[x, y, z]]), axis=0)
            N_ON -= 1
            if N_ON == 0:
                break

        i += 1

    # Create OFF coords (as above, but with the appropriate hex width/height)
    coords_off = np.empty((0, 3), float)
    i = 0  # Tiling counter

    while N_OFF:

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
            x, y = hex_coords(col, row, w_off, h_off)

            cur_ecc = xy_to_eccentricity(x=x, y=y, ecc_hor=ecc_hor, ecc_ver=ecc_ver)
            z = zIPL(cur_ecc, ON=False)

            coords_off = np.concatenate((coords_off, [[x, y, z]]), axis=0)
            N_OFF -= 1
            if N_OFF == 0:
                break

        i += 1

    if DISPLACE:
        coords_on[:, 0] += ecc_hor
        coords_off[:, 0] += ecc_hor
        coords_on[:, 1] += ecc_ver
        coords_off[:, 1] += ecc_ver

    coords_on[:, 0:2] *= 1000  # mm to microns for x,y (z is already micron)
    coords_off[:, 0:2] *= 1000  # mm to microns for x,y (z is already micron)

    # If noisy, add jitter to X, Y of ON/OFF. Z intact as depth is randomised.
    if jitter:
        std_on = h_on_um * std / 100
        coords_on[:, 0:2] += np.random.normal(
            0, std_on, size=np.shape(coords_on[:, 0:2])
        )
        std_off = h_off_um * std / 100
        coords_off[:, 0:2] += np.random.normal(
            0, std_off, size=np.shape(coords_off[:, 0:2])
        )

    # Population update
    population.coords_on = coords_on
    population.coords_off = coords_off

    print("INFO: Soma and ON/OFF stratification points placed.")

    # Title for plots
    title = f"Eccentricity x={ecc_hor:.2f}, y={ecc_ver:.2f} mm; N = {n_cells}"
    if jitter:
        title += f"; jitter (std = {std:.2f}%)"

    ##################### Connect somas with dendrites  #######################
    #################### and generate dendritic fields #######################

    # NOTE: cannot be parallelised as it involves iteratively assigning soma
    # to dendritic stratification points (need to avoid re-using same point)
    connect_soma_to_dendrite(
        coords_soma=coords_soma,
        coords_on=coords_on,
        coords_off=coords_off,
        population=population,
    )

    print("INFO: Soma connected to dendrites.")

    #################### Project axons from soma to OND  ######################

    if AXON:

        if PARALLEL:
            print("INFO: Using parallel axon projection.")
            Parallel(n_jobs=num_cores, require="sharedmem", prefer="threads")(
                delayed(population.project_axon)(
                    cell_id=i,
                    ecc_hor=ecc_hor,
                    ecc_ver=ecc_ver,
                )
                for i in range(population.size)
            )

        else:
            for i in range(population.size):
                population.project_axon(
                    cell_id=i,
                    ecc_hor=ecc_hor,
                    ecc_ver=ecc_ver,
                )

        print("INFO: Axon projection complete.")

    ###################### Crop population at limits ##########################

    # delete pop.NEURON, reduce pop.size
    if CROP_POPULATION:
        to_delete = []
        del_on, del_off = 0, 0
        for i, nrn in enumerate(population.neurons):
            x, y, _ = nrn.soma_loc
            if x < CROP_X[0] or x > CROP_X[1] or y < CROP_Y[0] or y > CROP_Y[1]:
                to_delete.append(i)
                if nrn.cell_type == "ON":
                    del_on += 1
                elif nrn.cell_type == "OFF":
                    del_off += 1
        population.neurons = np.delete(population.neurons, to_delete)
        population.size = len(population.neurons)
        n_cells = population.size
        population.n_on -= del_on
        population.n_off -= del_off
        if population.size > 9999:
            print(
                "Population has exceeded 9999 cells - consider changing "
                "PATCH_SIZE or CROP_POPULATION to reduce computational burden."
            )

    print(f"INFO: Number of cells finalised...{population.size = }.")

    ######################## Create plot directory ############################

    path = f"{root_distr}/results/trial_{TRIAL:03d}/tile_images/"
    if not os.path.isdir(path):
        os.makedirs(path)

    ###################### Create 3D population plots #########################

    if PLOT:
        population.create_3d_plot_for_presentations(path=path)
        # population.create_simplified_mosaic_plots(path=path)
        print("INFO: 3D plots saved.")

    #################### QUANTIFY CELLS INTO TEXT FILE ! ######################

    # 0. id of compartment (0-soma, 1-dendrite, 2-hillocks, 3-AIS, 4-axon),
    # 1. x coor, (of current node),
    # 2. y coor, (of current node),
    # 3. z coor, (of current node),
    # 4. x coor, (of previous node),
    # 5. y coor, (of previous node),
    # 6. z coor (of previous node),
    # 7. ID of current segment,  (start from 0 for 'soma' segment)
    # 8. ID of previous (parent) segment (start from NaN for 'soma' segment)
    # 9. length of each segment
    print("INFO: Saving cells as .txt files...")
    create_dir(sub_dir=TXT_PATH, get_cwd=False)

    # Create reproducibility files
    trial_root = TXT_PATH.split("/cellText")[0]
    create_trial_file(trial_root, population.size, std * jitter)

    if PARALLEL:
        print("INFO: Using parallel cell creation.")
        Parallel(n_jobs=num_cores, require="sharedmem", prefer="threads")(
            delayed(population.create_cell)(
                cell_id=i,
            )
            for i in range(population.size)
        )

    else:
        for i in range(population.size):  # Iterate over all cells
            population.create_cell(cell_id=i)

    print(f"INFO: ...{population.size} cells successfully saved as .txt files.")

    # Create population's .csv file
    population.create_csv()
    print("INFO: ...population successfully saved as .csv file.")

    # Pickle population and save so
    f = f"{trial_root}/trial_{TRIAL:03d}.pkl"
    with open(f, "wb") as output:
        pickle.dump(population, output, pickle.DEFAULT_PROTOCOL)

    return population


def create_dendritic_tree(
    ecc_hor: float,
    x_s: float,
    y_s: float,
    z_s: float,
    neuron: Neuron,
    ecc_ver: float = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a dendritic tree based on the diameter at eccentricity (`ecc_hor`,
    `ecc_ver`) (mm). `x_s`, `y_s`, `z_s` refers to the root position, the start
    of the tree and the end of the soma-dendrite path (microns).

    `ecc_hor`, `ecc_ver`, `x_s`, and `y_s` are used to determine the absolute
    eccentricity.

    The generator utilises a distribution of random carrier points within the
    boundary of a given dendritic field diameter and applies a minimum spanning
    tree algorithm to iteratively generate dendritic branches by connecting
    unused carrier points to node points of the tree (see Bai et al. 2015 for
    more).

    If the Neuron's `cell type` is "ON", the cell is treated as an ON mRGC,
    else OFF mRGC. Size disaprities (if any) are handled by ecc_to_diameter().

    The Neuron is given properties to characterise its dendritic tree:
        - `tree_diameter` - the tree's approximate hexagonal diameter (um).
        - `tree_area` - the tree's approximate hexagonal area (um^2).
        - `tree_loc` - (x,y,z) position of branching point.
        - `tree_poly` - matplotlib RegularPolygon of hexagonal boundary.

    Code is based on the neuron_morph_hex_updated.m MATLAB script.
    """

    # Calculate the eccentricity at this dendritic root/branching point.
    # x_s, y_s / 1000 to convert all into mm
    ecc = xy_to_eccentricity(
        x=x_s / 1000, y=y_s / 1000, ecc_hor=ecc_hor, ecc_ver=ecc_ver
    )

    # Dendritic diameter [um] ... [0] required as it is returned as a list
    D = np.squeeze(ecc_to_diameter(ecc, cell_type=neuron.cell_type)[0])

    ############################ Parameters ###################################

    N = 10  # Number of carrier points
    jit = 0.05  # Jitter in generated positions  # was 0.07 then 0.05
    seg = 1.50  # Neuron segment length [um] # was R / 25 then 1.25
    bf = 0.5  # Balance factor (see Bai et al.)  # was 0.5
    z_jit = 0.75  # Jitter in the depth of the dendritic tree # was 0.7
    ht = z_s + Z_BRANCH  # Branches along a distance of Z_BRANCH from cur. end

    ###########################################################################

    if NO_DENDRITES:  # reduce dendrites to speed up run-times (& for reviewer q)
        D = 0.01
        N = 2

    R = D / 2  # Mean annular radius (circumcircle radius of hexagon) (um)
    sd = R * 0.05  # Annular standard deviation (um)

    # Add Neuron's properties - radius, loc, matplotlib patch
    neuron.tree_diameter = D
    neuron.tree_area = diagonal_to_hex_area(D)
    neuron.tree_loc = np.array([x_s, y_s, z_s])
    neuron.tree_poly = RegularPolygon(
        xy=[x_s, y_s],
        numVertices=6,
        radius=R,
        orientation=np.deg2rad(90),
        facecolor="indianred" if neuron.cell_type == "ON" else "darkorchid",
        edgecolor="black",
        linewidth=7,
    )
    neuron.z_terminal_max = z_s
    neuron.z_terminal_min = 1e9

    # Generate random carrier points on dendrites
    theta = np.squeeze(2 * np.pi * np.random.normal(size=(1, N)))
    r = np.squeeze(R * np.random.normal(size=(1, N)))
    x_dn = np.squeeze(r * np.cos(theta))
    y_dn = np.squeeze(r * np.sin(theta))

    # Hexagon tile and bounding box
    # http://www.playchilla.com/how-to-check-if-a-point-is-inside-a-hexagon
    h_hex = R * np.cos(np.pi / 6)  # Horizontal boundary
    v_hex = R / 2  # Vertical boundary
    idx = np.where(abs(x_dn) > h_hex)  # Find points beyond bounding box
    if np.size(idx):
        idx = np.squeeze(idx)

    # Remove points beyond boundary box
    x_dn = np.delete(x_dn, idx)
    y_dn = np.delete(y_dn, idx)

    # Another boundary box check - following from the link above still...
    p_vt = [h_hex, v_hex]
    n_pt = [-v_hex, -h_hex]
    # d : dot product test to see if point exceeds boundary box
    d = n_pt[0] * (abs(x_dn) - p_vt[0]) + n_pt[1] * (abs(y_dn) - p_vt[1])

    # d >= 0 point is inside hex, i.e., corners/boundaries exceeded if d < 0
    idx = np.where(d < 0)
    if np.size(idx):
        idx = np.squeeze(idx)
    x_dn = np.delete(x_dn, idx)
    y_dn = np.delete(y_dn, idx)

    while len(x_dn) != N:

        # Generate random carrier angle and radius
        theta = 2 * np.pi * np.random.normal()
        r = R + sd * np.random.normal()

        x_tt = np.array([r * np.cos(theta)])
        y_tt = np.array([r * np.sin(theta)])

        if abs(x_tt) <= h_hex:

            d = n_pt[0] * (abs(x_tt) - p_vt[0] + n_pt[1] * abs(y_tt) - p_vt[1])

            if d >= 0:
                x_dn = np.concatenate((x_dn, x_tt))
                y_dn = np.concatenate((y_dn, y_tt))

    # Add the end-points of the soma-dendrite tree (x,y) to shift these all
    x_dn += x_s
    y_dn += y_s

    # Create the z-coordinates
    z_dn = ht + np.squeeze(z_jit * np.random.normal(size=(1, N)))
    idx = np.where(abs(z_dn - ht) > z_jit)
    if np.size(idx):
        idx = np.squeeze(idx)
    z_dn = np.delete(z_dn, idx)

    while len(z_dn) != N:
        tt = z_jit * np.random.normal()
        if abs(tt) < z_jit:
            z_dn = np.concatenate((z_dn, np.array([ht + tt])))

    # Create neural segments by finding closest upward point

    # Initialise root node - the following is used to represent a node
    # [0: pt label,
    # 1: x coor,
    # 2: y_coor,
    # 3: z_coor,
    # 4: connecting pt label (idx in neuron array),
    # 5: total length of branch till this point]
    neuron_tree = np.array([0, x_s, y_s, z_s, -1, 0])
    points_left = N
    nodes = 1
    x, y, z = x_dn, y_dn, z_dn
    neuron.z_diff = copy.deepcopy(z_dn - z_s)

    while points_left > 0:

        best_dist = 1e9  # Very large number
        det_jitt = 0  # control of jitter in case exceeds max segment length

        for ii in range(nodes):

            if nodes == 1:
                curr = neuron_tree
            else:
                curr = neuron_tree[ii]

            # Distance measures #
            # 1) Distance between points and curr
            dist_1 = np.sqrt(
                (x - curr[1]) ** 2 + (y - curr[2]) ** 2 + (z - curr[3]) ** 2
            )
            # 2) Branch length thus far
            dist_2 = np.array(dist_1 + curr[5])
            # Distance optimisation based on bf
            dist = (1 - bf) * dist_1 + bf * dist_2

            # Judgment 1 - find points below current and ignore them
            idx = np.where(z < curr[3])
            if np.size(idx):
                idx = np.squeeze(idx)
                dist[idx] = 1e9

            # Find closest (already placed) node
            min_dist, carrier_pt = dist.min(), dist.argmin()

            # Compare with other nodes to see which is the true closest
            if min_dist < best_dist:
                best_dist = min_dist
                best_dist_1 = dist_1[carrier_pt]
                best_pt = carrier_pt
                best_node = ii

        # Get best node for processing later
        if nodes == 1:
            best = neuron_tree
        else:
            best = np.squeeze(neuron_tree[best_node])

        # If length of segment is within the permitted length
        if best_dist_1 <= seg:
            # Create new node's xyz
            node_x = x[best_pt]
            node_y = y[best_pt]
            node_z = z[best_pt]

            # Add this to the neuron structure
            node = np.array(
                [nodes, node_x, node_y, node_z, best_node, best[5] + best_dist_1],
            )
            neuron_tree = np.vstack((neuron_tree, node))

            # Remove this carrier point
            x = np.delete(x, best_pt)
            y = np.delete(y, best_pt)
            z = np.delete(z, best_pt)

            points_left -= 1

        # Segment length was too long
        else:

            x0 = best[1]
            y0 = best[2]
            z0 = best[3]

            loop_idx = 1
            loop_seg = seg  # dynamic segment length to avoid infinite loops

            while det_jitt != 1:

                jitter = jit * np.random.normal(size=(2, 1))
                node_x = x0 + seg * (x[best_pt] - x0) / best_dist_1 + jitter[0]
                node_y = y0 + seg * (y[best_pt] - y0) / best_dist_1 + jitter[1]
                node_z = z0 + seg * (z[best_pt] - z0) / best_dist_1

                # Distance between jitter node and branch node
                jitt_dis = np.sqrt(
                    (x0 - node_x) ** 2 + (y0 - node_y) ** 2 + (z0 - node_z) ** 2
                )

                if jitt_dis <= loop_seg:
                    node = np.array(
                        [nodes, node_x, node_y, node_z, best_node, best[5] + loop_seg],
                        dtype=object,
                    )
                    neuron_tree = np.vstack((neuron_tree, node))
                    det_jitt = 1

                elif loop_idx % 100 == 0:
                    loop_seg += 0.05
                    assert (
                        loop_seg <= 1.75
                    ), f"ERROR: Dendritic tree could not be created with segment length <= 1.75 um."
                loop_idx += 1

        neuron.z_terminal_max = max(neuron.z_terminal_max, node_z)
        neuron.z_terminal_min = min(neuron.z_terminal_min, node_z)
        nodes += 1

    # Turn this neuron data into SWC format (as follows)
    # 0. id of compartment (0-soma, 1-dendrite, 2-hillocks, 3-AIS, 4-axon),
    # 1. x coor, (of current node),
    # 2. y coor, (of current node),
    # 3. z coor, (of current node),
    # 4. x coor, (of previous node),
    # 5. y coor, (of previous node),
    # 6. z coor (of previous node),
    # 7. ID of current segment,  (start from 0 for 'soma' segment)
    # 8. ID of previous (parent) segment (starts from -1 for 'soma' segment)
    # 9. length of each segment

    # Place the first node wrt tree root (end of soma-dendrite 'stalk')
    # First entry into neuron array [index 0] is the end of stalk so we use
    # index 1 for the first new point
    ID, x1, y1, z1 = neuron_tree[1][0:4]
    length = np.sqrt((x1 - x_s) ** 2 + (y1 - y_s) ** 2 + (z1 - z_s) ** 2)
    tree = np.array([1, x1, y1, z1, x_s, y_s, z_s, ID, ID - 1, length], dtype=object)
    neuron_tree = np.squeeze(neuron_tree)
    id_start = ID  # to correct ID offset after collecting all nodes
    ID += 1

    # Extract prev. connection (parent) IDs
    connect_id = []
    for i in neuron_tree[:, 4]:
        connect_id.append(int(i))

    # 0th point is ignored (ref; not new), 1st is added (above), so start at 2
    # NOTE: neuron_tree is NOT in SWC index formatting, the new node(s) are
    for i in range(2, nodes):
        # Current x, y, z
        x1, y1, z1 = neuron_tree[i][1:4]
        # Connecting point
        conn_idx = connect_id[i]
        x2, y2, z2 = neuron_tree[conn_idx][1:4]
        # Segment length
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        # Add this to the structure
        node = np.array([1, x1, y1, z1, x2, y2, z2, ID, conn_idx, length], dtype=object)
        tree = np.vstack((tree, node))
        ID += 1

    # Correct the curr/prev point IDs as per the starting offset
    tree[:, 7] -= id_start
    tree[:, 8] -= id_start

    # Structure point data into neat 1D numpy arrays
    x = np.array([])
    y = np.array([])
    z = np.array([])

    for i in range(np.shape(neuron_tree)[0]):
        x = np.append(x, neuron_tree[i, 1])
        y = np.append(y, neuron_tree[i, 2])
        z = np.append(z, neuron_tree[i, 3])

    return x, y, z, tree


def connect_soma_to_dendrite(
    coords_soma: list,
    coords_on: list,
    coords_off: list,
    population: Population,
) -> None:
    """
    Returns coordinates for points which constitute the connections between
    the input soma coordinates and MGC ON/OFF coordinates. The dendritic tree
    is also added but this is performed by create_dendritic_tree(...).
    The soma is connected to the closest (Eucledian XY distance) dendrite.
    z-distance is ignored because that of the potential bias associated with
    different stratification depths.

        - `coords_soma`: a Nx3 list of the coords of each RGC soma; (x,y,z)-by-N.
        - `coords_on`: a Nx3 list of the coords of each ON dendritic branching pt.
        - `coords_off`: a Nx3 list of the coords of each OFF dendritic branching pt.
        - `population`: the Population object associated with said cells.

    The soma-to-dendrite connections and dendritic trees are added to the
    Neuron objects of the Population. The soma-to-dendrite points are saved
    under the Neuron as x_sd, y_sd, z_sd. The dendrite points are saved under
    the Neuron as x_dn, y_dn, z_dn.
    """

    ecc_hor = population.ecc_hor
    ecc_ver = population.ecc_ver

    # Concatenate on and off arrays to form dendrite array if no dend array
    coords_dend_roots = np.concatenate((coords_on, coords_off), axis=0)

    # Types is a boolean array indicating ON if True, OFF if False
    types = np.concatenate((np.ones(len(coords_on)), np.zeros(len(coords_off))))

    assert len(coords_soma) == len(
        coords_dend_roots
    ), "INFO: Number of somas does not equal the number of ON + OFF cells."

    # Vector segmentation
    N1 = L_SD  # No. of connection points (initial segment [xy])
    N2 = N1  # No. of connection points (second segment [z])

    # First-points (start-points) are ignored to avoid overlap
    # (they are already included in the neuron as a soma point).
    N1 += 1
    N2 += 1

    for idx, node in enumerate(coords_soma):
        if idx % 1000 == 0:
            print(f"INFO: Connecting cell {idx} soma to dendrites...")
        node[0] -= SOMA_D * 1e3  # shifts branch to end of soma

        # Find closest dendrite bunch for this soma (consider only xy)
        closest_idx = distance.cdist([node[0:2]], coords_dend_roots[:, 0:2]).argmin()
        c_type = "ON" if types[closest_idx] else "OFF"
        population.neurons[idx].cell_type = c_type

        # Define points and direction vector, d
        pt1 = node  # Soma (start)
        pt2 = coords_dend_roots[closest_idx]  # Dendrite (end)

        d = pt2 - pt1  # Direction vector

        # Construct initial segment (xy displacement to match dend's z)
        i = np.linspace(0, 1.00, N1)[1::]
        x1 = pt1[0] + i * d[0]
        y1 = pt1[1] + i * d[1]
        z1 = pt1[2] + i * d[2] * 0  # *0 as we do not want any z-displacement yet

        # Construct second segment (only z displacement up to dendrite)
        i = np.linspace(0, 1.00, N2)[1::]  # ignore 1st-pt (last seg end-pt)
        x2 = np.squeeze(pt2[0] + i * d[0] * 0)
        y2 = np.squeeze(pt2[1] + i * d[1] * 0)
        z2 = np.squeeze(pt1[2] + i * d[2])

        # Construct dendritic tree
        # TODO: consider parallelising this
        x_dn, y_dn, z_dn, tree_node = create_dendritic_tree(
            ecc_hor=ecc_hor,
            ecc_ver=ecc_ver,
            x_s=x2[-1],
            y_s=y2[-1],
            z_s=z2[-1],
            neuron=population.neurons[idx],
        )

        # Update neuron population
        population.neurons[idx].x_sd = np.array([x1, x2]).flatten()
        population.neurons[idx].y_sd = np.array([y1, y2]).flatten()
        population.neurons[idx].z_sd = np.array([z1, z2]).flatten()

        population.neurons[idx].x_dn = x_dn
        population.neurons[idx].y_dn = y_dn
        population.neurons[idx].z_dn = z_dn

        population.neurons[idx].tree_node = tree_node

        # Remove this dendrite from list of dendrites (avoid double-up)
        coords_dend_roots = np.delete(coords_dend_roots, closest_idx, axis=0)
        types = np.delete(types, closest_idx, axis=0)


################################### MAIN ######################################


def main(argv) -> Population:

    global TRIAL
    global TXT_PATH
    global LYR_MAX

    if len(argv) == 2:
        TRIAL = int(argv[1])

    # .txt file path and trial data safe-check
    root = os.path.abspath(os.getcwd()).split("/RGC")[0]
    TXT_PATH = f"{root}/RGC-fovea/results/trial_{TRIAL:03d}/cellText/"

    while os.path.isdir(TXT_PATH) and WARN_USER:
        print("\n*** TRIAL ID is %.3d ***\n" % TRIAL)
        print("INFO: Pre-existing data for this trial has been found.")
        print("INFO: Failure to update the TRIAL field may overwrite data.")
        print("\nCommands - [Y]: proceed, [N]: update TRIAL, [_]: abort")
        response = input("Do you still wish to continue? [Y/N/_] : ")
        assert response in ["Y", "N"], "ABORTING - Trial number not updated!"
        if response == "Y":
            response = input("Are you sure? [Y/N] : ")
        assert response in ["Y", "N"], "ABORTING - invalid entry!"
        if response == "Y":  # Delete all .txt, .swc, .hoc
            shutil.rmtree(TXT_PATH)
            parent = TXT_PATH.split("cellText/")[0]
            dir_hoc = f"{parent}cellHoc/"
            dir_swc = f"{parent}cellSwc/"
            if os.path.isdir(dir_hoc):
                shutil.rmtree(dir_hoc)
            if os.path.isdir(dir_swc):
                shutil.rmtree(dir_swc)
            print(
                f"INFO: Deleted all pre-existing .hoc/.txt/.swc data for trial {TRIAL:03d}!\n"
            )
        elif response == "N":
            TRIAL = int(input("\nInput new TRIAL ID: "))
            TXT_PATH = f"{root}/RGC-fovea/results/trial_{TRIAL:03d}/cellText/"

    # Check that the patch size will not intrude into the foveal pit/slope
    validate_setup(ecc_hor=ECC, patch_size=PATCH_SIZE, ecc_ver=ECC_Y)
    print("\nINFO: Experimental setup validated!")
    print(f"\n*** COMMENCING TRIAL {TRIAL:03d} ({ECC} mm, {PATCH_SIZE = } mm) ***\n")

    patch_start = time.time()
    pop_result = create_population(ecc_hor=ECC, ecc_ver=ECC_Y, AXON=AXON, jitter=JITTER)
    patch_end = time.time()

    print(f"INFO: Cell creation took {(patch_end - patch_start):.2f} s.")

    # Update TRIAL in other files
    create_tile.TRIAL = TRIAL
    mass_generate.TRIAL = TRIAL

    # Call mass_generate to create the .swc and .hoc files
    mass_generate.main()

    # Call create_tileto create the tile's init-tile*.hoc and tile*.hoc.
    create_tile.main()

    print(f"\n*** TRIAL {TRIAL:03d} COMPLETE ***\n")
    return pop_result


if __name__ == "__main__":
    # create_plots()  # create fig1 plots
    main(sys.argv[:])
