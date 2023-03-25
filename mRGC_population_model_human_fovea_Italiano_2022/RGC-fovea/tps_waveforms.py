"""

A representation of waveforms/cells using classes. 

Intended for automated generation of simulation .hoc files and subsequent
simulation and analysis using in-built functions and helpers such as 
spike_detection.py. 

Embeds all key analytical metrics within the Generation and Agent objects.

@author: M.L. Italiano 

"""

import os
import re
import pickle
import copy
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.ticker as mticker
from matplotlib.patches import Circle, Patch

if os.environ.get("DISPLAY", "") == "":
    # print('INFO: No display found. Using non-interactive Agg backend')
    mpl.use("Agg")
import matplotlib.pyplot as plt

from typing import Union, Tuple, Dict, List
from joblib import Parallel, delayed
import multiprocessing

from helpers import spike_detection, stim_calculations
from helpers.spike_detection import butter_lowpass, TRIG_AMP
from helpers.common import create_dir, silent_remove
from helpers.common import SIN, EXP, BIPHASIC, ANODIC, TRIANGLE
from foveal_tiles import SOMA_BIN


results_path = ""  # trial/sim directory (edited by stimulate_tile.py)

N_WORKERS = multiprocessing.cpu_count()  # no. of processors for mpi

dt_nrn = 0.025 * 1e-3  # time between NEURON samples [s]
t_pre = 10  # sim. time before start of stim. waveform [ms]
t_post = 20  # sim. time after end of stim. waveform [ms]

n_step = 10  # number of discretised values per phase (for exp/sin)
t_tau = 0.1  # time-constant wrt t_pw (pulse duration) (for exp waveform)

AMP = 0
N_phase = 1

t_pw = 0.1  # stim. pulse width [ms]
t_interphase = 0  # time/gap between phases [ms] - rect. waves only
t_interpulse = 5  # time between adjacent stim. pulses [ms]
n_pulse = 0  # number of biphasic pulses

r_elec = 5  # default [microns] - but extracted from stimulation-info.txt


############################# PLOTTING STUFF ##################################

plt.style.use("./helpers/plots.mplstyle")
plt.set_cmap("inferno")

cmap = plt.cm.get_cmap("inferno")
cmaplist = [cmap(i) for i in range(cmap.N)]
cmaplist[0] = mpl.colors.to_rgba("lightgrey")
cmap = mpl.colors.LinearSegmentedColormap.from_list("mcm", cmaplist, cmap.N)

m_on = mpl.lines.Line2D(
    [], [], color="k", marker="o", linestyle="None", markersize=25, label="ON"
)
m_off = mpl.lines.Line2D(
    [], [], color="k", marker="s", linestyle="None", markersize=25, label="OFF"
)

view_elevations = [0, 90]
out_name = "summary"
pic_folder = "plots"

NBINS = 6

########################### HELPER FUNCTIONS ##################################


def get_hex_surround_positions(center: np.ndarray, pitch: float) -> list:
    """
    Given a `center` (x, y, z) position and `pitch`, returns a numpy array of
    the positions of the surrounding hex-polar electrodes.
    """
    x = center[0]
    y = center[1]

    dx = np.cos(np.deg2rad(60)) * pitch
    dy = np.sin(np.deg2rad(60)) * pitch

    surround = []
    surround.append([x + pitch, y])  # EAST
    surround.append([x + dx, y + dy])  # NORTH-EAST
    surround.append([x + dx, y - dy])  # SOUTH-EAST
    surround.append([x - dx, y - dy])  # SOUTH-WEST
    surround.append([x - pitch, y])  # WEST
    surround.append([x - dx, y + dy])  # NORTH-WEST

    return surround


def get_patches(
    pos: np.ndarray,
    radius: float,
    hex_polar: bool,
    hex_pitch: float = 30,
    **kwargs,
) -> List[Circle]:
    """
    Creates electrode patches at the given position, `pos`. The radius of the
    electrode is `radius`.
    If `hex_polar` is True, additional surrounding hex-polar electrodes are
    added as per the pitch specified in `hex_pitch`.
    **kwargs allows for other patch properties to be passed.
    """

    patch_options = {
        "facecolor": "lightgrey",
        "edgecolor": "black",
        "fill": True,
        "linewidth": 10,
        "alpha": 0.75,
    }

    patch_options.update(kwargs)

    patches = []
    patches.append(
        Circle(
            (pos[0], pos[1]),  # POSITION
            radius,  # radius
            **patch_options,
        )
    )

    if hex_polar:
        surround_pos = get_hex_surround_positions(center=pos, pitch=hex_pitch)
        for hex_pos in surround_pos:
            patches.append(
                Circle(
                    (hex_pos[0], hex_pos[1]),  # POSITION
                    radius,  # radius
                    facecolor=patch_options["facecolor"],
                    edgecolor=patch_options["edgecolor"],
                    linewidth=patch_options["linewidth"] * 0.85,
                    alpha=patch_options["alpha"] * 0.95,
                    fill=True,
                    # linestyle="--",
                )
            )

    return patches


############################ WAVEFORM STIMULI ##################################


class Stimulus:
    """
    Defines stimulation parameters (including stimulus type [biphasic, exp,
    sinusoidal]) and provides functions for plotting and writing .hoc
    procedures.
    Note that amplitudes are divided by 1000 to convert NEURON's default mA
    to uA. Amplitudes are negated to prioritise cathodic-first pulses.
    """

    def __init__(self, stim_type: str):

        # Stimulation amplitude (wrt standard biphasic cathodic-first)...
        # ...anodic-first needs to be negated
        self.AMP = AMP if stim_type != ANODIC else (-1 * AMP)

        # Stimulation waveform type
        self.stim_type = stim_type

        # NEURON/simulation timing parameters
        self.t_pre = t_pre  # sim. time before start of stim. waveform [ms]
        self.t_post = t_post  # sim. time after end of stim. waveform [ms]
        self.dt_nrn = dt_nrn  # time between NEURON samples [s]

        # Stimulation waveform timing
        self.t_pw = t_pw  # stim. pulse width [ms]
        self.t_interphase = t_interphase  # t between phases [ms], rect. waves only
        self.t_interpulse = t_interpulse  # time between adjacent stim. pulses [ms]
        self.n_pulse = n_pulse  # no. of biphasic pulses
        self.n_step = n_step  # no. of discretised values per phase (exp/sin)
        self.t_tau = t_tau  # time-constant wrt t_pw (pulse duration) (exp)
        self.N_phase = 1 if stim_type != ANODIC else N_phase  # pw_an / pw_cat

        # Calculate charge
        self.q_ph = self.calc_charge()  # charge per phase in uC

    def calc_charge(self) -> float:
        """
        Calculates the equivalent charge deposited in uC for a rectangular
        pulse of pulse width, t_pw (ms), and amplitude, AMP (uA).
        """
        return abs(self.AMP) * self.t_pw * (1e-3)

    def biphasic_stim_wave(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a biphasic waveform and returns the base (npulse=1) time and
        amplitude arrays. Negates the first phase for cathodic-first.
        """
        t = np.array(
            [
                0,
                0,
                0 + self.t_pw * self.N_phase,
                0 + self.t_pw * self.N_phase,  # interphase gap start
                0 + self.t_pw * self.N_phase + self.t_interphase,  # ipg end
                0 + self.t_pw * self.N_phase + self.t_interphase,
                0 + self.t_pw * (self.N_phase + 1) + self.t_interphase,
                0 + self.t_pw * (self.N_phase + 1) + self.t_interphase,
            ]
        )
        amp = np.array(
            [
                0,
                -self.AMP / self.N_phase,
                -self.AMP / self.N_phase,
                0,  # inter-phase gap start
                0,  # inter-phase gap end
                self.AMP,
                self.AMP,
                0,
            ]
        )

        amp /= 1000  # convert NEURON's mA to uA

        self.t_base, self.amp_base = (t, amp)

        return (t, amp)

    def sin_stim_wave(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates a sinusoidal waveform and returns the base (npulse=1) time and
        amplitude arrays.
        """
        # n_step per phase, extra point (+1) is to pad a zero
        t = np.arange(0, 2 * self.n_step + 1)

        # Negate wave (cathodic-first), /1000 to convert NEURON's mA to uA
        amp = -self.AMP / 1000 * np.sin(2 * np.pi * t / (2 * self.n_step))
        amp *= np.sqrt(2)  # so that the RMS value is the equivalent square val
        amp = np.repeat(amp, 2)

        # offset to create interpolated step profile
        amp[1:-1:] = amp[2::]
        amp[-1] = 0

        # Convert n_step index into time
        # conversion: * PERIOD (2 * pulse width) / n (2 * n_step)
        t = np.repeat(t, 2) * 2 * self.t_pw / (2 * self.n_step)

        self.t_base, self.amp_base = (t, amp)

        return (t, amp)

    def exp_stim_wave(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates an exponential waveform and returns the base (npulse=1) time
        and amplitude arrays.
        """
        # n_step per phase, extra point (+1) is to pad a zero
        t = np.arange(0, 2 * self.n_step + 1, dtype=float)

        # first phase is decaying exponential
        amp = np.exp(-1 * t / (self.t_tau * (self.n_step)))

        # second phase is a negated (wrt phase 1) square pulse
        amp[-(self.n_step + 1) : :] = -1

        # repeat to allow for interpolation; scale by amplitude, AMP
        amp = np.repeat(amp, 2)

        # shift to create interpolation/step profile
        amp[2::] = amp[1:-1:]

        # first/final entry are padded zeros
        amp[0], amp[-1] = 0, 0

        # Note that there must be equivalent charge deposited in each phase.
        # Thus, we integrate (in our case, sum) the square-pulse phase (known
        # amplitude) and note that this must equal the sum of theexponential
        # phase. We scale the exponential phase such that this is satisifed.
        ano_start = np.squeeze(np.where(amp == -1))[0]
        Q_ano = abs(np.sum(amp[ano_start:]))
        Q_cat = abs(np.sum(amp[:ano_start]))
        amp[:ano_start] *= Q_ano / Q_cat

        # -1 to produce cathodic-first (by conventing of AMP being positive)
        # /1000 to convert from NEURON's mA to uA
        amp *= -1 * self.AMP / 1000.00

        # Convert n_step index into time // repeat allows for interpolation
        t = np.repeat(t, 2) * self.t_pw / self.n_step

        self.t_base, self.amp_base = (t, amp)
        return (t, amp)

    def triangle_stim_wave(self) -> tuple:
        """
        Creates a triangular waveform and returns the base (npulse=1) time and
        amplitude arrays.
        """
        t = np.array(
            [
                0,
                0,
                0 + self.t_pw * self.N_phase / 2,  # triangular peak
                0 + self.t_pw * self.N_phase,  # end of triangle phase
                0 + self.t_pw * self.N_phase,  # interphase gap
                0 + self.t_pw * self.N_phase,
                0 + self.t_pw * (self.N_phase + 1),
                0 + self.t_pw * (self.N_phase + 1),
            ]
        )
        amp = np.array(
            [
                0,
                0,
                -self.AMP * np.sqrt(3),
                0,  # end of triangle phase
                0,  # inter-phase gap
                self.AMP,
                self.AMP,
                0,
            ]
        )

        amp /= 1000  # convert NEURON's mA to uA

        self.t_base, self.amp_base = (t, amp)

        return (t, amp)

    def stim_base(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns the base (npulse=1, t_pre=0) time and amplitude arrays as a
        tuple.
        """
        if self.stim_type == BIPHASIC or self.stim_type == ANODIC:
            return self.biphasic_stim_wave()
        elif self.stim_type == SIN:
            return self.sin_stim_wave()
        elif self.stim_type == EXP:
            return self.exp_stim_wave()
        elif self.stim_type == TRIANGLE:
            return self.triangle_stim_wave()
        else:
            assert False, f"ERROR: Unknown stim. type ({self.stim_type}) used."

    def stim_proc(self, fOut: str) -> None:
        """Calls the appropriate .hoc stim procedure based on stimulus type."""

        t, amp = self.stim_base()

        # Start with (0,0) for consistency
        amp_wave = np.array([0.0])
        t_wave = np.array([0.0])

        t_cur = self.t_pre  # start time for first pulse

        for _ in range(self.n_pulse):
            # add each pulse's amplitude vector
            amp_wave = np.append(amp_wave, amp)
            # move time along by t_cur, the time at which the last pulse ended
            t_wave = np.append(t_wave, t + t_cur)
            t_cur = t_wave[-1] + self.t_interpulse

        # Save/embed this data (amplitude vs. time)
        self.amplitude = amp_wave
        self.time = t_wave

        # Round for NEURON
        amp_wave = np.round(self.amplitude, 8)
        t_wave = np.round(self.time, 8)

        # proc header and comments
        fOut.write("proc stim_waveform0() {\n")
        fOut.write(f"    // Stim waveform procedure for {self.stim_type} wave.\n\n")
        for item in vars(self).items():  # stim. parameters
            if np.size(item[1]) > 2:
                continue
            fOut.write(f"    // {item[0]}: {item[1]}\n")  # parameter : value
        fOut.write("\n")

        # Generic proc filling using amp_wave/t_wave #

        # Resize vectors
        L_resize = np.size(amp_wave) + 1
        fOut.write("\n    // Resize vectors\n")
        fOut.write(
            f"    stim_amp.resize({L_resize})  // +1 in length to avoid vm hyp issue\n"
        )
        fOut.write("    stim_amp.fill(0)\n")
        fOut.write(f"    stim_time.resize({L_resize})\n")
        fOut.write("    stim_time.fill(0)\n")
        fOut.write(
            f"    stim_time.x[{np.size(amp_wave)}] = {(t_wave[-1]+self.t_pw):.3f} // pad end\n"
        )

        # Write out proc vectors
        self.proc_count = 1
        fOut.write("\n    // stim values\n")

        i = 0  # proc length counter

        # Amplitude values
        for idx, val in enumerate(amp_wave):
            if i and i % 25 == 0:  # limit proc length to 50 lines
                fOut.write("}\n\n")
                fOut.write(f"proc stim_waveform{self.proc_count}() {{\n")
                self.proc_count += 1

            fOut.write(f"    stim_amp.x[{idx}]  = {val}\n")
            i += 1

        # Time values
        for idx, val in enumerate(t_wave):
            if i and i % 25 == 0:  # limit proc length to 50 lines
                fOut.write("}\n\n")
                fOut.write(f"proc stim_waveform{self.proc_count}() {{\n")
                self.proc_count += 1

            fOut.write(f"    stim_time.x[{idx}] = {val}\n")
            i += 1

        t_idx = i - np.size(amp_wave) + 1

        while t_idx < L_resize:
            fOut.write(
                f"    stim_time.x[{t_idx}] = {(t_wave[-1]+self.t_pw):.3f} // pad end\n"
            )
            t_idx += 1

        # Wrapper
        fOut.write("}\n\n")

        # Overarching procedure - calls decomposed procedures
        fOut.write("proc stim_waveform() {\n")
        fOut.write("    del = $1\n")
        for i in range(0, self.proc_count):
            fOut.write(f"    stim_waveform{i}(del)\n")
        fOut.write("}\n\n")

    def stim_plot(self, pic_path: str) -> None:
        """Plots the stimulus waveform based on stimulus type."""

        t, amp = self.stim_base()
        plt.figure(123)
        plt.plot(
            t + self.t_pre,
            amp * 1000.00,
            c="r",
            marker="x",
            ms=20,
        )
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (uA)")
        plt.axhline(y=0, color="k", ls="--", alpha=0.5)

        # Add RMS value if sinusoidal plot
        if self.stim_type == SIN:
            plt.axhline(y=self.AMP, color="b", ls="--", alpha=0.5)

        cur_ax = plt.gca()
        curr_lims = cur_ax.get_ylim()
        ax3 = cur_ax.twinx()  # twinx for second y-axis
        ax3.set_ylabel("Charge density (diameter=10um) ($mC/cm^2$)", labelpad=15)
        ax3.set_ylim(curr_lims)

        formatter = mticker.FuncFormatter(  # apply a function formatter
            lambda x, pos: "{:.2f}".format(
                stim_calculations.calc_charge_density(amp=x, t_pw=self.t_pw, r_elec=5)
            )
        )
        ax3.yaxis.set_major_formatter(formatter)

        float_formatter = mticker.FuncFormatter(lambda x, pos: "{:.2f}".format(x))
        cur_ax.xaxis.set_major_formatter(float_formatter)

        cur_ax.locator_params(axis="both", nbins=4)
        ax3.locator_params(axis="both", nbins=4)
        ax3.tick_params(axis="y", length=20, width=5, direction="in", color="k")

        plt.title(
            f"1 pulse of stim. waveform (npulse={self.n_pulse}, ipi={self.t_interpulse} ms)",
            pad=60,
            fontsize=36,
        )
        plt.tight_layout()

        if pic_path:
            create_dir(pic_path + pic_folder)
            plt.savefig(pic_path + pic_folder + "/stim_waveform.jpg")

        plt.close()


############################ WAVEFORM AGENTS ##################################


class Agent:
    """
    For RGC-fovea populations, each Neuron/cell is represented by an Agent.
    """

    OUT_PREFIX_HOC = "waveform"  # TPS waveform hoc file name prefix
    OUT_PREFIX_RSP = "resp"  # NEURON output txt file name prefix
    PICKLED_DIR = "pickled"

    N_phase = 1

    def __init__(self, stim_type):
        """Creates an Agent object."""
        # Stimulus class
        self.stimulus = Stimulus(stim_type=stim_type)

    def save(self, sub_dir: str, i: int) -> None:
        """
        Saves an Agent to file located in directory tree `sub_dir/PICKLED_DIR`
        using identifier no. `i` and pickle.
        """

        # build subdir structure if needed
        path = f"{os.getcwd()}/{sub_dir}/{self.PICKLED_DIR}"
        create_dir(path, get_cwd=False)

        # save contents
        file_name = f"{path}/agent{i:04d}.pkl"
        with open(file_name, "wb") as output:
            pickle.dump(self, output, pickle.DEFAULT_PROTOCOL)

    @staticmethod
    def load(sub_dir: str, i: int) -> "Agent":
        """
        Loads an Agent from file located in directory tree `sub_dir` with
        identifier no. `i`. Returns the Agent object.
        """
        path = f"{os.getcwd()}/{sub_dir}/{Agent.PICKLED_DIR}"
        file_name = f"{path}/agent{i:04d}.pkl"

        with open(file_name, "rb") as file_in:
            try:
                agent = pickle.load(file_in)
                return agent
            except (Exception, EOFError) as e:
                print(f"ERROR with {file_name}.\n{e}")
                if i == 0:
                    raise e
                return None

    def write_stim_waveform(self, fOut: str) -> None:
        """
        Calls the Stimulus object to write the .hoc code for stimulation
        waveform procedure as per the many stim. parameters.
        - `fOut` is the output file path.
        """
        self.stimulus.stim_proc(fOut=fOut)


########################## WAVEFORM POPULATION ################################


class Generation:
    """
    Class representing a generation (population) of Agent objects.
    An Agent is assigned to each cell and/or stim. waveform.
    """

    GEN_SIZE = 64  # Agents (cells) in each generation (population)
    T_TOTAL = 130  # Total simulation time [ms]
    HEX = False  # Hex or monopolar electrode toggle
    STIM_XYZ = [0, 0, -80]

    T_BIN = 2.5  # Time used to bin events for plots [ms] (~=refractory period)

    def __init__(
        self,
        trial_id: int,
        run_id: int,
        gen_id: int,
        stim_type: str = BIPHASIC,
    ):
        """
        Creates a generation with:
            - `run_id' (run number)
            - `gen_id` (generation number)
            - `trial_id` is used to inform loading paths and for complete
                retraceability.
            - `stim_type` dictates the waveform type and should be either SIN,
                EXP, ANODIC or BIPHASIC. The appropriate Agent and Stimulus
                objects are made to this specification. If None, Generation is
                initialised empty (no Agents)
        """

        self.trial_id = trial_id
        self.run_id = run_id
        self.gen_id = gen_id

        self.agents = np.array([])
        self.gen_path = f"{results_path}/run{run_id:03d}/gen{gen_id:04d}/"

        if stim_type:
            for i in range(0, self.GEN_SIZE):
                self.agents = np.append(self.agents, Agent(stim_type=stim_type))

    def save_agents(self) -> None:
        """Saves (and pickles) this population's agents."""

        for idx, agent in enumerate(self.agents):
            agent.save(f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}", idx)

    @staticmethod
    def load(run_id: int, gen_id: int, trial_id: int) -> "Generation":
        """
        Loads the population located at the current working directory, with
        run number `run_id', generation number `gen_id', and trial `trial_id`.
        Static method as it instantiaties the Generation object.
        """
        sim_path = f"../RGC-fovea/results/trial_{trial_id:03d}/sim"
        path = f"{sim_path}/run{run_id:03d}/gen{gen_id:04d}"
        gen = Generation(
            trial_id=trial_id,
            run_id=run_id,
            gen_id=gen_id,
            stim_type=None,
        )
        for i in range(gen.GEN_SIZE):
            agent = Agent.load(path, i)
            if agent is not None:
                gen.agents = np.append(gen.agents, agent)

            else:  # failed to load Agent (corrupt Agent .pkl)
                # copy over Agent 0 but update characteristics with trial's .pkl
                gen.agents = np.append(gen.agents, copy.deepcopy(gen.agents[0]))
                f_open = f"../RGC-fovea/results/trial_{trial_id:03d}/trial_{trial_id:03d}.pkl"

                with open(f_open, "rb") as file_in:
                    population = pickle.load(file_in)

                gen.agents[i].soma_loc = population.neurons[i].soma_loc
                gen.agents[i].tree_loc = population.neurons[i].tree_loc
                gen.agents[i].cell_type = population.neurons[i].cell_type
                gen.agents[i].dendrite_area = population.neurons[i].tree_area

        return gen

    def clear_agents(self) -> None:
        """Resets all agent waveforms and scores to null."""

        for agent in self.agents:
            agent.score = 0
            agent.waveform = np.array([0] * agent.PULSES)

    def write_simultaneous_hoc(self) -> None:
        """
        Writes a .hoc file for simultaneous simulation of an entire population.
        NOTE: This assumes all agents possess the same waveform! This is not
        a valid assumption for all types of investigations.
        """

        path = f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}"
        cell_hoc_path = f"{path.split('/sim')[0]}/cellHoc"
        fOut = open(f"{path}/simulatePopulation.hoc", "w")

        for i in range(self.GEN_SIZE):
            fOut.write(f'{{load_file("{cell_hoc_path}/cell_{i:04d}.hoc")}}\n')

        fOut.write('{load_file("common/global.hoc")}\n\n')
        fOut.write(f"objref rgc[{self.GEN_SIZE}]\n")

        for i in range(self.GEN_SIZE):
            fOut.write(f"rgc[{i}] = new cell_{i:04d}()\n")

        fOut.write("\naccess rgc[0].soma\n")

        fOut.write("// load in extracellular instr and mechanisms \n")
        if self.HEX:
            fOut.write('{load_file("../RGC-fovea/stim/instr-hex.hoc")}\n\n')
        else:
            fOut.write('{load_file("../RGC-fovea/stim/instr.hoc")}\n\n')

        fOut.write(
            f"setelec({self.STIM_XYZ[0]}, {self.STIM_XYZ[1]}, {self.STIM_XYZ[2]})\n\n"
        )

        fOut.write("// update time according to stim waveform\n")
        fOut.write(f"tstop={Generation.T_TOTAL}\n")
        fOut.write("tstop_changed()\n\n")

        # Stim waveform
        self.agents[0].write_stim_waveform(fOut=fOut)
        fOut.write(f"setstim({t_pre})  // updates waveform with 10 ms stim delay\n")

        # Set up membrane potential arrays
        fOut.write("\n// saving vm of soma\n")
        fOut.write(f"objref attDv[{self.GEN_SIZE}]\n")

        fOut.write(f"for c = 0, {self.GEN_SIZE-1} {{\n")
        fOut.write("    attDv[c] = new Vector()\n")
        fOut.write("    attDv[c].record(&rgc[c].soma.v(0.5))\n")
        fOut.write("}\n")

        # Run
        fOut.write("// run\nrun()\n\n")

        # Save voltages
        fOut.write("strdef fName\n")
        fOut.write("objref gFobj\n")
        fOut.write(f"for c = 0, {self.GEN_SIZE-1} {{\n")
        fOut.write(
            f'    sprint(fName, "{path}/{Agent.OUT_PREFIX_RSP}-cell%.4d-soma.txt", c)\n'
        )
        fOut.write("    gFobj = new File()\n")
        fOut.write("    gFobj.wopen(fName)\n")
        fOut.write('    attDv[c].printf(gFobj, "%f\\n")\n')
        fOut.write("    gFobj.close()\n")
        fOut.write("}\n")

        # Finish up
        fOut.write(f'printf("INFO: finished population: gen{self.gen_id:04d}!")\n')
        fOut.write("\n")
        fOut.close()

    def write_master_hoc_files(self, cell_id: int = None, master: bool = False) -> None:
        """
        Writes the master hoc file that spawns in cell no. waveId, defines
        recording/save procedures, and runs the simulation according to the
        generation's common stim. waveform.
            - `cell_id`: cell number/id.
            - `master`: write a single .hoc file that inherits dynamic
                variables based on the cell id using bash's `here`. (TODO)

        NOTE: currently only valid for waveforms which do NOT exhibit temporal
        variance...amplitude and IPI cannot change over time.
        """

        # TODO: `master` requires the NEURON command to include cell ID number
        # and for this number to be sprint'd into the appropriate strings

        path = f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}"
        cell_hoc_path = f"{path.split('/sim')[0]}/cellHoc"

        if master:
            fOut = open(f"{path}/master.hoc", "w")
            fOut.write("strdef fCell\n")
            fOut.write(f'sprint(fCell, "{cell_hoc_path}/cell_%.4d.hoc", waveId)\n')
            fOut.write("{load_file(fCell)}\n")
        else:
            fOut = open(f"{path}/waveform{cell_id:04d}.hoc", "w")
            fOut.write(f'{{load_file("{cell_hoc_path}/cell_{cell_id:04d}.hoc")}}\n')

        fOut.write('{load_file("common/global.hoc")}\n\n')
        fOut.write("objref rgc[1]\n")

        if master:
            pass  # TODO: master needs to use waveId
        else:
            fOut.write(f"rgc[0] = new cell_{cell_id:04d}()\n")
        fOut.write("access rgc[0].soma\n")

        fOut.write("// load in extracellular instr and mechanisms \n")
        if self.HEX:
            fOut.write('{load_file("../RGC-fovea/stim/instr-hex.hoc")}\n\n')
        else:
            fOut.write('{load_file("../RGC-fovea/stim/instr.hoc")}\n\n')

        fOut.write(
            f"setelec({self.STIM_XYZ[0]}, {self.STIM_XYZ[1]}, {self.STIM_XYZ[2]})\n\n"
        )

        fOut.write("// update time according to stim waveform\n")
        fOut.write(f"tstop={Generation.T_TOTAL}\n")
        fOut.write("tstop_changed()\n\n")

        # Stim waveform
        self.agents[cell_id].write_stim_waveform(fOut=fOut)
        fOut.write(
            f"setstim({t_pre})  // updates waveform with {t_pre} ms stim delay\n"
        )

        # Set up membrane potential arrays
        fOut.write("\n// saving vm of neuron - soma, ais, axon\n")
        if cell_id == 0:  # save stim vector
            fOut.write("objref attDv[7]\n")
        else:
            fOut.write("objref attDv[5]\n")
        fOut.write("attDv[0] = new Vector()\n")
        fOut.write("attDv[0].record(&rgc[0].soma.v(0.5))\n")
        fOut.write("attDv[1] = new Vector()\n")
        fOut.write("attDv[1].record(&rgc[0].ais.v(0.5))\n")
        fOut.write("attDv[2] = new Vector()\n")
        fOut.write("attDv[2].record(&rgc[0].axon.v(0.1))\n")
        fOut.write("attDv[3] = new Vector()\n")
        fOut.write("attDv[3].record(&rgc[0].axon.v(0.25))\n")
        fOut.write("attDv[4] = new Vector()\n")
        fOut.write("attDv[4].record(&rgc[0].axon.v(0.5))\n\n")
        if cell_id == 0:
            fOut.write("attDv[5] = new Vector()\n")
            fOut.write("attDv[5] = stim_amp\n\n")
            fOut.write("attDv[6] = new Vector()\n")
            fOut.write("attDv[6] = stim_time\n\n")

        fOut.write("// run\nrun()\n\n")

        fOut.write("// Save all membrane voltages in their appropriate files\n\n")
        fOut.write("objref gFobj\n")
        fOut.write("strdef fName\n")
        fOut.write("gFobj = new File()\n")

        # SOMA
        if master:
            # TODO sprint for fName and open file with such
            pass
        else:
            open_path = f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}/{Agent.OUT_PREFIX_RSP}-cell{cell_id:04d}"
            fOut.write(f'gFobj.wopen("{open_path}-soma.txt")\n')
        fOut.write('attDv[0].printf(gFobj, "%f\\n")\n')
        fOut.write("gFobj.close()\n\n")

        # AIS
        if master:
            # TODO sprint for fName and open file with such
            pass
        else:
            fOut.write(f'gFobj.wopen("{open_path}-ais.txt")\n')
        fOut.write('attDv[1].printf(gFobj, "%f\\n")\n')
        fOut.write("gFobj.close()\n\n")

        # AXON (10%)
        if master:
            # TODO sprint for fName and open file with such
            pass
        else:
            fOut.write(f'gFobj.wopen("{open_path}-axon10.txt")\n')
        fOut.write('attDv[2].printf(gFobj, "%f\\n")\n')
        fOut.write("gFobj.close()\n\n")

        # AXON (25%)
        if master:
            # TODO sprint for fName and open file with such
            pass
        else:
            fOut.write(f'gFobj.wopen("{open_path}-axon25.txt")\n')
        fOut.write('attDv[3].printf(gFobj, "%f\\n")\n')
        fOut.write("gFobj.close()\n\n")

        # AXON (50%)
        if master:
            # TODO sprint for fName and open file with such
            pass
        else:
            fOut.write(f'gFobj.wopen("{open_path}-axon50.txt")\n')
        fOut.write('attDv[4].printf(gFobj, "%f\\n")\n')
        fOut.write("gFobj.close()\n\n")

        if cell_id == 0:
            stim_vector = f"{open_path.split(Agent.OUT_PREFIX_RSP)[0]}stim-amp-vector"
            fOut.write(f'gFobj.wopen("{stim_vector}.txt")\n')
            fOut.write('attDv[5].printf(gFobj, "%f\\n")\n')
            fOut.write("gFobj.close()\n\n")
            stim_vector = f"{open_path.split(Agent.OUT_PREFIX_RSP)[0]}stim-time-vector"
            fOut.write(f'gFobj.wopen("{stim_vector}.txt")\n')
            fOut.write('attDv[6].printf(gFobj, "%f\\n")\n')
            fOut.write("gFobj.close()\n\n")

        fOut.write(
            f'printf("INFO: done trial{self.trial_id:03d}, run{self.run_id:03d}, gen{self.gen_id:04d}, cell{cell_id:04d}")\n'
        )
        fOut.write("\n")
        fOut.close()

    def simulate_waveforms(self) -> None:
        """Start NEURON and simulate the waveform hoc files in parallel with mpi."""

        cmd = f"cd ../RGC && mpirun -n {N_WORKERS} nrniv -mpi "
        hoc = f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}/simulate_waveforms.hoc"
        os.system(cmd + hoc)

    def simulate_parallel_waveforms(
        self,
        cell_id: int,
        write_hoc: bool = False,
        master: bool = False,
    ) -> None:
        """
        Drives parallel simulations using Python (not mpi).
            - `cell_id`: cell number/id.
            - `write_hoc`: write the waveform .hoc files.
            - `master`: write a single .hoc file that inherits dynamic
                variables based on the cell id. (TODO)
        """

        if master:  # TODO: write_master_hoc_files w/ master=True
            if write_hoc:
                self.write_master_hoc_files(cell_id=None, master=True)
            cmd = f"cd ../RGC && {self.gen_path}master.hoc "
            here = f" << here strdef waveId\n waveId = {cell_id}\n here"
            os.system(cmd + here)

        else:
            if write_hoc:
                self.write_master_hoc_files(cell_id=cell_id, master=False)

            wave_hoc = f"{self.gen_path}{Agent.OUT_PREFIX_HOC}{cell_id:04d}"
            cmd = f"cd ../RGC && nrniv {wave_hoc}.hoc"
            os.system(cmd)

    def simulate_simultaneous_waveforms(self, write_hoc: bool = False) -> None:
        """
        One file to rule them all. Using a single .hoc file, simulates an
        entire population. Very slow and unoptimised (compared to independently
        simulating each cell).
            - `write_hoc`: write the file to rule them all or not?
        """

        if write_hoc:
            self.write_simultaneous_hoc()

        cmd = f"cd ../RGC && nrniv ../RGC-fovea/{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}/simulatePopulation.hoc"
        os.system(cmd)

    def neuron_responses_to_csv(self) -> None:
        """
        Iteratively reads the text files that constitute the NEURON response
        vectors, and amalgamates so into .csv files which summarises all cells
        and response vectors.

        Generates .csv files:
            - segment file(s): responses for each cell but one unique file for
                each segment of interest,
            - master file: all segments and cells are summarised.

        """

        # Create consolidation directory
        clean_dir = f"{self.gen_path}/compressed_nrn_responses"
        create_dir(clean_dir)

        out = self.agents[0].OUT_PREFIX_RSP
        master_csv_gz = f"{clean_dir}/{out}-master.csv.gz"
        seg_base_csv = f"{clean_dir}/{out}-"

        # allow for easy access to soma voltage for other functions
        self.soma_csv_gz = f"{clean_dir}/{out}-soma.csv.gz"

        # the second condition is for older runs which weren't compressed
        if os.path.isfile(master_csv_gz) or os.path.isfile(
            master_csv_gz.replace(".gz", "")
        ):
            return

        # regex search response vectors to dynamically acquire segment names
        # groups results as [(resp_prefix), (segment name), (.txt)]
        r = re.compile(f"(?:({out}-cell0000-))(.*)(?=(.txt))")
        segments = [
            r.match(f).group(2) for f in os.listdir(self.gen_path) if r.match(f)
        ]

        # Load first response vector to get length for establishing 2D arrays
        dummy = f"{self.gen_path}{out}-cell0000-{segments[0]}.txt"
        len_data = len(np.loadtxt(dummy))

        # data structures to convert into pandas dataframes after looping
        n_col = len(self.agents)
        n_row = len_data
        n_segments = len(segments)

        # master has n_col = n_cells * n_segments
        master_list = np.ones((n_row, n_col * n_segments), dtype=float) * -70
        seg_lists = {}
        for s in segments:
            seg_lists[s] = np.empty((n_row, n_col), dtype=float)

        hdr_master = [""] * n_col * n_segments
        hdr_seg = np.arange(n_col)

        for c in range(n_col):  # loop over each cell
            cell = f"{self.gen_path}{out}-cell{c:04d}-"
            for i, s in enumerate(segments):  # loop over each segment
                response = np.loadtxt(f"{cell}{s}.txt")
                master_list[:, c * n_segments + i] = response
                hdr_master[c * n_segments + i] = f"{s} ({c * n_segments + i:04d})"
                seg_lists[s][:, c] = response

        # Write to .csv
        pd.DataFrame(master_list).to_csv(
            master_csv_gz,
            mode="w",
            compression="gzip",
            header=hdr_master,
            sep=",",
            float_format="%.3f",
        )

        for s in segments:
            pd.DataFrame(seg_lists[s]).to_csv(
                f"{seg_base_csv}{s}.csv.gz",
                mode="w",
                compression="gzip",
                header=hdr_seg,
                sep=",",
                float_format="%.3f",
            )

    def delete_neuron_responses(self) -> None:
        """Remove all individual NEURON response vectors."""

        # Make sure NEURON responses have been amalgamated into .csv files
        self.neuron_responses_to_csv()

        to_del = f"{self.agents[0].OUT_PREFIX_RSP}-cell*.txt"
        for p in Path(self.gen_path).glob(to_del):
            p.unlink()

        print(f"INFO: Deleted all NEURON response .txt files.")

    def get_pickled_cells(self) -> None:
        """
        Saves soma location, dendritic location/area, and cell types as per the
        .csv file.
        """
        f_open = f"../RGC-fovea/results/trial_{self.trial_id:03d}/trial_{self.trial_id:03d}.pkl"

        with open(f_open, "rb") as file_in:
            population = pickle.load(file_in)

        for i, nrn in enumerate(population.neurons):
            self.agents[i].soma_loc = nrn.soma_loc
            self.agents[i].tree_loc = nrn.tree_loc
            self.agents[i].cell_type = nrn.cell_type
            self.agents[i].dendrite_area = nrn.tree_area

    def get_csv_cells(self) -> None:
        """
        Saves soma location, dendritic location/area, and cell types as per the
        .csv file.
        """
        df_pop = pd.read_csv(
            f"{results_path}/../trial{self.trial_id:03d}-population.csv"
        )
        soma_loc = df_pop[["x (um)", "y (um)", "z (um)"]].to_numpy()
        tree_loc = df_pop[["x-tree (um)", "y-tree (um)", "z-tree (um)"]].to_numpy()
        cell_type = df_pop["Cell type"].to_numpy(dtype=str)
        if "Dendritic area (um^2)" in df_pop.head():
            dendrite_area = df_pop["Dendritic area (um^2)"].to_numpy()
        else:
            dendrite_area = np.array([np.nan] * len(cell_type))

        for i in range(self.GEN_SIZE):
            self.agents[i].soma_loc = soma_loc[i]
            self.agents[i].tree_loc = tree_loc[i]
            self.agents[i].cell_type = cell_type[i]
            self.agents[i].dendrite_area = dendrite_area[i]

    def get_cell_locations_and_type(self, pkl: bool = False) -> None:
        """
        Adds soma location (`soma_loc`), dendritic tree location (`tree_loc`)
        and cell type (`cell_type`) to Generation's Agent objects.
        Also saves so as population lists in this Generation object...
        ...(`pop_soma_loc`, `pop_tree_loc`, `pop_cell_type`).

            - `pkl`: if True, use pickle load to determine properties, else,
                load the trial's csv and determine from so.

        """
        if hasattr(self, "pop_soma_loc"):  # prevent overwrite
            return

        self.get_pickled_cells() if pkl else self.get_csv_cells()

        self.pop_soma_loc = np.empty((self.GEN_SIZE, 3), dtype=float)
        self.pop_tree_loc = np.empty((self.GEN_SIZE, 3), dtype=float)
        self.pop_cell_types = np.empty(self.GEN_SIZE, dtype="<U3")

        for i, nrn in enumerate(self.agents):
            self.pop_soma_loc[i] = nrn.soma_loc
            self.pop_tree_loc[i] = nrn.tree_loc
            self.pop_cell_types[i] = nrn.cell_type

    def get_experiment_characteristics(self) -> None:
        """
        Parses the trial and generation .txt files to determine characteristics
        such as population size and electrode radius.
        Uses the patch size to enforce axes limits for plots.
        """

        if hasattr(self, "patch_size"):
            return

        self.amp = abs(self.agents[0].stimulus.AMP)
        self.stim_type = self.agents[0].stimulus.stim_type
        if self.stim_type == ANODIC:
            self.stim_type += f" ({self.agents[0].stimulus.N_phase}:1 PW)"

        # Determine trial's population characteristics
        self.ecc = None
        self.patch_size = None
        self.n_gcl = None
        f_open = f"../RGC-fovea/results/trial_{self.trial_id:03d}/trial_{self.trial_id:03d}.txt"
        f = open(f_open)

        for line in f:
            if "Eccentricity" in line:
                ecc = line.split("Eccentricity of ")[-1].split(" ")[0]
                self.ecc = float(ecc)
            elif "Patch size" in line:
                self.patch_size = float(line.split("size of ")[-1].split(" mm")[0])
            elif "GCLs" in line:
                self.n_gcl = int(line.split("no. of GCLs is ")[-1][0])
                break
        f.close()

        # Determine stimulation electrode parameters
        self.r_elec = None
        self.HEX = False
        self.pitch = 30
        self.t_pw = 0.1
        f_open = f"{self.gen_path}/stimulation-info.txt"
        f = open(f_open)

        for line in f:
            if "Hex electrode" in line:
                self.HEX = True
            elif "t_pw" in line:
                self.t_pw = float(line.split("t_pw is ")[-1].split(" ")[0])
            elif "elecRad" in line:
                self.r_elec = float(line.split("= ")[-1].split(" ")[0])
            elif "stim XYZ" in line:
                self.STIM_XYZ[0] = float(line.split("[")[-1].split(",")[0])
                self.STIM_XYZ[1] = float(line.split(", ")[1].split(",")[0])
                self.STIM_XYZ[2] = float(line.split(", ")[-1].split("]")[0])
            elif "hexDisp" in line:
                self.pitch = float(line.split("= ")[-1].split(" ")[0])
                break
        f.close()

        # Checks
        assert (
            self.ecc is not None
        ), f"Eccentricity was not found in the trial file ({f_open})."

        assert (
            self.patch_size is not None
        ), f"Patch size was not found in the trial file ({f_open})."

        assert (
            self.n_gcl is not None
        ), f"Number of GCLs was not found in the trial file ({f_open})."
        assert (
            self.r_elec is not None
        ), f"Electrode radius was not found in the trial file ({f_open})."

        self.charge_dens = stim_calculations.calc_charge_density(
            self.amp, self.t_pw, self.r_elec
        )

        # Axes limits
        self.patch_upper = 1.1 * self.patch_size / 2 * 1e3  # 1e3 mm to um
        self.patch_lower = -self.patch_upper

        self.patch_right = self.patch_upper
        self.patch_left = -self.patch_right
        self.patch_top = self.patch_right
        self.patch_bot = -self.patch_top

        # Determine no. of bins based on length of PATCH and SOMA_D
        self.n_bin = int(np.ceil(self.patch_size * 1e3 / SOMA_BIN))  # 1e3 mm -> um
        if self.n_bin % 2 != 1:  # enforce odd L to make the centre the centre of a bin
            self.n_bin += 1

    def get_filtered_responses(
        self,
        dt: float = dt_nrn * 1e3,
        fc: float = 3500,
    ) -> np.ndarray:
        """
        Saves the filtered membrane potential of each Agent/cell's soma
        response vector. Embedded into Agent objects.

            - `dt`: the time-interval used for simulations (ms).
            - `fc`: the low-pass filter cut-off frequency (Hz).

        Returns a N-dimensional numpy array of the filtered membrane potentials
        where N is the no. of agents (`GEN_SIZE`).
        """

        if hasattr(self, "cell_voltages"):
            return self.cell_voltages

        self.neuron_responses_to_csv()  # need to extract from the .csv files

        fs = 1 / (dt * 1e-3)  # neuron sample frequency

        # 2D-voltage array of cells (rows) and time-bins (columns)
        # set the first entry up to establish sizes and structures
        first_response = np.squeeze(
            pd.read_csv(
                self.soma_csv_gz,
                compression="gzip",
                index_col=0,
                usecols=[0, 1],
            ).to_numpy()
        )
        self.agents[0].dataF = np.reshape(
            butter_lowpass(first_response, fs, fc), (1, -1)
        )
        voltages = copy.deepcopy(self.agents[0].dataF)

        for c in range(1, self.GEN_SIZE):  # from 1 to avoid re-doing 1st cell
            # load
            response = np.squeeze(
                pd.read_csv(
                    self.soma_csv_gz,
                    compression="gzip",
                    index_col=0,
                    usecols=[0, c + 1],
                ).to_numpy()
            )

            # Skips filtering of inactive cells to save time
            if np.sum(response > TRIG_AMP) == 0:  # no spikes
                self.agents[c].dataF = response
            elif hasattr(self.agents[c], "dataF"):
                pass
            else:  # filter
                self.agents[c].dataF = np.reshape(
                    butter_lowpass(response, fs, fc), (1, -1)
                )
            voltages = np.concatenate((voltages, self.agents[c].dataF))

        self.cell_voltages = copy.deepcopy(voltages)

        return voltages

    def parallelised_spike_detection(self, cell_id: int) -> None:
        """Parallelised function for speeding up spike detection."""

        # Progress printer
        if cell_id % 500 == 0:
            print(f"INFO: Running spike detection for cell {cell_id:04d}...")

        if not hasattr(self.agents[cell_id], "spikeV"):

            # usecols for cell_id+1 as column 1 is cell 0 (column 0 is index column)
            response = np.squeeze(
                pd.read_csv(
                    self.soma_csv_gz,
                    compression="gzip",
                    index_col=0,
                    usecols=[0, cell_id + 1],
                ).to_numpy()
            )
            (spikeV, spikeT, _) = spike_detection.run(data=response)

            self.agents[cell_id].spikeV = spikeV
            self.agents[cell_id].spikeT = spikeT

        n_spikes = np.size(self.agents[cell_id].spikeT)

        if n_spikes:
            self.spike_weights[cell_id] = n_spikes
            self.activated_area[cell_id] = self.agents[cell_id].dendrite_area

    def add_to_population_spikes_csv(self) -> None:
        """
        Adds a spike count to each cell for this generation's response.
        This is saved to a .csv file denoted 'population-spikes.csv'. This file
        is created using 'trialXXX-population.csv' as a template for the cell
        positions, if need be.
        """
        # Use existing population-spikes.csv if present
        stim_csv = f"{results_path}/run{self.run_id:03d}/population-spikes.csv"
        if os.path.isfile(stim_csv):  # population-spikes.csv exists -> use so
            csv_name = stim_csv
            run_df = pd.read_csv(csv_name)
        else:  # does not exist -> use trialXXX-population.csv as template
            csv_name = f"{results_path.split('/sim')[0]}/trial{self.trial_id:03d}-population.csv"
            run_df = pd.read_csv(csv_name)

        run_df[f"Spikes (config. {self.gen_id})"] = self.spike_weights
        run_df.to_csv(stim_csv, index=False)

    def population_spike_detection(self) -> None:
        """
        Performs spike detection across the whole population, embedding
        windowed voltages (spikeV) and event times (spikeT).
        """

        self.neuron_responses_to_csv()  # need to extract from the .csv files

        # Spike counters
        self.spikes_on = 0
        self.spikes_off = 0
        self.spikes_on_unique = 0
        self.spikes_off_unique = 0
        self.spike_weights = np.zeros(self.GEN_SIZE, dtype=int)  # spikes/cell
        self.activated_area = np.zeros(self.GEN_SIZE, dtype=float)

        # Parallel spike detection #

        # prevent overwrite and allows for this to be called, even after
        # the NEURON response vectors have been cleaned-up/deleted
        Parallel(n_jobs=N_WORKERS, require="sharedmem")(
            delayed(self.parallelised_spike_detection)(
                cell_id=c,
            )
            for c in range(0, self.GEN_SIZE)
        )

        cell_types = np.squeeze(np.unique(self.pop_cell_types))

        for c in cell_types:  # ["ON", "OFF"]
            # Filter dataset by cell type
            filtered_idx = np.where(self.pop_cell_types == c)
            filtered_spikes = self.spike_weights[filtered_idx]
            n_spikes = np.sum(filtered_spikes)
            n_unique = np.count_nonzero(filtered_spikes)

            if c == "ON":
                self.spikes_on += n_spikes
                self.spikes_on_unique += n_unique

            else:  # OFF
                self.spikes_off += n_spikes
                self.spikes_off_unique += n_unique

        # Total activation (no. of spikes)
        self.total_spikes = self.spikes_on + self.spikes_off
        self.total_spikes_unique = self.spikes_on_unique + self.spikes_off_unique

        # Record this spike vector in the trial's overall .csv file which shows
        # spike counts for each configuration/generation.
        self.add_to_population_spikes_csv()

    def count_spikes_in_parallel(self, cell_id: int) -> None:
        """
        Counts the spikes of the corresponding cell and assigns it to the
        Population (self variable).
        """
        n_spikes = np.size(self.agents[cell_id].spikeT)
        self.spike_weights[cell_id] = n_spikes

        if n_spikes:

            self.activated_area[cell_id] = self.agents[cell_id].dendrite_area

            if self.agents[cell_id].cell_type == "ON":
                self.spikes_on += n_spikes
                self.spikes_on_unique += 1

            else:  # OFF
                self.spikes_off += n_spikes
                self.spikes_off_unique += 1

    def count_spikes(self) -> None:
        """
        Iterates over the population of agents/cells to tally the number of
        spikes, as well as the number of unique spikes.
        type is also considered and used to save counts of so.
        Cell positions are also saved as a class property, and also are
        saved as a second property which is weighted by spike count.
        """

        if hasattr(self, "spike_weights"):
            return

        self.population_spike_detection()  # need to count spikeT arrays

        # Spike counters
        self.spikes_on = 0
        self.spikes_off = 0
        self.spikes_on_unique = 0
        self.spikes_off_unique = 0
        self.spike_weights = np.empty(self.GEN_SIZE, dtype=int)  # spikes/cell
        self.activated_area = np.zeros(self.GEN_SIZE, dtype=float)

        Parallel(n_jobs=N_WORKERS, require="sharedmem")(
            delayed(self.count_spikes_in_parallel)(
                cell_id=c,
            )
            for c in range(0, self.GEN_SIZE)
        )

        # Total activation (no. of spikes)
        self.total_spikes = self.spikes_on + self.spikes_off
        self.total_spikes_unique = self.spikes_on_unique + self.spikes_off_unique

        # Record this spike vector in the trial's overall .csv file which shows
        # spike counts for each configuration/generation.
        self.add_to_population_spikes_csv()

    def plot_contours(
        self,
        cell_dataset: Dict[str, np.ndarray],
        label: str,
        pop_grid: bool = True,
    ) -> None:
        """
        Produce a contour plot of cell activation.
            - `cell_dataset`: dictionary with position arrays tagged by keys
                "x", "y", "z", with a "Weight" array (spike counts or some
                other metric), and cell type tagged "Cell type".
            - `label`: name to save plot under.
            - `pop_grid`: include an outline of the population mosaic behind
                the contour plot.
        """

        # create the new map
        cmap = plt.cm.get_cmap("inferno")
        levels = [0.1, 0.4, 0.7, 1]
        cmaplist = [cmap(x) for x in levels]
        cmaplist[-2] = cmap(0.95)
        cmaplist[-1] = cmap(0.95)
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, len(levels)
        )
        cmap.set_under("white")

        joint_kw = {
            "weights": cell_dataset["Weight"],
            "colors": cmaplist,
            "levels": levels,
            "cmap": None,
        }
        marg_kw = {"lw": 6, "weights": cell_dataset["Weight"]}

        legend_labels = ["Response", "Electrode"]
        marg_kw["color"] = cmap(0.25)
        label = f"population-{label}"

        _ = plt.figure(123)

        try:
            g = sns.jointplot(
                data=cell_dataset,
                x="x",
                y="y",
                kind="kde",
                fill=True,
                joint_kws=joint_kw,
                marginal_kws=marg_kw,
            )

        except (ValueError, IndexError, np.linalg.LinAlgError) as e:
            print(
                f"\nINFO: run {self.run_id:04d} gen{self.gen_id:04d} - insufficient data for KDE contour plot!\n"
                f"ERROR: {e}\n"
            )
            # Remove any existing spike plots to avoid confusion
            # (though diff. stim. parameters should typically be under a new
            # run or gen id)
            silent_remove(
                f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}/{pic_folder}"
                "/contours-{label}.jpg"
            )
            return

        elec_patches = get_patches(
            pos=self.STIM_XYZ,
            radius=self.r_elec,  # radius (float)
            hex_polar=self.HEX,
            hex_pitch=self.pitch,
            facecolor="lime",
        )

        if pop_grid:
            g.ax_joint.hexbin(
                x=cell_dataset["x"],
                y=cell_dataset["y"],
                C=cell_dataset["Weight"],
                reduce_C_function=np.sum,
                gridsize=self.n_bin,
                extent=(
                    self.patch_lower,
                    self.patch_upper,
                    self.patch_lower,
                    self.patch_upper,
                ),
                cmap=cmap,
                edgecolors="white",
                linewidth=2.5,
                alpha=0.5,
                zorder=-1,
            )

        g.fig.set_size_inches(mpl.rcParams["figure.figsize"])  # Match mpl fig size
        g.ax_joint.locator_params(axis="both", nbins=NBINS)  # Major tick labels

        # Marginal tick labels
        g.ax_marg_y.tick_params(labelbottom=True)
        plt.setp(g.ax_marg_y.xaxis.get_majorticklabels(), rotation=-90)
        g.ax_marg_y.xaxis.set_major_locator(MaxNLocator(1))
        g.ax_marg_x.tick_params(labelleft=True)
        g.ax_marg_x.yaxis.set_major_locator(MaxNLocator(1))
        g.set_axis_labels("x $(\mu m)$", "y $(\mu m)$")

        g.ax_joint.set_xlim(self.patch_left, self.patch_right)
        g.ax_joint.set_ylim(self.patch_bot, self.patch_top)
        g.ax_marg_x.set_xlim(self.patch_left, self.patch_right)
        g.ax_marg_y.set_ylim(self.patch_bot, self.patch_top)

        # Get legend handles
        handles = []

        pop_patch = Patch(color=cmap(0.8))
        handles.append(pop_patch)

        # Add circle to legend
        for patch in elec_patches:
            g.ax_joint.add_patch(patch)

        # Legend
        g.ax_joint.legend(
            title=label.split("-")[-1],
            title_fontsize=36,
            handles=handles,
            labels=legend_labels,
            fontsize=32,
            loc="upper right",
            bbox_to_anchor=(1.285, 1.22),
            ncol=1,
        )

        g.ax_joint.set_title(
            f"ecc={self.ecc:.2f} mm, {self.amp:.2f} uA, {self.charge_dens:.3f} mC/cm$^2$",
            pad=30,
            fontsize=36,
            x=0.1,
            y=-0.275,
        )

        ## COLOUR BAR ##
        # make new ax object for the cbar
        plt.subplots_adjust(bottom=0.325)
        cbar_ax = g.fig.add_axes([0.20, 0.1, 0.5, 0.05])

        values = levels
        cmap_bar = cmap
        norm = mpl.colors.BoundaryNorm(values, cmap_bar.N)
        sm = mpl.cm.ScalarMappable(cmap=cmap_bar, norm=norm)
        sm.set_clim(vmin=0.10, vmax=1.0)

        cbar = plt.colorbar(
            sm,
            cax=cbar_ax,
            ticks=values,
            orientation="horizontal",
            extend="min",
            spacing="proportional",
        )
        cbar.set_ticks(values)
        cbar.ax.set_title(
            "Density contours",
            pad=40,
            fontsize=56,
        )
        plt.tight_layout()

        g.savefig(
            f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}/{pic_folder}"
            f"/contours-{label}.jpg",
        )

        plt.close()

    def plot_xz_activation(self) -> None:
        """
        Plots the x-z profile in a 3D plot with slight elevation to visualise
        the activation profile in three-dimensional space.
        """
        # Pre-processing
        self.get_experiment_characteristics()  # need patch size + elec radius
        self.count_spikes()  # need spike count arrays
        self.get_cell_locations_and_type()  # need cell positions + types

        x, y, z = getattr(self, "pop_soma_loc", self.pop_soma_loc).T
        x += self.ecc * 1000  # shift by populations eccentricity
        spikes = self.spike_weights

        # Set up plot
        fig = plt.figure(1, figsize=(40, 30))
        label_size = 65
        label_pad = 155
        ax_upper = fig.add_subplot(projection="3d")
        ax_upper.set_xlabel("x ($\mu m$)", labelpad=95, fontsize=label_size)
        ax_upper.set_ylabel("", labelpad=label_pad)
        ax_upper.set_zlabel("z ($\mu m$)", labelpad=label_pad, fontsize=label_size)

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
        ax_upper.tick_params(
            axis="x", labeltop=False, labelleft=False, labelright=False
        )
        ax_upper.tick_params(
            axis="z", labeltop=False, labelright=False, labelbottom=False
        )

        # Get rid of colored axes planes
        # First remove fill
        ax_upper.xaxis.pane.fill = False
        ax_upper.yaxis.pane.fill = False
        ax_upper.zaxis.pane.fill = False
        # Now set color to white (or whatever is "invisible")
        ax_upper.xaxis.pane.set_edgecolor("w")
        ax_upper.yaxis.pane.set_edgecolor("w")
        ax_upper.zaxis.pane.set_edgecolor("w")

        plt.tight_layout()

        for spine in ax_upper.spines.values():
            spine.set_visible(False)

        soma_size = 450  # marker size (not to scale, visual aid only)
        labelled_active = False
        labelled_inactive = False

        # Find x-limits to keep size of plot reasonable
        buffer = 50
        mask = np.where(spikes > 0)

        if not np.size(mask):  # no activation, no need to plot
            return

        elec_pos = (self.STIM_XYZ[0] + self.ecc * 1000, self.STIM_XYZ[1])

        x_left = np.amin(x[mask]) - buffer
        x_right = max(np.amax(x[mask]), elec_pos[0]) + buffer
        y_left = np.amin(y[mask]) - buffer
        y_right = max(np.amax(y[mask]), elec_pos[1]) + buffer

        for idx in range(self.GEN_SIZE):

            if (
                x[idx] > x_right
                or x[idx] < x_left
                or y[idx] > y_right
                or y[idx] < y_left
            ):
                continue

            label = None
            if not labelled_active and spikes[idx]:
                labelled_active = True
                label = "Activated"
            if not labelled_inactive and not spikes[idx]:
                labelled_inactive = True
                label = "Inactivated"

            cur_clr = cmap(0.85) if spikes[idx] else cmap(0.30)

            alpha_up = 0.80 if not spikes[idx] else 0.95 if spikes[idx] > 10 else 1.00

            ax_upper.scatter(
                x[idx],
                y[idx],
                z[idx],
                facecolors=cur_clr,
                edgecolors="black",
                linewidth=3,
                s=soma_size,
                marker="o",
                depthshade=False,
                alpha=alpha_up,
                label=label,
            )

        q_dens = np.round(self.charge_dens, 4)
        ax_upper.set_title(
            f"{q_dens:.3f} $mC/cm^2$",
            fontsize=78,
            pad=-320,
        )

        patch_options = {
            "facecolor": "lime",
            "edgecolor": "black",
            "fill": True,
            "linewidth": 3,
            "alpha": 1.00,
            "zorder": 2,
        }
        elec_patches = get_patches(
            pos=elec_pos,
            radius=self.r_elec,
            hex_polar=self.HEX,
            hex_pitch=self.pitch,
            **patch_options,
        )

        # Add electrodes to axis + legend
        for i, patch in enumerate(elec_patches):
            patch.set(alpha=1.0)
            if i == 0:
                patch.set(label="Electrode", linewidth=0.75)
            ax_upper.add_patch(patch)
            art3d.pathpatch_2d_to_3d(patch, z=self.STIM_XYZ[2], zdir="z")

        ax_upper.legend(
            fontsize=46,
            loc=(0.025, 0.025),
            ncol=3,
            frameon=False,
            facecolor="white",
            framealpha=0.0,
            markerscale=1.75,
        )

        ax_upper.set_zlim(-82, 0)
        ax_upper.view_init(elev=3.5, azim=-90)

        pad = 65
        f_size = 56

        ax_upper.xaxis.set_tick_params(labelsize=f_size, pad=10)
        ax_upper.yaxis.set_tick_params(labelsize=0)
        ax_upper.zaxis.set_tick_params(labelsize=f_size, pad=pad)
        ax_upper.xaxis.set_major_locator(MaxNLocator(2))
        ax_upper.zaxis.set_major_locator(MultipleLocator(30))

        plt.tight_layout()

        plt.savefig(
            f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}/{pic_folder}"
            f"/activation-cross-section.jpg",
        )
        plt.savefig(
            f"{results_path}/run{self.run_id:03d}/gen{self.gen_id:04d}/{pic_folder}"
            f"/activation-cross-section-{q_dens}.pdf",
        )
        plt.close()

    def plot_mosaics_and_contours(
        self,
        loc_attribute: str = "pop_soma_loc",
    ) -> None:
        """
        Extracts the cell positions of interest (as per the segment specified
        by `loc_attribute`) and passes so into functions for plotting the
        hex-bin mosaics and KDE contours.

            - `loc_attribute`: the variable attribute of which contains the
                (x,y,z) positions of interest. This allows for different
                positions such as the dendritic tree (loc_attribute would be
                "pop_tree_loc") to be considered. By default, the location of
                soma is used.
        """

        # Pre-processing
        self.get_experiment_characteristics()  # need patch size + elec radius
        self.count_spikes()  # need spike count arrays
        self.get_cell_locations_and_type()  # need cell positions + types

        x_loc, y_loc, z_loc = getattr(self, loc_attribute, self.pop_soma_loc).T
        label = loc_attribute.split("_")[1].capitalize()  # for title/file names

        # Create dataset for visualisation/analysis functions
        cell_dataset = {
            "x": x_loc,
            "y": y_loc,
            "z": z_loc,
            "Cell type": self.pop_cell_types,
            "Weight": self.spike_weights,
        }

        # Plot contours
        if loc_attribute == "pop_tree_loc":
            self.plot_contours(cell_dataset, label=label)

    def write_stimulation_csv(self) -> None:
        """
        Consolidates the analytics and metrics into a .csv file,
        `stimulation.csv`, appending to so if it already exists.
        """
        # csv name
        spike_csv = f"{results_path}/run{self.run_id:03d}/stimulation.csv"

        spike_hdr = [
            "Current (uA)",
            "Charge density (mC/cm^2)",
            "ON",
            "OFF",
            "Total",
            "Total (normalised)",
            "ON (unique)",
            "OFF (unique)",
            "Total (unique)",
            "Total (unique) (normalised)",
            "Dendritic area activated (um^2)",
        ]

        if hasattr(self, "activated_area"):
            activated_area = np.sum(self.activated_area[~np.isnan(self.activated_area)])
        else:
            activated_area = 0

        spike_entry = np.array(
            [
                [self.amp],
                [np.round(self.charge_dens, 4)],
                [self.spikes_on],
                [self.spikes_off],
                [int(self.total_spikes)],
                [np.round(self.total_spikes / self.GEN_SIZE, 4)],
                [self.spikes_on_unique],
                [self.spikes_off_unique],
                [int(self.total_spikes_unique)],
                [np.round(self.total_spikes_unique / self.GEN_SIZE, 4)],
                [np.round(activated_area, 4)],
            ]
        ).T

        # Append/write to .csv
        pd.DataFrame(spike_entry).to_csv(
            spike_csv,
            mode="a",  # appends if existing, creates otherwise
            header=None if os.path.isfile(spike_csv) else spike_hdr,
            sep=",",
            index=False,  # indexing is messy when appending
            float_format="%.3f",
        )

    def update_threshold_csv(self, sort: bool = True) -> None:
        """
        Reads the response threshold .csv file to compare against any existing
        entries for this electrode position. The lower threshold is retained.

            - `sort`: sorts the output threshold .csv file based on electrode
                position.

        """

        spikes = int(self.total_spikes)

        # Get absolute electrode pos relative to foveola
        x, y, z = (
            float(self.STIM_XYZ[0]),
            float(self.STIM_XYZ[1]),
            float(self.STIM_XYZ[2]),
        )
        self.absolute_elec_position = [x, y, z]
        self.absolute_elec_position[0] += self.ecc * 1000  # shift by pop's ecc
        self.absolute_elec_position = np.round(self.absolute_elec_position, 1)
        self.absolute_elec_position = str(self.absolute_elec_position)

        threshold_csv = f"{results_path.split('/trial')[0]}/thresholds.csv"

        header = [
            "Electrode location (um)",
            "Waveform",
            "Current (uA)",
            "Charge density (mC/cm^2)",
            "Activated cells",
            "Trial/run/gen",
        ]
        amp_index = header.index("Current (uA)")

        entry = np.array(
            [
                [self.absolute_elec_position],
                [self.stim_type],
                [np.round(self.amp, 4)],
                [np.round(self.charge_dens, 3)],
                [spikes],
                [(self.trial_id, self.run_id, self.gen_id)],
            ],
            dtype=object,
        ).T

        # Goal is to record
        # 1) highest amp for ZERO response
        # 2) response threshold (minimum amp to elicit a response)
        # 3) first amplitude for more than 1 spike (end of single-cell window)

        if not os.path.isfile(threshold_csv):  # .csv does not exist yet
            pd.DataFrame(entry).to_csv(
                threshold_csv,
                mode="a",  # appends if existing, creates otherwise
                header=header,
                sep=",",
                index=False,  # indexing is messy when appending
                float_format="%.3f",
            )
            return

        # read existing .csv (if any) and compare against current threshold
        editing_csv = False
        thresholds = pd.read_csv(threshold_csv)

        # No spikes (null response)
        if spikes == 0:
            mask = thresholds.loc[
                (thresholds["Electrode location (um)"] == self.absolute_elec_position)
                & (thresholds["Waveform"] == self.stim_type)
                & (thresholds["Activated cells"] == 0)
            ]
            if np.size(mask.index):
                index = mask.index[0]
                existing_entry = thresholds.iloc[index].to_numpy()
                existing_amp = existing_entry[amp_index]
                if self.amp > existing_amp:  # HIGHEST amp whilst null
                    editing_csv = True

        # Single-cell resolution (spikes = 1)
        elif spikes == 1:
            mask = thresholds.loc[
                (thresholds["Electrode location (um)"] == self.absolute_elec_position)
                & (thresholds["Waveform"] == self.stim_type)
                & (thresholds["Activated cells"] == 1)
            ]
            if np.size(mask.index):
                index = mask.index[0]
                existing_entry = thresholds.iloc[index].to_numpy()
                existing_amp = existing_entry[amp_index]
                if self.amp < existing_amp:  # LOWEST amp for threshold
                    editing_csv = True

        # End of single-cell resolution (spikes > 1)
        elif spikes > 1:
            mask = thresholds.loc[
                (thresholds["Electrode location (um)"] == self.absolute_elec_position)
                & (thresholds["Waveform"] == self.stim_type)
                & (thresholds["Activated cells"] > 1)
            ]
            if np.size(mask.index):
                index = mask.index[0]
                existing_entry = thresholds.iloc[index].to_numpy()
                existing_amp = existing_entry[amp_index]
                if self.amp < existing_amp:  # LOWEST amp above threshold
                    editing_csv = True

        # this config does not exist in the .csv file
        if np.size(mask.index) == 0:
            pd.DataFrame(entry).to_csv(
                threshold_csv,
                mode="a",  # appends if existing, creates otherwise
                header=None,
                sep=",",
                index=False,  # indexing is messy when appending
                float_format="%.3f",
            )

        elif editing_csv:
            thresholds.iloc[index] = entry[0]
            if sort:
                thresholds.sort_values(
                    ["Electrode location (um)", "Waveform", "Current (uA)"],
                    axis=0,
                    ascending=True,
                    inplace=True,
                    ignore_index=True,
                )
            thresholds.to_csv(
                threshold_csv,
                mode="w",
                header=header,
                sep=",",
                index=False,
                float_format="%.3f",
            )

    def run_visualisation_pipeline(self) -> None:
        """
        Runs a series of functions to generate quantitative metrics and
        visualisations/plots associated with the population response.
        """

        # Visualisation
        self.plot_mosaics_and_contours(loc_attribute="pop_soma_loc")
        self.plot_mosaics_and_contours(loc_attribute="pop_tree_loc")
        self.plot_xz_activation()

        # ensure all figures are closed
        plt.close("all")

    def run_analysis_pipeline(
        self,
        run_visualisation: bool = True,
        delete_neuron_responses: bool = False,
        update_threshold_csv: bool = False,
        save: bool = True,
    ) -> None:
        """
        Runs a series of analysis functions to save relevant population data
        and characteristics within Agents.

            - `run_visualisation`: if True, run visualisation pipeline.
            - `delete_neuron_responses`: if True, after amalgamating all NEURON
                response vectors into .csv files, deletes all such NEURON
                produced .txt vector files.
            - `update_threshold_csv`: if True, compares current threshold and
                electrode location to the master threshold .csv file, updating
                the file if required.
            - `save`: extra toggle to disable saving when seeking to only load
                or leverage analysis scripts (avoids interfering/overwriting data).

        """
        # Analyse response vectors
        self.neuron_responses_to_csv()
        print("INFO: Neuron responses amalgamated into .csv files.")
        self.get_experiment_characteristics()
        print("INFO: Experiment characteristics embedded into population.")
        self.get_cell_locations_and_type()
        print("INFO: Cell locations and types embedded into population.")
        self.population_spike_detection()
        # self.count_spikes()
        print("INFO: Spikes counted.")
        self.write_stimulation_csv()
        print("INFO: Stimulation .csv file updated.")

        if self.HEX and update_threshold_csv:
            self.update_threshold_csv()

        if save:
            self.save_agents()

        if self.total_spikes > 0:
            if run_visualisation:
                self.run_visualisation_pipeline()

        if delete_neuron_responses:
            self.delete_neuron_responses()
