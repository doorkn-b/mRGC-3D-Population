"""

A driver script that leverages the classes of tps_waveforms.py to mass-create 
and mass-simulate cells/waveforms for stimulation of RGC populations. 

@author: M.L. Italiano

"""

import time
import os
import sys
import fnmatch
import numpy as np
import shutil
import re
from typing import Tuple
from joblib import Parallel, delayed
import multiprocessing
import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "":
    print("INFO: No display found. Using non-interactive Agg backend")
    mpl.use("Agg")  # avoids issue with non-interactive displays as w/ EC2

import matplotlib.pyplot as plt

import tps_waveforms as tg

from foveal_tiles import Population, Neuron  # needed for pickle load
from helpers.common import SIN, EXP, BIPHASIC, ANODIC, TRIANGLE

plt.style.use("./helpers/plots.mplstyle")

num_cores = multiprocessing.cpu_count()


class Settings:

    TRIAL = 0  # The tile trial number (from which the cells are loaded)
    RUN = 0
    SIMULATING = True  # Toggle for simulating w/ or w/o NEURON [T/F]
    UPDATE_THRESHOLD_CSV = False  # toggle: track thresholds across retina
    ANALYSE = True
    DELETE_ON_EXIT = True  # Delete NEURON .txt files after combining into .csv
    RUN_VISUALS = True  # run visualisation pipeline
    PARALLEL = True  # Python driven parallel simulations
    SIMULTANEOUS = False  # simulate ALL cells at once (one file, one tile)

    HEX = False  # hex array (True), monopolar (False)
    STIM_FILE = "stim/stimHex-foveal.hoc" if HEX else "stim/stimTps-foveal.hoc"

    SPIKE_PLOT = False  # Individual spike plots toggle

    WARN = False  # Warn user to double-check TRIAL if existing data is found
    OVERWRITE = False  # Enable overwrite of any existing data

    ####################### STIM. WAVEFORM PARAMETERS #########################

    WAVE_TYPE = BIPHASIC

    stim_xyz = [0, 0, -80]  # (x,y,z) microns

    amplitude = [1.0]
    SKIP = 0  # skip this many amplitudes (i.e., starts from this index)

    ####################### STIM. TIMING PARAMETERS ###########################

    ########## DO NOT CHANGE ##########
    dt = 0.025 * 1e-3  # time between NEURON samples [ms]
    ###################################

    n_pulse = 4  # no. of pulses used in a simulation
    t_pw = 0.1  # pulse width [ms]

    n_step = 10  # number of discretised values per phase (for exp/sin)
    N_phase = 10  # ratio of phase 1 to phase 2 duration (inverse amp ratio) (AA)

    if WAVE_TYPE == ANODIC:
        t_pulse = (N_phase + 1) * t_pw  # duration of each biphasic pulse [ms]
    else:
        t_pulse = 2 * t_pw
    t_pre = 10  # [ms]
    t_post = 20  # [ms]
    t_interphase = 0 if WAVE_TYPE in (BIPHASIC, ANODIC) else 0  # [ms]
    t_pulse += t_interphase
    t_interpulse = 5.0  # [ms]
    t_tau = 0.1  # time-constant wrt t_pw (pulse duration) (for exp waveform)

    t_stim = t_pulse * n_pulse + (n_pulse - 1) * t_interpulse  # [ms]
    t_total = t_pre + t_stim + t_post  # [ms]

    ###########################################################################


def save_stim_info(path: str, amp: float) -> None:
    """Saves waveform and electrode configuration for record-keeping."""

    fName = f"{path}/stimulation-info.txt"
    f = open(fName, "w")

    # Waveform type
    f.write(f"Waveform type: {Settings.WAVE_TYPE}\n\n")

    # Electrode info
    if Settings.HEX:
        f.write(f"Hex electrode array was used.\n")
    else:
        f.write(f"Monopolar electrode was used.\n")
    f.write(f"stim XYZ: {Settings.stim_xyz}\n\n")

    # Timing info
    f.write(f"t_pw is {Settings.t_pw} ms.\n")
    f.write(f"t_pulse is {Settings.t_pulse} ms.\n")
    f.write(f"t_pre is {Settings.t_pre} ms.\n")
    f.write(f"t_post is {Settings.t_post} ms.\n")
    f.write(f"t_interpulse is {Settings.t_interpulse} ms.\n")
    f.write(f"t_interphase is {Settings.t_interphase} ms.\n")
    f.write(f"t_stim is {Settings.t_stim} ms.\n")
    f.write(f"t_total (tstop) is {Settings.t_total} ms.\n\n")

    # N pulses and amplitude
    f.write(f"N_phase is {Settings.N_phase}.\n")
    f.write(f"n_pulse is {Settings.n_pulse}.\n")
    f.write(f"n_step is {Settings.n_step}.\n")
    f.write(f"t_tau is {Settings.t_tau} * pw.\n")
    f.write(f"amplitude is {amp:.4f} uA.\n\n")

    # stimTps-foveal info
    f2 = open(f"{path.split('/results')[0]}/{Settings.STIM_FILE}")
    printer = False

    for line in f2:
        if "START point" in line:
            printer = True
            continue
        elif "elecRad" in line:
            tg.r_elec = float(line.split("= ")[-1].split(" ")[0])
        elif "STOP point" in line:
            break
        if printer:
            f.write(line)

    f2.close()
    f.close()


def save_sim_time(gen_path: str, sim_time: float) -> None:
    """
    Saves simulation time (hours) to .txt file in the corresponding gen folder
    located at `gen_path`.
    """
    with open(f"{gen_path}/simulation-time.txt", mode="w") as f:
        f.write(f"Simulation took {sim_time / 3600:.4f} hours.")


def check_existing_data(trial: int, path: str) -> Tuple[int, str]:

    while os.path.isdir(path):

        print("\n*** TRIAL ID is %.3d ***" % trial)
        print("\nINFO: Pre-existing data for this trial has been found.")
        print("INFO: Failure to update the TRIAL field may overwrite data.")
        assert (
            Settings.overwrite
        ), "Overwrite is disabled...initiating obligatory abort."
        print("\nCommands - [Y]: proceed, [N]: update TRIAL, [other]: abort")
        response = input("Do you still wish to continue? [Y/N/other] : ")
        assert (
            response == "Y" or response == "N"
        ), "Trial number not updated - ABORTING!"
        if response == "Y":
            response = input("Are you sure? [_/N] : ")
        if response == "Y":
            if Settings.OVERWRITE:
                shutil.rmtree(path)
                print(
                    f"INFO: Deleted all pre-existing simulation data for trial {trial:03d}!\n"
                )
        elif response == "N":
            TRIAL = int(input("\nInput new TRIAL ID: "))
        path = path[:-4] + "%.3d" % TRIAL
        path = f"{path.split('trial_')[0]}{TRIAL:03d}/sim"

    return trial, path


def drive():

    TRIAL = Settings.TRIAL
    results_path = f"../RGC-fovea/results/trial_{TRIAL:03d}/sim"
    tg.results_path = results_path

    simulating = Settings.SIMULATING
    tg.Generation.T_TOTAL = Settings.t_total  # update timing variable
    tg.Generation.HEX = Settings.HEX
    tg.Generation.STIM_XYZ = Settings.stim_xyz
    tg.dt_nrn = Settings.dt  # NEURON sample time [ms]
    tg.t_pw = Settings.t_pw
    tg.t_pre = Settings.t_pre
    tg.t_post = Settings.t_post
    tg.n_pulse = Settings.n_pulse
    tg.t_interphase = Settings.t_interphase
    tg.t_interpulse = Settings.t_interpulse
    tg.n_step = Settings.n_step
    tg.t_tau = Settings.t_tau

    warn_user = Settings.WARN

    ### WARN USER ###
    path = os.getcwd()
    path = f"{path}/{results_path}/"
    if warn_user:
        TRIAL, path = check_existing_data(TRIAL, path)

    # Find number of cells and update gen_size accordingly
    ROOT_PATH = os.path.abspath(os.getcwd()).split("/RGC")[0]
    HOC_SOURCE = f"{ROOT_PATH}/RGC-fovea/results/trial_{TRIAL:03d}/cellHoc"
    n_cells = len(fnmatch.filter(os.listdir(HOC_SOURCE), "cell*.hoc"))
    tg.Generation.GEN_SIZE = n_cells
    print(f"INFO: Number of cells in this population is {n_cells}")
    assert (
        tg.Generation.GEN_SIZE
    ), f"ERROR: Empty population - no .hoc files were found at {HOC_SOURCE}."

    tick = time.time()  # timer
    print(f"\n\n*** INFO: Commencing TRIAL {TRIAL:03d}! ***\n")

    # Determine number of generations
    if simulating:
        n_gen = len(Settings.amplitude)

    else:
        r_search = re.compile("gen\d{3}")  # regex search for genXXX
        n_gen = len(
            list(
                filter(
                    r_search.match, os.listdir(f"{results_path}/run{Settings.RUN:03d}/")
                )
            )
        )

    for i in range(n_gen):
        if Settings.SKIP and i < Settings.SKIP:
            if simulating:
                print(f"INFO: Skipping {Settings.amplitude[i]:.2f} uA (amp {i}).")
            else:
                print(f"INFO: Skipping gen{i:03d}.")
            continue

        # Initialisation values
        tot_start = time.time()
        gen_path = f"{results_path}/run{Settings.RUN:03d}/gen{i:04d}/"

        # Simulate waveforms
        if simulating:

            cur_amp = Settings.amplitude[i]
            tg.AMP = cur_amp
            tg.N_phase = Settings.N_phase

            # Create waveform and spawn + update generation
            tg.Agent.PULSES = Settings.n_pulse
            tg.Agent.NON_ZERO_AMPS = [cur_amp]
            tg.Agent.N_phase = Settings.N_phase

            # spawn population
            gen = tg.Generation(
                trial_id=TRIAL,
                run_id=Settings.RUN,
                gen_id=i,
                stim_type=Settings.WAVE_TYPE,
            )
            gen.save_agents()  # establishes directories

            # Update records
            gen.agents[0].stimulus.stim_plot(pic_path=gen_path)
            save_stim_info(gen_path, cur_amp)

            if Settings.SIMULTANEOUS:
                gen.simulate_simultaneous_waveforms(write_hoc=True)

            elif Settings.PARALLEL:
                gen.get_cell_locations_and_type()
                Parallel(n_jobs=num_cores, require="sharedmem")(
                    delayed(gen.simulate_parallel_waveforms)(
                        cell_id=c,
                        write_hoc=True,
                        master=False,
                    )
                    for c in range(gen.GEN_SIZE)
                )

            else:
                gen.write_hoc_files()
                gen.simulate_waveforms()

            gen.save_agents()

        else:  # loading from prev. sim. results
            gen = tg.Generation.load(
                run_id=Settings.RUN,
                gen_id=i,
                trial_id=Settings.TRIAL,
            )

            # Defaults - the rest is embedded in the agent data
            npulse = None
            t_ipi = None
            cur_amp = None
            n_step = 10
            N_phase = 1
            t_pw = 0.1
            t_interphase = 0
            pos_elec = np.array([0, 0, -80])  # default elec position

            fName = f"{gen_path}/stimulation-info.txt"

            # Apply tpsGa variables from gen stimulation-info.txt
            with open(fName, mode="r") as f:
                for line in f:
                    if "Hex electrode" in line:
                        tg.Generation.HEX = True
                    elif "stim XYZ" in line:
                        pos_elec[0] = float(line.split("[")[-1].split(",")[0])
                        pos_elec[1] = float(line.split(", ")[1].split(",")[0])
                        pos_elec[2] = float(line.split(", ")[-1].split("]")[0])
                    elif "t_pw" in line:
                        t_pw = float(line.split("t_pw is ")[-1].split(" ms")[0])
                    elif "t_pulse" in line and t_pw is None:
                        t_pulse = line.split("t_pulse is ")[-1].split(" ms")[0]
                        t_pw = f"{float(t_pulse)/2}"
                    elif "t_interphase" in line:
                        t_interphase = float(
                            line.split("t_interphase is ")[-1].split(" ms")[0]
                        )
                        tg.t_interphase = t_interphase
                    elif "t_interpulse" in line:
                        t_ipi = float(
                            line.split("t_interpulse is ")[-1].split(" ms")[0]
                        )
                        tg.t_interpulse = t_ipi
                    elif "t_total" in line:
                        t_stop = float(line.split("is ")[-1].split(" ms")[0])
                    elif "N_phase" in line:
                        N_phase = int(line.split("N_phase is ")[-1].split(".")[0])
                        tg.Agent.N_phase = N_phase
                    elif "n_pulse" in line:
                        npulse = int(line.split("n_pulse is ")[-1].split(".")[0])
                        tg.n_pulse = npulse
                    elif "n_step" in line:
                        n_step = int(line.split("n_step is ")[-1].split(".")[0])
                        tg.n_step = npulse
                    elif "amplitude" in line:
                        cur_amp = float(line.split("amplitude is ")[-1].split("uA")[0])
                        break

            assert cur_amp is not None, f"LOAD FAILED: amplitude not found in {fName}"
            assert npulse is not None, f"LOAD FAILED: npulse not found in {fName}"
            assert t_ipi is not None, f"LOAD FAILED: ipi not found in {fName}"
            tg.Generation.STIM_XYZ = pos_elec
            tg.t_pw = t_pw

        sim_done = time.time()
        sim_time = np.round(sim_done - tot_start, 2)
        if simulating:
            save_sim_time(gen_path=gen_path, sim_time=sim_time)

        print(
            f"\nINFO: Simulation for trial {TRIAL:03d} run{Settings.RUN:04d} gen{i:04d} took: {sim_time} seconds\n"
        )
        if Settings.ANALYSE:
            gen.run_analysis_pipeline(
                run_visualisation=Settings.RUN_VISUALS,
                delete_neuron_responses=Settings.DELETE_ON_EXIT,
                update_threshold_csv=Settings.UPDATE_THRESHOLD_CSV,
            )

        t_analysis = time.time()

        print(
            f"INFO: Analysis for trial {TRIAL:03d} run{Settings.RUN:04d} gen{i:04d} took: {t_analysis - sim_done:.2f} seconds\n"
        )

        print(f"INFO: Overall gen time taken: {time.time() - tot_start:0.2f} seconds\n")

    print(f"\nINFO: Overall time taken: {time.time() - tick:0.2f} seconds\n")
    return gen.total_spikes


################################### MAIN ######################################


def main(argv):
    L = len(argv)
    if L >= 2:
        print("INFO: Using command-line inputs:\n")
        Settings.TRIAL = int(argv[1])
        print(f"INFO: TRIAL is {Settings.TRIAL}.\n")
        if L >= 3:
            Settings.amplitude = [float(argv[2])]
            print(f"INFO: Amplitude is {Settings.amplitude}.\n")
        if L >= 4:
            Settings.n_pulse = int(argv[3])
            print(f"INFO: No. of pulses is {Settings.n_pulse}.\n")

    drive()


if __name__ == "__main__":
    main(sys.argv[:])
