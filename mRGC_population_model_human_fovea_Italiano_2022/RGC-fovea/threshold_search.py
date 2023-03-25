"""

Automated threshold sweep of all populations (trials) requested. Only retains 
gen data starting from the highest stimulus strength null response and ending 
with the response with more than 1 activated cell. 

@author: M.L. Italiano

"""

import os
from typing import List

import tps_waveforms as tg
import stimulate_tile
from helpers.common import SIN, EXP, BIPHASIC, ANODIC, TRIANGLE

############################## PARAMETERS #####################################

TRIAL_START = 0
TRIAL_END = 0

JITTER = False  # jitter the electrode +- 5 in x- and y-directions

RUN = 0

WAVEFORM = BIPHASIC
HEX = True
ELEC_Y = 0  # electrode's y-position

# removes null results (excl. highest amplitude null result) to save space
REMOVE_EXCESS = True

###############################################################################


def get_starting_amplitude(pos: tuple = None) -> float:
    """
    Returns an appropriate starting amplitude based on the position (x,y,z),
    `pos`.
    XXX: incomplete and currently requires (informed) manual adjustment.
    """
    if pos is None:
        amp = 0.7
    else:
        # TODO: check for nearest pos in list of known thresholds and select
        # appropriate starting amplitude based on so.
        pass

    return amp


def get_jitter_pos(pos: List[float]) -> List[List[float]]:
    """
    Returns the jittered electrode positions as a list of [x,y,z] based on the
    mean (no jitter) `pos`.
    """
    x, y, z = pos
    jittered_pos = []
    jittered_pos.append([x - 5, y, z])
    jittered_pos.append([x + 5, y, z])
    jittered_pos.append([x, y - 5, z])
    jittered_pos.append([x, y + 5, z])
    return jittered_pos


def search(run: int = RUN, stim_xyz: List[float] = [0, ELEC_Y, -80]) -> None:

    x, y, z = stim_xyz
    if y:
        SPEEDRUN_X = [x - 155, x + 40]
        SPEEDRUN_Y = [min(-30, y - 65), y + 65]
    else:  # along horizontal meridian
        SPEEDRUN_X = [-125, 65]
        SPEEDRUN_Y = [-75, 75]

    tg.SPEEDRUN_X = SPEEDRUN_X
    tg.SPEEDRUN_Y = SPEEDRUN_Y

    stimulate_tile.Settings.SIMULATING = True
    stimulate_tile.Settings.ANALYSE = True
    stimulate_tile.Settings.RUN_VISUALS = True
    stimulate_tile.Settings.UPDATE_THRESHOLD_CSV = True
    stimulate_tile.Settings.HEX = HEX
    stimulate_tile.Settings.STIM_FILE = (
        "stim/stimHex-foveal.hoc" if HEX else "stim/stimTps-foveal.hoc"
    )
    stimulate_tile.Settings.WAVE_TYPE = WAVEFORM

    stimulate_tile.Settings.RUN = run
    stimulate_tile.Settings.stim_xyz = stim_xyz

    last_known_null = get_starting_amplitude()

    for trial in range(TRIAL_START, TRIAL_END + 1):

        test_amplitudes = [last_known_null]

        stimulate_tile.Settings.TRIAL = trial
        stimulate_tile.Settings.SKIP = 0

        searching = True

        while searching:

            stimulate_tile.Settings.amplitude = test_amplitudes
            spikes = stimulate_tile.drive()

            if spikes == 0:

                # Increment next testing amplitude
                last_known_null = test_amplitudes[-1]
                n_tested = len(test_amplitudes)
                stimulate_tile.Settings.SKIP = n_tested
                test_amplitudes.append(test_amplitudes[-1] + 0.1)

                # don't store multiple old null results, only need one
                if n_tested >= 2 and REMOVE_EXCESS:
                    path_to_gen = f"./results/trial_{trial:03d}/sim/run{RUN:03d}"
                    rm_command = f"rm -r {path_to_gen}/gen{(n_tested-2):04d}"
                    os.system(rm_command)

            elif spikes == 1:  # keep searching for end of single-cell activation
                stimulate_tile.Settings.SKIP = len(test_amplitudes)
                test_amplitudes.append(test_amplitudes[-1] + 0.1)

            elif spikes > 1:
                searching = False


if __name__ == "__main__":
    if JITTER is False:
        search()
    else:
        jittered_pos = get_jitter_pos([0, ELEC_Y, -80])
        for i, p in enumerate(jittered_pos):
            search(run=RUN + i, stim_xyz=p)
            print(f"INFO: Done with threshold search for pos {p}.")

    print("INFO: Done with threshold search.")
