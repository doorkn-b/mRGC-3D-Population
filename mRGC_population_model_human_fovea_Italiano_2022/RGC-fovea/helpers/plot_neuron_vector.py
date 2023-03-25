"""
Move a generation's `stim-amp-vector.txt` and `stim-time-vector.txt` to 
RGC-fovea/results/ and run this script to quickly plot the stimulation
waveform as interpreted by NEURON.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

ROUND = False


def calc_net_charge(amp: np.ndarray, time: np.ndarray) -> float:
    charge = 0.00
    L = len(amp)
    assert L <= len(time)
    i = 0
    while i < L:

        amp_cur = amp[i]

        if i == (L - 1) and amp_cur == 0:
            break

        amp_next = amp[i + 1]
        assert amp_cur == amp_next

        t_delta = time[i + 1] - time[i]

        charge += t_delta * (amp_cur)

        i += 2

    return charge


if __name__ == "__main__":
    stim_txt = np.loadtxt(fname="../results/stim-amp-vector.txt")
    time_txt = np.loadtxt(fname="../results/stim-time-vector.txt")

    if ROUND:
        stim_txt = np.round(stim_txt, 6)
        time_txt = np.round(time_txt, 6)

    plt.plot(time_txt, stim_txt * 1000)  # *1000 as NEURON amp is in mA
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (uA)")

    print(f"Max. amplitude is \t {np.amax(abs(stim_txt[stim_txt > 0])*1000):.5f} uA.")
    print(f"Min. amplitude is \t-{np.amin(abs(stim_txt[stim_txt < 0])*1000):.5f} uA.")
    print(f"Net charge is  \t\t{calc_net_charge(stim_txt, time_txt)*1.00e3:.2e} nC.")
