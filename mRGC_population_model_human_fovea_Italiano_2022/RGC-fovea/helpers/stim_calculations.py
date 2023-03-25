"""
Script to determine various stimulation (waveform) characteristics, such as 
pulse train frequency and charge density. 

@author: M.L. Italiano
"""

import sys
import numpy as np
from typing import Union


def calc_train_frequency(t_pw: float, t_ipi: float) -> float:
    """
    Given phase-width (t_pw) and interpulse interval length (t_ipi) (both in
    ms), returns the equivalent pulse train frequency.
    """

    t_pulse = 2 * t_pw
    freq = 1000 / (t_pulse + t_ipi)  # pulses in a second (i.e. frequency)

    print(f"TRAIN FREQUENCY: {freq:.2f} Hz ... for t_pw: {t_pw} ms, t_ipi: {t_ipi} ms")

    return freq


def calc_charge_density(
    amp: Union[float, np.ndarray], t_pw: float, r_elec: float
) -> float:
    """
    Uses amplitude (amp) in uA, pulse width (t_pw) in ms, and electrode radius
    (r_elec) in um to calculate the charge density of a single pulse.
    Returns charge density in mC/cm^2.
    """
    amp = np.squeeze(np.array([amp]))
    amp_a = amp * 1e-6  # uA to A
    t_pw_s = t_pw * 1e-3  # ms to s
    q_c = amp_a * t_pw_s  # charge [C]
    r_elec_cm = r_elec * 1e-4  # um to cm
    Q_D = q_c * 1e3 / (np.pi * r_elec_cm**2)  # 1e3 to get to mC/cm^2

    return Q_D


def main(argv):

    if len(argv) == 3:

        t_pw = float(argv[1])
        t_ipi = float(argv[2])

    else:
        t_pw = 0.1
        t_ipi = 5

    # calc_train_frequency(t_pw=t_pw, t_ipi=t_ipi)
    print(calc_charge_density(amp=250, t_pw=0.1, r_elec=200))


if __name__ == "__main__":
    main(sys.argv[:])
