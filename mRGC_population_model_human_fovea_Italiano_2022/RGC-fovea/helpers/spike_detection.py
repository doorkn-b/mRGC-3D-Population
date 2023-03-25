import sys
import os
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple, List, Union
import matplotlib as mpl

if os.environ.get("DISPLAY", "") == "":
    mpl.use("Agg")
import matplotlib.pyplot as plt

TRIG_AMP = -10  # event detection threshold (mV)


def butter_lowpass(data, fs, cornerFreq, order=5):
    """
    Generate a Butterworth lowpass filter with passbands below `cornerFreq`,
    with order `order`. Then run filter on `data` with sampling frequency `fs`.
    """

    nyq = 0.5 * fs
    freq = cornerFreq / nyq
    b, a = butter(order, freq, btype="low", analog=False)
    return filtfilt(b, a, data)


def detect_spikes(data, time, period, trigAmp, winPre, winPost):
    """
    Detects spikes in `data` with sampling `period`. Crossing over `trigAmp` mV
    causes the event to be extracted, with `winPre` s before the peak and
    `winPost` s including and after the peak. These latter two values are
    rounded to align with data sampling period.
    """

    spikeV = []
    spikeT = []
    spikeCount = 0

    winPrePts = int(np.round(winPre / period))
    winPostPts = int(np.round(winPost / period))
    idx = winPrePts + 1

    mode = 2  # ignore=0, training, search, findPostPeak, track
    while idx < len(data):

        if mode == 2:  # search
            if data[idx] > trigAmp:
                mode = 3
            idx = idx + 1

        elif 3:  # findPostPeak
            if abs(data[idx]) < abs(data[idx - 1]):

                # prevent over-run
                if idx - 1 + winPostPts > len(data):
                    idx = len(data)
                    mode = 2
                    continue

                newSpike = data[idx - 1 - winPrePts : idx - 1 + winPostPts + 1]

                # preallocation
                if spikeCount == 0:
                    init_len = len(newSpike)  # make concatenation len a constant
                    spikeV = np.zeros([50, init_len])
                    spikeT = np.zeros([50, 1])
                elif spikeCount >= len(spikeT):
                    spikeV = np.concatenate((spikeV, np.zeros([50, init_len])), axis=0)
                    spikeT = np.concatenate((spikeT, np.zeros([50, 1])), axis=0)

                # Ran out of data; fix: pad until the end
                while np.size(newSpike, 0) < np.size(spikeV[spikeCount], 0):
                    newSpike = np.append(newSpike, data[-1])

                spikeV[spikeCount] = newSpike
                spikeT[spikeCount] = period * (idx - 1) + time[0]  # pre-trig offset
                spikeCount += 1

                idx = idx + winPostPts
                mode = 2

            else:
                idx = idx + 1

    # trim out extra allocations
    if spikeCount:
        spikeV = spikeV[0:spikeCount, :]
        spikeT = spikeT[0:spikeCount]

    return (spikeV, spikeT, trigAmp)


def run(
    data: Union[str, np.ndarray],
    fs: float = 1 / 0.025e-3,
) -> Tuple[List[List[float]], List[float], float]:
    """
    Run spike detection on response vector associated with `data`.
        - `data`: path to response vector (str) or response vector itself
            (np.ndarray).
        - `fs`: the sample frequency used (Hz).

    Returns a tuple of (spikeV, spikeT, trigAmp):
        - `spikeV`: voltage list windowed about spike events.
        - `spikeT`: times at which spikes occured.
        - `trigAmp`: voltage thresholded used to define spike (mV).
    """

    FILT_FREQ = 3500  # lowpass filter corner
    WIN_PRE = 1.0e-3  # time before event peak (s)
    WIN_POST = 3.0e-3  # time after event peak (s)

    if isinstance(data, str):  # load and filter is given file path
        data = np.loadtxt(data)
    dataF = butter_lowpass(data, fs, FILT_FREQ)

    # detect spikes
    time = np.arange(0, len(dataF) - 1) * 1 / fs
    (spikeV, spikeT, trigAmp) = detect_spikes(
        dataF, time, 1 / fs, TRIG_AMP, WIN_PRE, WIN_POST
    )

    return (spikeV, spikeT, trigAmp)


#############################################################################


def main(argv):
    pass


if __name__ == "__main__":
    plt.style.use("./plots.mplstyle")
    main(sys.argv[:])
