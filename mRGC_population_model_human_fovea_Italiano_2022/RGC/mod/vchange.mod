TITLE Tracking delta Vm, without and with lowpass filtering, for RGCs
: 
: David Tsai, June 2015
: 

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)

}

PARAMETER {
    baseline = -65 (mV)
}

NEURON {
    SUFFIX vchange
    RANGE delta
    RANGE xv1, xv2, xv3, xv4, xv5, xv6, yv1, yv2, yv3, yv4, yv5, yv6, deltaLP
}

ASSIGNED {
    delta    (mV)
    xv1      (mV)
    xv2      (mV)
    xv3      (mV)
    xv4      (mV)
    xv5      (mV)
    xv6      (mV)
    yv1      (mV)
    yv2      (mV)
    yv3      (mV)
    yv4      (mV)
    yv5      (mV)
    yv6      (mV)
    deltaLP  (mV)
    v        (mV)
}

INITIAL {
    delta = 0
    deltaLP = 0
}

BREAKPOINT {
    delta = v - baseline

    : The following coeffs are calculated for 40kHz sampling (0.025 ms dt)
    : 3-rd order Butterworth lowpass filter with 20 Hz corner
    xv1 = xv2
    xv2 = xv3
    xv3 = xv4
    xv4 = delta / 2.588234798e+08
    yv1 = yv2
    yv2 = yv3
    yv3 = yv4
    yv4 = (xv1 + xv4) + 3 * (xv2 + xv3)
                 + ( 0.9937365101 * yv1) + ( -2.9874533582 * yv2)
                 + ( 2.9937168173 * yv3)
    deltaLP = yv4
}

