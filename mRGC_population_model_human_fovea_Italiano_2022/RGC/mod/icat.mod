TITLE HH style ICaT channel for RGCs
: 
: David Tsai, March 2012
: Using activation and inactivation functions of RGC_prime.m by T Gou
: 

NEURON {
    SUFFIX icat
    USEION ca READ cai, eca, cao WRITE ica
    RANGE gcatbar
    RANGE i

    : parameters for optimization
    RANGE catm1, catm2, catm3, catm4, catm5, catm6, catm7, catm8
    RANGE cath1, cath2, cath3, cath4, cath5, cath6
    RANGE catd1, catd2, catd3, catd4, catd5, catd6
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    gcatbar = 0.004  (mho/cm2)
    eca              (mV)
    cao     = 1.8    (mM)
    cai     = 0.0001 (mM)
    dt               (ms)
    v                (mV)

    : parameters for optimization
    catm1 = 0.7428
    catm2 = -42.2732
    catm3 = -0.3359
    catm4 = 1.9992
    catm5 = -65.0674
    catm6 = -0.2734
    catm7 = -88.3245
    catm8 = 0.0371
    cath1 = 1.5086
    cath2 = -150.1338
    cath3 = 0.0694
    cath4 = 0.8956
    cath5 = -42.4090
    cath6 = -0.0201
    catd1 = 0.0057
    catd2 = -84.533
    catd3 = 0.0782
    catd4 = 0.0022
    catd5 = 14.9596
    catd6 = -0.0385          
}

STATE {
    m h d
}

ASSIGNED {
    i      (mA/cm2)
    ica    (mA/cm2)
}

INITIAL {
    m = 0.9912
    h = 0.0085
    d = 0.3863
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gcatbar * m^3 * h * (v-eca)
    ica = i
}

DERIVATIVE states {
    : compute state variables at present v and t
    m' = (1-m) * alpha_m(v) - m * beta_m(v)
    h' = (1-h-d) * alpha_h(v) - h * beta_h(v)
    d' = (1-h-d) * alpha_d(v) - d * beta_d(v)
}

FUNCTION alpha_m(Vm (mV))  (/ms) {
    UNITSOFF
    alpha_m = catm1 / (1 + exp((v-catm2)*catm3))
    UNITSON
}

FUNCTION beta_m(Vm (mV))  (/ms) {
    UNITSOFF
    beta_m = catm4 / (1 + exp((v-catm5)*catm6)) + catm4/(1 + exp((v-catm7)*catm8))
    UNITSON
}

FUNCTION alpha_h(Vm (mV))  (/ms) {
    UNITSOFF
    alpha_h = cath1 * exp((v-cath2)*cath3)
    UNITSON
}

FUNCTION beta_h(Vm (mV))  (/ms) {
    UNITSOFF
    beta_h = cath4 / (1 + exp((v-cath5)*cath6))
    UNITSON
}

FUNCTION alpha_d(Vm (mV))  (/ms) {
    UNITSOFF
    alpha_d = catd1 / (1 + exp((v-catd2)*catd3))
    UNITSON
}

FUNCTION beta_d(Vm (mV))  (/ms) {
    UNITSOFF
    beta_d = catd4 * (v-catd5) / (1 - exp((v-catd5)*catd6))
    UNITSON
}

