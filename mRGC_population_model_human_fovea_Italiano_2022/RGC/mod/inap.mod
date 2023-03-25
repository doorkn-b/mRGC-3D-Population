
TITLE HH style INaP channel for RGCs
: 
: David Tsai, March 2012
: Using activation and inactivation functions of RGC_prime.m by T Gou
: 

NEURON {
	SUFFIX inap
    USEION na READ ena WRITE ina
    RANGE gnapbar
    RANGE p_inf, tau_p, p_exp
    RANGE i

    : parameters for optimization
    RANGE napp1, napp2, napp3, napp4, napp5, napp6, napp7, napp8
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    gnapbar = 2.125e-4 (mho/cm2)
    ena     = 35       (mV)
    dt                 (ms)
    v                  (mV)

    : parameters for optimization
    napp1 = 17.3413
    napp2 = -35.5471
    napp3 = -0.0244
    napp4 = -28.9507
    napp5 = 0.1120
    napp6 = 52.9680
    napp7 = -57.8397
    napp8 = 0.2625
}

STATE {
    p
}

ASSIGNED {
    i (mA/cm2)
    ina (mA/cm2)
    p_inf
    tau_p
    p_exp
}

INITIAL {
    p = 0.3863
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gnapbar * p * (v-ena)
    ina = i
}

DERIVATIVE states {
	evaluate(v) 
	p' = (p_inf - p)/tau_p

}

PROCEDURE evaluate(v(mV)) { LOCAL a, b
    :INap
    a = napp1 / ((1 + exp((v-napp2)*napp3)) * (1 + exp((v-napp4)*napp5)))
    b = napp6 / (1 + exp((v-napp7)*napp8))
    tau_p = 1 / (a + b)
    p_inf = p * tau_p

}




