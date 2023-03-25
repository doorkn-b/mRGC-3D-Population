TITLE HH style channels for spiking retinal ganglion cells
:
: Modified from Fohlmeister et al, 1990, Brain Res 510, 343-345
: by TJ Velte March 17, 1995
: must be used with calcium pump mechanism, i.e. capump.mod
:
:

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
    SUFFIX spikeOptm
    USEION na READ ena WRITE ina
    USEION k READ ek WRITE ik
    USEION ca READ cai, eca, cao WRITE ica
    RANGE gnabar, gkbar, gabar, gcabar, gkcbar
    RANGE m_inf, h_inf, n_inf, p_inf, q_inf, c_inf
    RANGE tau_m, tau_h, tau_n, tau_p, tau_q, tau_c
    RANGE m_exp, h_exp, n_exp, p_exp, q_exp, c_exp
    RANGE idrk, iak, icak

    : parameters for optimization
    RANGE nam1, nam2, nam3, nam4, nam5, nam6, nam7
    RANGE nah1, nah2, nah3, nah4, nah5, nah6
    RANGE kn1, kn2, kn3, kn4, kn5, kn6, kn7
    RANGE kp1, kp2, kp3, kp4, kp5, kp6, kp7
    RANGE kq1, kq2, kq3, kq4, kq5, kq6
    RANGE ca1, ca2, ca3, ca4, ca5, ca6, ca7
}


UNITS {
    (molar) = (1/liter)
    (mM) = (millimolar)
    (mA) = (milliamp)
    (mV) = (millivolt)

}

PARAMETER {
    gnabar  = 0.04    (mho/cm2)
    gkbar   = 0.012   (mho/cm2)
    gabar   = 0.036   (mho/cm2)
    gcabar  = 0.002   (mho/cm2)
    gkcbar  = 0.00005 (mho/cm2)
    ena     = 35  (mV)
    ek      = -75 (mV)
    eca           (mV)
    cao     = 1.8 (mM)
    cai     = 0.0001 (mM)
    dt            (ms)
    v             (mV)

    : parameters for optimization
    nam1 = -0.6
    nam2 = 30
    nam3 = -0.1
    nam4 = 30
    nam5 = 20
    nam6 = 55
    nam7 = 18
    nah1 = 0.4
    nah2 = 50
    nah3 = 20
    nah4 = 6
    nah5 = -0.1
    nah6 = 20
    kn1 = -0.02
    kn2 = 40
    kn3 = -0.1
    kn4 = 40
    kn5 = 0.4
    kn6 = 50
    kn7 = 80
    kp1 = -0.006
    kp2 = 90
    kp3 = -0.1
    kp4 = 90
    kp5 = 0.1
    kp6 = 30
    kp7 = 10
    kq1 = 0.04
    kq2 = 70
    kq3 = 20
    kq4 = 0.6
    kq5 = -0.1
    kq6 = 40
    ca1 = -0.3
    ca2 = 13
    ca3 = -0.1
    ca4 = 13
    ca5 = 10
    ca6 = 38
    ca7 = 18
}

STATE {
    m h n p q c 
}

INITIAL {
: The initial values were determined at a resting value of -66.3232 mV in a 
: single-compartment
:    m = 0.0155
:    h = 0.9399
:    n = 0.0768
:    p = 0.0398
:    q = 0.4526
:    c = 0.0016
: at -60 mV
    m = 0.0345
    h = 0.8594
    n = 0.1213
    p = 0.0862
    q = 0.2534
    c = 0.0038
}

ASSIGNED {
    ina    (mA/cm2)
    ik     (mA/cm2)
    idrk   (mA/cm2)
    iak    (mA/cm2)
    icak   (mA/cm2)
    ica    (mA/cm2)
    m_inf h_inf n_inf p_inf q_inf c_inf
    tau_m tau_h tau_n tau_p tau_q tau_c
    m_exp h_exp n_exp p_exp q_exp c_exp
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina = gnabar * m*m*m*h * (v - ena)
    idrk = gkbar * n*n*n*n * (v - ek)
    iak =  gabar * p*p*p*q * (v - ek)
    icak = gkcbar * ((cai / 0.001)/ (1 + (cai / 0.001))) * (v - ek)
    ik = idrk + iak + icak
    ica = gcabar * c*c*c * (v - eca)
}

DERIVATIVE states {    : exact when v held constant
    evaluate_fct(v)
    m' = (m_inf-m)/tau_m
    h' = (h_inf-h)/tau_h
    n' = (n_inf-n)/tau_n
    p' = (p_inf-p)/tau_p
    q' = (q_inf-q)/tau_q
    c' = (c_inf-c)/tau_c
} 

UNITSOFF

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b
    
:NA m
    a = (nam1 * (v+nam2)) / ((exp(nam3*(v+nam4))) - 1)
    b = nam5 * (exp((-1*(v+nam6))/nam7))
    tau_m = 1 / (a + b)
    m_inf = a * tau_m

:NA h
    a = nah1 * (exp((-1*(v+nah2))/nah3))
    b = nah4 / ( 1 + exp(nah5 *(v+nah6)))
    tau_h = 1 / (a + b)
    h_inf = a * tau_h

:K n (non-inactivating, delayed rectifier)
    a = (kn1 * (v+kn2)) / ((exp(kn3*(v+kn4))) - 1)
    b = kn5 * (exp((-1*(v + kn6))/kn7))
    tau_n = 1 / (a + b)
    n_inf = a * tau_n

:K p (inactivating)
    a = (kp1 * (v+kp2)) / ((exp(kp3*(v+kp4))) - 1)
    b = kp5 * (exp((-1*(v + kp6))/kp7))
    tau_p = 1 / (a + b)
    p_inf = a * tau_p

:K q (inactivating)
    a = kq1 * (exp((-1*(v+kq2))/kq3))
    b = kq4 / (1 + exp(kq5 *(v+kq6)))    
    tau_q = 1 / (a + b)
    q_inf = a * tau_q

:CA channel
    a = (ca1 * (v+ca2)) / ((exp(ca3*(v+ca4))) - 1)
    b = ca5 * (exp((-1*(v + ca6))/ca7))
    tau_c = 1 / (a + b)
    c_inf = a * tau_c

}

UNITSON

