import numpy as np
from scipy.integrate import solve_ivp

# TEAM (Threshold Adaptive Memristor) model as a reusable class
class TEAMMemristor:

    def __init__(
        self,
        k_off=1,          # switching rate for off-state
        k_on=-1,          # switching rate for on-state
        alpha_off=5,      # exponent controlling nonlinearity when switching off
        alpha_on=5,       # exponent controlling nonlinearity when switching on
        i_off=0.5e-3,     # threshold current to trigger off state switching
        i_on=-0.5e-3,     # threshold current to trigger on state switching
        g_on=1/1e3,       # maximum conductance (1 kohm = 1 ms)
        g_off=1/10e3,     # minimum conductance (10 kohm = 0.1 ms)
        w_init=0.5,       # initial state variable (0=off, 1=on)
        p=2               # window function exponent
    ):
        self.k_off = k_off
        self.k_on = k_on
        self.alpha_off = alpha_off
        self.alpha_on = alpha_on
        self.i_off = i_off
        self.i_on = i_on
        self.G_on = G_on
        self.G_off = G_off
        self.w_init = w_init
        self.p = p

    def set_state(self, w):
        # Update initial condition for next simulation
        self.w_init = np.clip(w, 0, 1)

    def window(self, w, i):
        # Nonlinear window function: reduces switching rate near the boundaries
        w = np.clip(w, 0.0, 1.0)
        if i >= 0:
            return 1 - w**(2*self.p)  # switching off: slower near w=1
        else:
            return 1 - (1-w)**(2*self.p)  # switching on: slower near w=0

    def conductance(self, w):
        # Linear interpolation between off and on conductance based on state w
        w = np.clip(w, 0.0, 1.0)
        return self.G_off + w*(self.G_on - self.G_off)

    def dw_dt(self, w, i):
        # TEAM state dynamics: dw/dt depends on current magnitude and direction
        w = np.clip(w, 0.0, 1.0)
        
        if i >= self.i_off:  # positive current above threshold = switch off
            dw = (
                self.k_off
                * ((i/self.i_off)-1)**self.alpha_off
                * self.window(w, i)
            )
        elif i <= self.i_on:  # negative current below threshold = switch on
            dw = (
                self.k_on
                * (((-i)/abs(self.i_on))-1)**self.alpha_on
                * self.window(w, i)
            )
        else:  # between thresholds = no switching
            dw = 0.0
        
        # Enforce physical bounds: prevent state from leaving [0,1]
        if w <= 0 and dw < 0:
            dw = 0
        if w >= 1 and dw > 0:
            dw = 0
        
        return dw

    def simulate(self,
                 freq=1,          # excitation frequency (Hz)
                 V_amp=1.5,       # sinusoid amplitude (V)
                 cycles=3):       # number of periods to simulate
        # Solve the memristor ODE for given frequency and voltage amplitude
        
        def voltage(t):  # sinusoidal excitation signal
            return V_amp*np.sin(2*np.pi*freq*t)
        
        def ode(t, y):  # dy/dt: current through memristor
            w = y[0]
            v = voltage(t)
            G = self.conductance(w)
            i = G*v  # Ohm's law: i = G*v
            return [self.dw_dt(w, i)]
        
        T = 1/freq  # period
        t_end = cycles*T
        t_eval = np.linspace(0, t_end, 10000)  # dense time grid for smooth curves
        
        # Solve with RK45, small max step
        sol = solve_ivp(
            ode,
            [0, t_end],
            [self.w_init],
            t_eval=t_eval,
            method='RK45',
            max_step=T/1000,  # max step keeps resolution within one period
            rtol=1e-8,
            atol=1e-10
        )
        
        raw_w = sol.y[0]
        
        # Check if numerical solver violated physical bounds
        eps = 1e-6
        if np.any(raw_w < -eps) or np.any(raw_w > 1+eps):
            print(
                "WARNING: solver left bounds "
                f"min={raw_w.min():.12f}, "
                f"max={raw_w.max():.12f}"
            )
        
        w = np.clip(raw_w, 0, 1)  # enforce bounds just in case
        t = sol.t
        v = voltage(t)
        G = self.conductance(w)
        i = G*v  # current throughout simulation
        
        return t, w, v, i

    def resistance(self, w):
        # Compute resistance as reciprocal of conductance
        return 1/self.conductance(w)
    
    def reset(self):
        # Reset state to default initial condition
        self.w_init = 0.5


