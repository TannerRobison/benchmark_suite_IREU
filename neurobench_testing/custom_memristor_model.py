import torch
import memtorch
from memtorch.bh.memristor.Memristor import Memristor
from memtorch.utils import clip, convert_range #idk if ill need this

class MemtorchMemristor(Memristor):
    def __init__(
            self,
            k_off       = 1.0,      # switching rate for off state
            k_on        = -1.0,     # switching rate for on state
            alpha_off   = 5,        # exponent controlling nonlinearity
            alpha_on    = 5,        # exponent controlling nonlinearity
            i_off       = 0.5e-3,   # threshhold current to trigger off state
            i_on        = 0.5e-3,   # threshold current to trigger on state
            r_on        = 1e3,      # maximum resistance
            r_off       = 10e3,     # minimum resistance
            p           = 2,        # window function exponent
            **kwargs
    ):
        #initializing base memristor class
        super(MemtorchMemristor, self).__init__(r_off=r_off, r_on=r_on, **kwargs)

        # hyper parameters
        self.k_off = k_off
        self.k_on = k_on
        self.alpha_off = alpha_off
        self.alpha_on = alpha_on
        self.i_on = i_on
        self.i_off = i_off
        self.p = p

        #state variables
        self.w = 0.5
        self.g = 1/self.r_on

    def window()
