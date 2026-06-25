import ctypes
import numpy
import torch

spires_lib = ctypes.CDLL("../spires/build/.libspires.so")

spires_lib.spires_reservoir_step.argtypes = [ctypes.POINTER(ctypes.c_float)]
spires_lib.spires_reservoir_step.restype = None

#change currents from tensor to a flat C pointer array for spires to read
def send_currents_to_spires(currents):
    # ----- extract from pytorhc graph -----
    # .detach() removes it from auto gradient tracking
    # .cpu() make sure data is in RAM, not VRAM
    # .numpy() maps it to numpy array
    numpy_array = torch_tensor.detach().cpu().numpy()

    # ----- makes suren layout matches 32 bit float C-array -----
    # .astype(npfloat32) forces standard CC float precistion
    # .flatten() makes sure the memory is a 1D block
    contiguous_array = numpy_array.astype(np.float32).flatten()

    # ----- Ensure raw memory pointer -----
    c_float_ptr = contiguous_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    array_size = contiguous_array.size

    # ----- Call the C library -----
    spires_lib.spires_reservoir_step(c_float_ptr, array_size)
    return contiguous_array

    # NEED TO MODIFY SPIRES FOR THIS?? :((

def read_spikes_from_spires():
    placeholder = 0
    return 0
