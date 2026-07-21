import ctypes
import numpy
import torch
import random

LIF_DISCRETE = 0

#load the spires library
spires_lib = ctypes.CDLL("../spires/lib/libspires.so")

# C signautes for creating reservoir
spires_lib.create_reservoir.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double)
]
spires_lib.create_reservoir.restype = ctypes.c_void_p

# C signatures for reservoir destruction
spires_lib.free_reservoir.argtypes = [ctypes.c_void_p]
spires_lib.free_reservoir.restype = None

# C signatures for step reservoir function
spires_lib.update_neuron.argtypes = [
        ctypes.c_void_p,                #Pointer to spires reservoir struct
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double
]
spires_lib.update_neuron.restype = None

# C signatures for get neuron spike function
spires_lib.read_reservoir_spikes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float)
]
spires_lib.read_reservoir_spikes.restype = None

def init_spires_reservoir(reservoir_size):
    print("Creating spires reservoir")

    neuron_parameters = (ctypes.c_double * 4)(0.0, 1.0, 0.2, 0.5)
    c_neuron_parameters = ctypes.cast(neuron_parameters, ctypes.POINTER(ctypes.c_double))

    reservoir_ptr = spires_lib.create_reservoir(
            ctypes.c_size_t(int(reservoir_size)), # num neurons
            ctypes.c_size_t(int(reservoir_size)), # num neurons
            ctypes.c_size_t(2),                  # num_outputs
            ctypes.c_double(0),                # spectral radius
            ctypes.c_double(0.8),                # ei_ratio
            ctypes.c_double(1.0),                # input_strength
            ctypes.c_double(0.0),                # connectivity
            ctypes.c_double(1.0),                # dt
            ctypes.c_int(1),                  # connectivity type ( 1 = sparse)
            ctypes.c_int(LIF_DISCRETE),       # neuron type
            c_neuron_parameters # neuron params
    )
    
    #allocate empty void pointer, (reservoir will go here)
    if not reservoir_ptr: 
        raise RuntimeError(f"Spires faile to initialize the reservoir: {status_code}")
    else:
        print("Spires reservoir initialized")

    # reservoir_ptr._keep_alive = neuron_parameters

    return reservoir_ptr

def free_spires_reservoir(reservoir_ptr):
    print("Freeing the spires reservoir")
    spires_lib.free_reservoir(reservoir_ptr)
    return 0
    

#change currents from tensor to a flat C pointer array for spires to read
def send_currents_to_spires(reservoir_ptr, currents_tensor):
    # print("Sending currents to spires")
    # ----- extract from pytorhc graph -----
    # .detach() removes it from auto gradient tracking
    # .cpu() make sure data is in RAM, not VRAM
    # .numpy() maps it to numpy array
    # ----- makes suren layout matches 64 bit double C-array -----
    # .astype(npfloat64) forces standard CC float precistion
    # .flatten() makes sure the memory is a 1D block
    numpy_array = currents_tensor.detach().cpu().numpy().astype(numpy.float64).flatten()

    some_ptr = ctypes.cast(reservoir_ptr, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))

    neurons_array = some_ptr[0]

    # print("updating the neurons")
    for i in range(len(numpy_array)):
        neuron_ptr = neurons_array[i]
        input_current = numpy_array[i]

        spires_lib.update_neuron(neuron_ptr, LIF_DISCRETE, input_current, 1.0)

    return numpy_array

def read_spikes_from_spires(reservoir_ptr, size=0):
    # print("Recieved spikes from spires")
    #didnt do any safety checking womp womp
    returned_spikes = numpy.zeros(size, dtype=numpy.float64)
    c_spike_ptr = returned_spikes.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    spires_lib.read_reservoir_spikes(reservoir_ptr, c_spike_ptr)

    return returned_spikes
