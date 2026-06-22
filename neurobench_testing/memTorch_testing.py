import sys
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from torch.utils.data import DataLoader, Subset

from neurobench.models import SNNTorchModel
from neurobench.benchmarks import Benchmark, benchmark
from neurobench.datasets import SpeechCommands
from neurobench.metrics.workload import (
        ActivationSparsity,
        SynapticOperations,
        ClassificationAccuracy,
)

from neurobench.metrics.static import (
        Footprint,
        ConnectionSparsity,
)

from neurobench.processors.preprocessors import S2SPreProcessor
from neurobench.processors.postprocessors import ChooseMaxCount

import copy
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map 
from memtorch.bh.memristor import VTEAM
from memtorch.map.Input import naive_scale

beta = 0.9
device = torch.device("cpu")
spike_grad = surrogate.fast_sigmoid()
net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(20, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 256),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 35),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True),
)

vteam_params = {
        'time_series_resolution': 1e-3,
        'r_on': 50,
        'r_off': 1000,
}

net.load_state_dict(torch.load("examples/gsc/model_data/s2s_gsc_snntorch", map_location=device))

patched_net = patch_model(
        copy.deepcopy(net),
        memristor_model=VTEAM,
        memristor_model_params=vteam_params,
        module_parameters_to_patch=[torch.nn.Linear],
        mapping_routine=naive_map,
        transistor=True,
        tile_shape=(128, 128),
        max_input_voltage=0.3,
        scaling_routine=naive_scale,
        ADC_resolution=16,
        use_bindings=True,
        verbose=True,
)

static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ActivationSparsity, SynapticOperations, ClassificationAccuracy]

test_set = SpeechCommands(path="data/SpeechCommands/", subset="testing")

#shorten data set so I can actually run it lol
tiny_indices = list(range(10)) 
tiny_test_set = Subset(test_set, tiny_indices)
test_set_loader = DataLoader(tiny_test_set, batch_size=16, shuffle=True)

pre_processor = [S2SPreProcessor(device=device)]
post_processor = [ChooseMaxCount()]

model = SNNTorchModel(net)
benchmark = Benchmark(
        model,
        test_set_loader,
        pre_processor,
        post_processor,
        [static_metrics, workload_metrics]
)

print("\n --Checking signal strength--")
dummy_input = torch.randn(2, 20).to(device)

try:
    raw_signal = patched_net(dummy_input)

    if isinstance(raw_signal, tuple) and len(raw_signal) > 1:
        voltage_signal = raw_signal[0]
        print("Checking Neuron voltages")
    else:
        voltage_signal = raw_signal
        print("Checking RAW OUTPUTS")


    print(f"Signal Max: {raw_signal.max().item():.8f}")
    print(f"Signal Min: {raw_signal.min().item():.8f}")
    print(f"Signal Mean: {raw_signal.mean().item():.8f}")

except Exception as e:
    print("Error getting signal:", e)

sys.exit()

results = benchmark.run()
print("\n\n----- IDEAL BENCHMARK -----")
print(f"Footprint: {results['Footprint']}")
print(f"Connection Sparsity: {results['ConnectionSparsity']}")
print(f"Activation Sparsity: {results['ActivationSparsity']}")
print(f"Synaptic Operations: {results['SynapticOperations']}")
print(f"Classification Accuracy: {results['ClassificationAccuracy']}\n")

#energy calculations
ENERGY_PER_MAC = 0.9e-12
ENERGY_PER_AC = 0.1e-12

macs = results['SynapticOperations']['Effective_MACs']
acs = results['SynapticOperations']['Effective_ACs']

total_energy = (macs * ENERGY_PER_MAC) + (acs * ENERGY_PER_AC)

print("Energy Report:")
print(f"Total Operations: {macs} MACS, {acs} acs ")
print(f"Calculated Energy cost: {total_energy} joules per batch\n\n")


model = SNNTorchModel(patched_net)
benchmark = Benchmark(
        model,
        test_set_loader,
        pre_processor,
        post_processor,
        [static_metrics, workload_metrics]
)


with torch.no_grad(): #makes sure its in inference mode
    #otherwise you get memory leaks : ( 
    results = benchmark.run()
    
print("\n\n----- MEMRISTOR BENCHMARK -----")
print(f"Footprint: {results['Footprint']}")
print(f"Connection Sparsity: {results['ConnectionSparsity']}")
print(f"Activation Sparsity: {results['ActivationSparsity']}")
print(f"Synaptic Operations: {results['SynapticOperations']}")
print(f"Classification Accuracy: {results['ClassificationAccuracy']}")

#energy calculations
ENERGY_PER_MAC = 0.9e-12
ENERGY_PER_AC = 0.1e-12

macs = results['SynapticOperations']['Effective_MACs']
acs = results['SynapticOperations']['Effective_ACs']

total_energy = (macs * ENERGY_PER_MAC) + (acs * ENERGY_PER_AC)

print("Energy Report:")
print(f"Total Operations: {macs} MACS, {acs} acs ")
print(f"Calculated Energy cost: {total_energy} joules per batch")
