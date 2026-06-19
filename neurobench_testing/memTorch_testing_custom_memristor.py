from memtorch import memristor
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from torch.utils.data import DataLoader

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

from kaevin_memristor import TEAMMemristor

beta = 0.9
class SNN(nn.Module):
    def __init__(self):
        super().__init__()
    
        #standard layers
        self.fc1 = nn.Linear(20, 128)
        self.fc2 = nn.Linear(128, 35)

        #Spiking neurons
        self.lif1 = snn.Leaky(beta=beta, init_hidden=True)
        self.lif2 = snn.Leaky(beta=beta, init_hidden=True, output=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        cur1 = self.fc1(x)
        spk1 = self.lif1(cur1)

        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2)

        return spk2, mem2

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

#memristor patch
reference_memristor = TEAMMemristor

net.load_state_dict(torch.load("examples/gsc/model_data/s2s_gsc_snntorch", map_location=device))

patched_net = patch_model(
        copy.deepcopy(net),
        memristor_model=reference_memristor,
        memristor_model_params={'time_series_resolution': 1e-8},
        mapping_routine=naive_map,
        transistor=True,
        tile_shape=(128, 128),
        ADC_resolution=8,
        use_bindings=True
)

static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ActivationSparsity, SynapticOperations, ClassificationAccuracy]

# data loader here maybe??
test_set = SpeechCommands(path="data/SpeechCommands/", subset="testing")
test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

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

results = benchmark.run()
print("\n\n----- IDEAL BENCHMARK -----")
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


model = SNNTorchModel(patched_net)
benchmark = Benchmark(
        model,
        test_set_loader,
        pre_processor,
        post_processor,
        [static_metrics, workload_metrics]
)

with torch.no_grad(): #makes sure its in inference mode
                      #otherwise you get memory leaks
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
