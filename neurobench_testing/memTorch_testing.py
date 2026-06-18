from memtorch import memristor
import torch
import torch.nn as nn
import snntorch as snn

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

beta = 0.9

class SimpleSNN(nn.Module):
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
net = SimpleSNN().to(device)

#memristor patch
reference_memristor = VTEAM()

print("Patching model to memristive crossbar")
patched_net = patch_model(
        copy.deepcopy(net),
        memristor_model=reference_memristor,
        memristor_model_params={},
        mapping_routine=naive_map,
        transistor=True,
        ADC_resolution=8,
        use_bindings=False
)

model = SNNTorchModel(patched_net)

static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ActivationSparsity, SynapticOperations, ClassificationAccuracy]

# data loader here maybe??
test_set = SpeechCommands(path="data/SpeechCommands/", subset="testing")
test_set_loader = DataLoader(test_set, batch_size=50, shuffle=True)

pre_processor = [S2SPreProcessor(device=device)]
post_processor = [ChooseMaxCount()]

# print("Layer 1 Beta:", net[2].beta)

benchmark = Benchmark(
        model,
        test_set_loader,
        pre_processor,
        post_processor,
        [static_metrics, workload_metrics]
)

results = benchmark.run()
print(results)

#energy calculations
ENERGY_PER_MAC = 0.9e-12
ENERGY_PER_AC = 0.1e-12

macs = results['SynapticOperations']['Effective_MACs']
acs = results['SynapticOperations']['Effective_ACs']

total_energy = (macs * ENERGY_PER_MAC) + (acs * ENERGY_PER_AC)

print("Energy Report:")
print(f"Total Operations: {macs} MACS, {acs} acs ")
print(f"Calculated Energy cost: {total_energy} joules per batch")
