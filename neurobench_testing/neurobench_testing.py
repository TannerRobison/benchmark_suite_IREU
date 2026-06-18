import torch
from torch.utils.data import DataLoader

from neurobench.datasets import SpeechCommands
from neurobench.processors.preprocessors import S2SPreProcessor
from neurobench.processors.postprocessors import ChooseMaxCount
from neurobench.models import NeuroBenchModel

from neurobench.models import SNNTorchModel

from neurobench.metrics.workload import (
        ActivationSparsity,
        SynapticOperations,
        ClassificationAccuracy,
)

from neurobench.metrics.static import (
        Footprint,
        ConnectionSparsity,
)

from neurobench.benchmarks import Benchmark

from torch import nn
import snntorch as snn 
from snntorch import surrogate 

beta = 0.5 #this does nothing with the current model
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

test_set = SpeechCommands(path="data/SpeechCommands/", subset="testing")
test_set_loader = DataLoader(test_set, batch_size=500, shuffle=True)

net.load_state_dict(torch.load("examples/gsc/model_data/s2s_gsc_snntorch", map_location=device))
print("weight before noise: ", net[1].weight[0][0].item())

#adds noise for 'simulating' memristors
# not perfect ik but just testing it out
noise = 0.05
with torch.no_grad():
    for param in net.parameters():
        param.add_(torch.randn_like(param) * noise)

print("weight after noise: ", net[1].weight[0][0].item())

model = SNNTorchModel(net)

preprocessors = [S2SPreProcessor(device=device)]
postprocessors = [ChooseMaxCount()]

static_metrics = [Footprint, ConnectionSparsity]
workload_metrics = [ClassificationAccuracy, ActivationSparsity, SynapticOperations]

print("Layer 1 Beta:", net[2].beta)
print("Layer 2 Beta:", net[4].beta)

benchmark = Benchmark(
    model, 
    test_set_loader, 
    preprocessors, 
    postprocessors, 
    [static_metrics, workload_metrics]
)
results = benchmark.run()
print(results)

# Energy calulations
ENERGY_PER_MAC = 0.9e-12
ENERGY_PER_AC = 0.1e-12

macs = results['SynapticOperations']['Effective_MACs']
acs = results['SynapticOperations']['Effective_ACs']

total_energy = (macs * ENERGY_PER_MAC) + (acs * ENERGY_PER_AC)

print("Energy Report:")
print(f"Total Operations: {macs} MACS, {acs} acs ")
print(f"Calculated Energy cost: {total_energy} joules per batch")
