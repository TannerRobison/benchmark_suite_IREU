import torch 
import memtorch
from spires_interface import (
        free_spires_reservoir,
        init_spires_reservoir,
        send_currents_to_spires,
        read_spikes_from_spires,
)

NUM_INPUTS = 4
NUM_OUTPUTS = 4
NUM_NEURONS = 800

input_layer = torch.nn.Linear(NUM_INPUTS, NUM_NEURONS, bias=False)
#this is a recurrent layer sort of???
reservoir_layer = torch.nn.Linear(NUM_NEURONS, NUM_NEURONS, bias=False)
readout_layer = torch.nn.Linear(NUM_NEURONS, NUM_OUTPUTS, bias=False)

spires_reservoir = init_spires_reservoir(NUM_NEURONS)

#make the reservoir sparse and random
with torch.no_grad():
    reservoir_layer.weight.data.normal_(0.0, 0.5) #random weight

    #mask so 10% of connections exist
    mask = (torch.rand(NUM_NEURONS, NUM_NEURONS) < 0.10).float()
    reservoir_layer.weight.data *= mask

# patch layers into memristor crossbars with memTorch
memristor_model = memtorch.bh.memristor.VTEAM
memristor_model_params = {
        'time_series_resolution': 1e-3,
        'r_on': 50,
        'r_off': 1000,
}

mem_input_layer = memtorch.mn.Module.patch_model(
        input_layer, 
        memristor_model,
        memristor_model_params,
)

mem_reservoir_layer = memtorch.mn.Module.patch_model(
        reservoir_layer,
        memristor_model,
        memristor_model_params,
)

#run spiking loop with spires
previous_spikes = torch.zeros(1, NUM_NEURONS)
for step in range(500):
    cartpole_state = [0, 1, 2, 3] #this isn't final
    input_tensor = torch.tensor(cartpole_state).float().unsqueeze(0)
    currents_in = mem_input_layer(input_tensor)

    currents_recv = mem_reservoir_layer(previous_spikes)

    #may need to scale currents?
    total_currents = currents_in + currents_recv
    
    send_currents_to_spires(spires_reservoir, total_currents)
    
    current_spikes_np = read_spikes_from_spires(spires_reservoir, NUM_NEURONS)
    previous_spikes = torch.from_numpy(current_spikes_np).float().unsqueeze(0)

free_spires_reservoir(spires_reservoir)
#pass current spikes np to readout layer

