import torch
import memtorch

# 1. Standard memTorch setup (Static crossbar weights)
ann_layer = torch.nn.Linear(100, 10)
patched_layer = memtorch.mn.Module.patch_model(ann_layer, memristor_model)

# 2. Your custom SNN wrapper loop
def forward_snn(input_spikes_over_time):
    # input_spikes_over_time shape: (time_steps, batch_size, input_dim)
    time_steps = input_spikes_over_time.shape[0]
    v_mem = torch.zeros(batch_size, 10) # Hidden neuron membrane potentials
    output_spikes = []

    for t in range(time_steps):
        # Pass binary spikes through memTorch's physical crossbar simulation
        current_in = patched_layer(input_spikes_over_time[t])
        
        # Leaky Integrate-and-Fire (LIF) logic (Written by you!)
        v_mem = 0.9 * v_mem + current_in  # Leak & Integrate
        
        # Fire threshold
        spike = (v_mem >= 1.0).float()
        v_mem[v_mem >= 1.0] = 0.0         # Reset
        
        output_spikes.append(spike)
        
    return torch.stack(output_spikes)
