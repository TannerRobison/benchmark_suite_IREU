import numpy as np
import torch 
import memtorch
from spires_interface import (
        free_spires_reservoir,
        init_spires_reservoir,
        send_currents_to_spires,
        read_spikes_from_spires,
)
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import time

# ----- PARAMETERS -----
NUM_INPUTS = 1
NUM_OUTPUTS = 1
NUM_NEURONS = 800
time_steps = 2000

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
        'time_series_resolution': 1e-10,
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

start_time = time.perf_counter()

# ----- run spires & collect spikes-----
previous_spikes = torch.zeros(1, NUM_NEURONS)
spike_history = []

input_signal = []
t = np.linspace(0, 20, time_steps)

input_signal = np.sin(-2.5 * t) + np.sin(5 * t)
target_signal = np.roll(input_signal, -5) #predicting 5 time steps in the future

for step in range(time_steps):

    input_tensor = torch.tensor([input_signal[step]]).float().unsqueeze(0)

    currents_in = mem_input_layer(input_tensor)
    currents_recv = mem_reservoir_layer(previous_spikes)

    noise_multiplier = 0.08
    current_scaler = 0.05

    total_currents = (currents_in + currents_recv)
    total_currents = total_currents - total_currents.mean() # remove DC bias
    total_currents = total_currents + (torch.randn_like(total_currents) * noise_multiplier )
    total_currents = total_currents * current_scaler #scale to fit threshhold

    send_currents_to_spires(spires_reservoir, total_currents)
    current_spikes = read_spikes_from_spires(spires_reservoir, NUM_NEURONS)

    spike_history.append(current_spikes.copy())

    # convert spikes back to tensor
    previous_spikes = torch.from_numpy(current_spikes).float().unsqueeze(0)
    
#low pass filter
spike_matrix = np.array(spike_history)

decay_rate = 0.85
filtered_spikes = np.zeros_like(spike_matrix, dtype=float)

current_trace = np.zeros(NUM_NEURONS)
for i in range(len(spike_matrix)):
    current_trace = current_trace * decay_rate + spike_matrix[i]
    filtered_spikes[i] = current_trace

#training readout layer
print("training the readout layer")
x_train = filtered_spikes[100:800]
y_train = target_signal[100:800]

x_test = filtered_spikes[800:]
y_test = target_signal[800:]

ridge = Ridge(alpha=5)
ridge.fit(x_train, y_train)

#predict and plot results
print("Generating predictions...")
predictions = ridge.predict(x_test)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"execution time: {execution_time}")

test_steps = np.arange(800, time_steps)

plt.figure(figsize=(10, 5))
plt.plot(test_steps, y_test, label="True Future Wave", color="black", linestyle="dashed")
plt.plot(test_steps, predictions, label="Reservoir Prediction", color="blue", alpha=0.8)
plt.title("spires memristor Time-Series prediction")
plt.xlabel("time steps")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("memristor_spires_proof.png")

free_spires_reservoir(spires_reservoir)

# # ---------- Plotting ----------
# #This block is all AI generated to be transparent
# #plotting to verify neurons are firing randomly
#
# print("Simulation complete, plotting results") #why? cuz its fun and almost 5
# spike_matrix = np.array(spike_history)
# # Plotting
# plt.figure(figsize=(12, 6))
# # Transpose (.T) so Time is the X-axis and Neurons are the Y-axis
# plt.imshow(spike_matrix.T, aspect='auto', cmap='binary', interpolation='nearest')
#
# plt.title("Reservoir Spiking Activity (Raster Plot)")
# plt.xlabel("Time Step")
# plt.ylabel("Neuron ID (0 to 799)")
# plt.colorbar(label="Spike (0 or 1)")
# plt.tight_layout()
# plt.show()
#
