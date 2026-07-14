from spicelib import RawRead
import matplotlib.pyplot as plt

raw = RawRead("simulation.raw")
print(raw.get_trace_names())

time = raw.get_trace("time").get_wave()
current = raw.get_trace("i(v.xmem.v_emem)").get_wave()
voltage = raw.get_trace("v(in)").get_wave()
state_variable = raw.get_trace("v(xmem.x)").get_wave()

print(f"final current: {current[-1]}")

print(f"state variable : {state_variable[-1]}")

# Create a figure with two stacked subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# 1. Plot Current vs Time
ax1.plot(time, current, color='blue', linewidth=2, label='Memristor Current')
ax1.set_ylabel('Current (A)')
ax1.set_title('Memristor Simulation Results')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# 2. Plot Voltage vs Time
ax2.plot(time, voltage, color='red', linewidth=2, label='Input Voltage v(in)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Voltage (V)')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

# Adjust layout to prevent overlap and display the plot
plt.tight_layout()
plt.savefig("memristor_waveforms.png", dpi=300, bbox_inches='tight')
print("Saved waveforms to memristor_waveforms.png")
