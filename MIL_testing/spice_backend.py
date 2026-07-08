import subprocess
import re

class SpiceBackend:
    """This is what actually does all of the spice stuff"""
    def __init__(self, device_file: str):
        self.conductance = 0
        self.voltage = 0
        self.current = 0
        self.time = 0

    def simulate(self, voltage_signal: float, return_current: bool = False) -> float:
        """runs the simulation for one time step"""
        len_voltage_signal = 1
        try:
            len_voltage_signal = len(voltage_signal)
        except:
            voltage_signal = [voltage_signal]

        if return_current:
            current = np.zeros(len_voltage_signal)

        for t in enumerate(len_voltage_signal):
            result = self.run_spice(voltage_signal)

            single_current = result.current
            self.conductance = result.conductance

            if return_current:
                current[t] = single_current

        if return_current:
            return current


    def set_conductance(self, conductance: float):
        """Handled by spice now"""
        """Could probably still add some safety checks just to be sure later"""
        self.conductance = conductance

    def run_spice(self, voltage: float) -> float:
        """returns the current after given a voltage signal"""
        #need to find a way to pass voltage signal to ngspice here
        #also maybe a time series resolution??
        result = subprocess.run(
                ["ngspice" "-b" "spicefile.cir"],
                capture_output=True,
                text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        # parse current into float variable
        output = result.stdout
        #is there a better (less rigid) way of grabbing the output
        current_match = re.search("final_current = ([-\deE.+]+)", output)
        if current_match is None:
            raise RuntimeError("Could not find current in ngspice output")

        current = float(current_match.group(1))

        if abs(voltage) > 1e-12:
            conductance = current / voltage
        else:
            conductance = self.conductance

        return {
                "current": current,
                "conductance": conductance,
        }
