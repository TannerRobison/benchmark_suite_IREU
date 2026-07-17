#include <stdio.h>
#include "crossbar_generator.h"

int main() {

    double voltages[] = {1.2, 3.3, 5.0, 4.3, 2.1};

    double resistances[] = {
        1000, 1000, 3000, 5000, 4000,
        1000, 1000, 3000, 5000, 4000,
        1000, 1000, 3000, 5000, 4000,
        1000, 1000, 3000, 5000, 4000,
        1000, 1000, 3000, 5000, 4000,
    }; 

    const Crossbar_Config config = {
       .rows = 5,
       .columns = 5,
       .input_voltages = voltages,
       .initial_resistance = resistances,
       .model_path = "hp_memristor.cir",
       .subcircuit_name = "memristor",
       .load_resistance = 50.0,
       .time_step = 1e-6, //1 microsecond
       .stop_time = 1e-3, //1 millisecond
       .print_state_nodes = 0 //false
    };

    generate_crossbar("crossbar.cir", &config);

    return 0;
}
