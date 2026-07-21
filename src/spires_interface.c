#include "spires_interface.h"
#include <spires.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

int collect_reservoir_states(
        const spires_reservoir *reservoir,
        const double *input_series,
        size_t series_length,
        Reservoir_State_Matrix *result
) {
    //error checking
    //RIP

    //clear the result first 
    //
    result->num_samples = 0;
    result->num_features = 0;
    result->states = NULL;

    const size_t num_inputs = spires_num_inputs(reservoir);

    const size_t num_neurons = spires_num_neurons(reservoir);

    if (series_length > SIZE_MAX / num_neurons || series_length * num_neurons 
            > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "matrix size overloaded!!");
        return -1;
    }

    double *states = malloc(num_neurons * series_length * sizeof(*states));
    if (!states) {
        fprintf(stderr, "failed to allocate memory for states");
        return -1;
    }

    spires_status status = spires_reservoir_reset(reservoir);
    if (status != SPIRES_OK) {
        fprintf(stderr, "reservoir reset error");
        free(states);
        return -1;
    }
    
    // build state_matrix
    for (size_t i = 0; i < series_length; i++) {
        const double *current_input = &input_series[i * num_inputs];
        status = spires_step(reservoir, current_input);
        if (status != SPIRES_OK){
            free(states);
            return -1;
        }

        double *current_state = &states[i * num_neurons];
        status = spires_read_reservoir_state(reservoir, current_state);
        if (status != SPIRES_OK){
            free(states);
            return -1;
        }
    }

    result->num_samples = series_length;
    result->num_features = num_neurons;
    result->states = states;

    return 0;
}

void free_reservoir_state_matrix(Reservoir_State_Matrix *matrix) {
    if (!matrix) {
        return;
    }

    free(matrix->states);

    matrix->states = NULL;
    matrix->num_samples = 0;
    matrix->num_features = 0;
}
