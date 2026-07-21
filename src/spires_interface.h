#ifndef SPIRES_BACKEND_H
#define SPIRES_BACKEND_H

#include <stddef.h>
#include <spires.h>

typedef struct {
    size_t num_samples;
    size_t num_features;
    double *states;
} Reservoir_State_Matrix;

int collect_reservoir_states(
    const spires_reservoir *reservoir,
    const double *input_series,
    size_t series_length,
    Reservoir_State_Matrix *result
);

void free_reservoir_state_matrix(
        Reservoir_State_Matrix *matrix
);

#endif
