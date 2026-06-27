# Spires reservoir simulated on memristor crossbar using memTorch

## overview
### simulation
Here we are using both memTorch and the Spiresrc libraries working together to simulate
a spires reservoir on a memristor crossbar. memTorch is acting as the weights, simulating
at a physics level including device to device differences, conductance drag, etc....
Spires is acting as the actual reservoir neurons and determining the spikes.

### benchmark

## setup


## Goal
The goal is to able to simulate a spires reservoir on a memristor crossbar using
a custom memristor spice model(HIL), thus you can run multiple simulations with 
multiple models to compare how each performs using benchmark results.

## task
The plan right now is to give the cart-pole task for benchmarking.

## metrics

