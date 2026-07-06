# Spires reservoir simulated on memristor crossbar using memTorch
## setup
### general
python3 -m .venv venv

pip install -r requirements.txt

### memTorch

git clone --recursive https://github.com/coreylammie/MemTorch

cd memTorch

python setup.py install

### spires


## overview
### simulation
Here we are using both memTorch and the Spiresrc libraries working together to simulate
a spires reservoir on a memristor crossbar. memTorch is acting as the weights, simulating
at a physics level including device to device differences, conductance drag, etc....
Spires is acting as the actual reservoir neurons and determining the spikes.

### benchmark

## Goal
The goal is to able to simulate a spires reservoir on a memristor crossbar using
a custom memristor spice model(HIL), thus you can run the same simulation with 
multiple models to compare how each performs using benchmark results.

## task

## metrics

