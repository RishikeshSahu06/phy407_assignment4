#!/bin/bash
# Script to compile, run, and plot the MD simulation results

# Compile the code
echo "Compiling MD simulation code..."
make clean
make

# Run Anderson thermostat simulations with different collision frequencies
echo "Running Anderson thermostat simulations..."
./nvt 0 10
./nvt 0 50
./nvt 0 100

# Run Nose-Hoover thermostat simulations with different masses
echo "Running Nose-Hoover thermostat simulations..."
./nvt 1 5
./nvt 1 20
./nvt 1 100

# Run the plotting script
echo "Generating plots..."
python3 plot_results.py

echo "Simulation and plotting complete!"