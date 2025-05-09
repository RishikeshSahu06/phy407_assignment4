#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys

def load_data(filename):
    """Load data from a file"""
    data = np.loadtxt(filename, skiprows=1)
    return data

def plot_energy(anderson_files, nosehoover_files):
    """Plot energy data for both thermostats"""
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot Anderson thermostat results
    for i, file in enumerate(sorted(anderson_files)):
        data = load_data(file)
        time = data[:, 0]
        potential = data[:, 1]
        kinetic = data[:, 2]
        total = data[:, 3]
        
        freq = int(file.split('_')[1].split('.')[0])
        label = f"ν = {freq}"
        
        axs[0, 0].plot(time, potential, label=label)
        axs[1, 0].plot(time, kinetic, label=label)
        axs[2, 0].plot(time, total, label=label)
    
    # Plot Nose-Hoover thermostat results
    for i, file in enumerate(sorted(nosehoover_files)):
        data = load_data(file)
        time = data[:, 0]
        potential = data[:, 1]
        kinetic = data[:, 2]
        total = data[:, 3]
        
        Q = int(file.split('_')[1].split('.')[0])
        label = f"Q = {Q}"
        
        axs[0, 1].plot(time, potential, label=label)
        axs[1, 1].plot(time, kinetic, label=label)
        axs[2, 1].plot(time, total, label=label)
    
    # Set titles and labels
    axs[0, 0].set_title("Anderson Thermostat - Potential Energy")
    axs[1, 0].set_title("Anderson Thermostat - Kinetic Energy")
    axs[2, 0].set_title("Anderson Thermostat - Total Energy")
    
    axs[0, 1].set_title("Nose-Hoover Thermostat - Potential Energy")
    axs[1, 1].set_title("Nose-Hoover Thermostat - Kinetic Energy")
    axs[2, 1].set_title("Nose-Hoover Thermostat - Total Energy")
    
    for i in range(3):
        for j in range(2):
            axs[i, j].set_xlabel("Time (τ)")
            axs[i, j].set_ylabel("Energy per particle")
            axs[i, j].legend()
            axs[i, j].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("energy_plots.png")
    plt.close()

def main():
    """Main function to execute all plotting"""
    # Find all data files
    anderson_energy_files = sorted(glob.glob("anderson_*_energy.dat"))
    nosehoover_energy_files = sorted(glob.glob("nosehoover_*_energy.dat"))
    
    anderson_temp_files = sorted(glob.glob("anderson_*_temperature.dat"))
    nosehoover_temp_files = sorted(glob.glob("nosehoover_*_temperature.dat"))
    
    anderson_msd_files = sorted(glob.glob("anderson_*_msd.dat"))
    nosehoover_msd_files = sorted(glob.glob("nosehoover_*_msd.dat"))
    
    anderson_vel_files = sorted(glob.glob("anderson_*_velocity_dist.dat"))
    nosehoover_vel_files = sorted(glob.glob("nosehoover_*_velocity_dist.dat"))
    
    # Check if files exist
    if not anderson_energy_files or not nosehoover_energy_files:
        print("Error: No data files found. Run the simulation first.")
        return
    
    print("Plotting energy data...")
    plot_energy(anderson_energy_files, nosehoover_energy_files)
    
    print("Plotting temperature data...")
    plot_temperature(anderson_temp_files, nosehoover_temp_files)
    
    print("Plotting velocity distribution data...")
    plot_velocity_dist(anderson_vel_files, nosehoover_vel_files)
    
    print("Plotting MSD and diffusion coefficient data...")
    plot_msd(anderson_msd_files, nosehoover_msd_files)
    
    print("All plots have been saved!")

if __name__ == "__main__":
    main()

def plot_msd(anderson_files, nosehoover_files):
    """Plot mean squared displacement and diffusion coefficient"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot Anderson thermostat results - MSD
    for i, file in enumerate(sorted(anderson_files)):
        data = load_data(file)
        time = data[:, 0]
        msd = data[:, 1]
        diff_coeff = data[:, 2]
        
        freq = int(file.split('_')[1].split('.')[0])
        label = f"ν = {freq}"
        
        # Normal plot
        axs[0, 0].plot(time, msd, label=label)
        
        # Log-log plot for MSD
        valid_idx = (time > 0) & (msd > 0)
        axs[0, 1].loglog(time[valid_idx], msd[valid_idx], label=label)
        
        # Calculate and plot linear fit to extract exponent (for the first collision frequency)
        if i == 0:
            log_time = np.log(time[valid_idx])
            log_msd = np.log(msd[valid_idx])
            
            # Early time fit (first 20% of data)
            early_idx = int(len(log_time) * 0.2)
            early_fit = np.polyfit(log_time[:early_idx], log_msd[:early_idx], 1)
            early_alpha = early_fit[0]
            early_fit_line = np.exp(early_fit[1]) * time[valid_idx]**early_alpha
            axs[0, 1].plot(time[valid_idx], early_fit_line, 'k--', 
                         label=f"Early α ≈ {early_alpha:.2f}")
            
            # Late time fit (last 30% of data)
            late_idx = int(len(log_time) * 0.7)
            late_fit = np.polyfit(log_time[late_idx:], log_msd[late_idx:], 1)
            late_alpha = late_fit[0]
            late_fit_line = np.exp(late_fit[1]) * time[valid_idx]**late_alpha
            axs[0, 1].plot(time[valid_idx], late_fit_line, 'r--', 
                         label=f"Late α ≈ {late_alpha:.2f}")
        
        # Plot diffusion coefficient
        axs[1, 0].plot(time, diff_coeff, label=label)
    
    # Plot Nose-Hoover thermostat results - MSD
    for i, file in enumerate(sorted(nosehoover_files)):
        data = load_data(file)
        time = data[:, 0]
        msd = data[:, 1]
        diff_coeff = data[:, 2]
        
        Q = int(file.split('_')[1].split('.')[0])
        label = f"Q = {Q}"
        
        # Normal plot for Nose-Hoover
        axs[1, 1].plot(time, msd, label=label)
        
        # Plot diffusion coefficient
        axs[1, 0].plot(time, diff_coeff, label=label, linestyle='--')
    
    # Set titles and labels
    axs[0, 0].set_title("Anderson - Mean Squared Displacement")
    axs[0, 1].set_title("Anderson - MSD (Log-Log Scale)")
    axs[1, 0].set_title("Diffusion Coefficient")
    axs[1, 1].set_title("Nose-Hoover - Mean Squared Displacement")
    
    for i in range(2):
        for j in range(2):
            axs[i, j].set_xlabel("Time (τ)")
            axs[i, j].legend()
            axs[i, j].grid(True, linestyle='--', alpha=0.7)
    
    axs[0, 0].set_ylabel("MSD")
    axs[0, 1].set_ylabel("MSD (log scale)")
    axs[1, 0].set_ylabel("Diffusion Coefficient (D)")
    axs[1, 1].set_ylabel("MSD")
    
    plt.tight_layout()
    plt.savefig("msd_plots.png")
    plt.close()

def plot_temperature(anderson_files, nosehoover_files):
    """Plot temperature data for both thermostats"""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot Anderson thermostat results
    for i, file in enumerate(sorted(anderson_files)):
        data = load_data(file)
        time = data[:, 0]
        temp = data[:, 1]
        
        freq = int(file.split('_')[1].split('.')[0])
        label = f"ν = {freq}"
        
        axs[0].plot(time, temp, label=label)
    
    # Plot Nose-Hoover thermostat results
    for i, file in enumerate(sorted(nosehoover_files)):
        data = load_data(file)
        time = data[:, 0]
        temp = data[:, 1]
        
        Q = int(file.split('_')[1].split('.')[0])
        label = f"Q = {Q}"
        
        axs[1].plot(time, temp, label=label)
    
    # Set titles and labels
    axs[0].set_title("Anderson Thermostat - Temperature")
    axs[1].set_title("Nose-Hoover Thermostat - Temperature")
    
    for i in range(2):
        axs[i].set_xlabel("Time (τ)")
        axs[i].set_ylabel("Temperature (T)")
        axs[i].axhline(y=2.0, color='r', linestyle='--', label="Target T=2.0")
        axs[i].legend()
        axs[i].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("temperature_plots.png")
    plt.close()

def plot_velocity_dist(anderson_files, nosehoover_files):
    """Plot velocity distribution for both thermostats"""
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Function to calculate Maxwell-Boltzmann distribution
    def maxwell_boltzmann(v, T=2.0):
        """Maxwell-Boltzmann distribution"""
        return (1.0/(2.0*np.pi*T)**(3.0/2.0)) * np.exp(-v**2/(2.0*T))
    
    # Time points to plot
    time_points = [0, 50, 100]
    
    # Plot Anderson thermostat results
    for i, file in enumerate(sorted(anderson_files)):
        data = np.loadtxt(file, skiprows=2)
        
        # Reshape data to get velocity distributions at different times
        num_particles = data.shape[0] // 3 // len(time_points)
        
        for t_idx, t in enumerate(time_points):
            start_idx = t_idx * num_particles * 3
            end_idx = start_idx + num_particles * 3
            velocities = data[start_idx:end_idx]
            
            # Create histogram
            hist, bins = np.histogram(velocities, bins=30, range=(-3, 3), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            freq = int(file.split('_')[1].split('.')[0])
            label = f"ν = {freq}"
            
            axs[0, t_idx].plot(bin_centers, hist, label=label)
            
            # Plot theoretical Maxwell-Boltzmann distribution
            v_range = np.linspace(-3, 3, 100)
            mb_dist = maxwell_boltzmann(v_range)
            if t_idx == 0 and i == 0:  # Only plot once
                axs[0, t_idx].plot(v_range, mb_dist, 'k--', label='Maxwell-Boltzmann')
    
    # Plot Nose-Hoover thermostat results
    for i, file in enumerate(sorted(nosehoover_files)):
        data = np.loadtxt(file, skiprows=2)
        
        # Reshape data to get velocity distributions at different times
        num_particles = data.shape[0] // 3 // len(time_points)
        
        for t_idx, t in enumerate(time_points):
            start_idx = t_idx * num_particles * 3
            end_idx = start_idx + num_particles * 3
            velocities = data[start_idx:end_idx]
            
            # Create histogram
            hist, bins = np.histogram(velocities, bins=30, range=(-3, 3), density=True)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            Q = int(file.split('_')[1].split('.')[0])
            label = f"Q = {Q}"
            
            axs[1, t_idx].plot(bin_centers, hist, label=label)
            
            # Plot theoretical Maxwell-Boltzmann distribution
            v_range = np.linspace(-3, 3, 100)
            mb_dist = maxwell_boltzmann(v_range)
            if t_idx == 0 and i == 0:  # Only plot once
                axs[1, t_idx].plot(v_range, mb_dist, 'k--', label='Maxwell-Boltzmann')
    
    # Set titles and labels
    for t_idx, t in enumerate(time_points):
        axs[0, t_idx].set_title(f"Anderson - t = {t}τ")
        axs[1, t_idx].set_title(f"Nose-Hoover - t = {t}τ")
        
        for i in range(2):
            axs[i, t_idx].set_xlabel("Velocity")
            axs[i, t_idx].set_ylabel("Probability Density")
            axs[i, t_idx].legend()
            axs[i, t_idx].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("velocity_distribution_plots.png")
    plt.close()