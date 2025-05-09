// md_simulation.cpp
// NVT ensemble molecular dynamics simulation of Lennard-Jones fluid
// with Anderson and Nose-Hoover thermostats

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>

// Constants
constexpr double L = 20.0;           // Box size
constexpr double rho = 0.7;          // Density
constexpr double dt = 0.001;         // Base time step
constexpr double temperature = 2.0;  // Target temperature
constexpr double cutoff = 2.5;       // LJ potential cutoff
constexpr double min_dist = 0.9;     // Minimum initial particle distance
constexpr int total_steps = 500;     // Total simulation steps (in units of tau)

// Structure to hold particle data
struct Particle {
    double x, y, z;       // Position
    double vx, vy, vz;    // Velocity
    double fx, fy, fz;    // Force
};

// Main simulation class
class MDSimulation {
private:
    std::vector<Particle> particles;
    std::vector<std::vector<int>> neighbor_list;
    int N;                   // Number of particles
    double box_size;         // Box size
    double time_step;        // Time step
    double target_temp;      // Target temperature
    int steps_per_tau;       // Steps per time unit tau
    int total_steps;         // Total simulation steps

    int thermostat_type;     // 0 = Anderson, 1 = Nose-Hoover
    int collision_freq;      // Collision frequency for Anderson
    double Q;                // Effective mass for Nose-Hoover
    double xi;               // Nose-Hoover thermostat variable
    double v_xi;             // Nose-Hoover thermostat variable velocity

    // Random number generators
    std::mt19937 gen;
    std::uniform_real_distribution<double> uniform_dist;
    std::normal_distribution<double> normal_dist;

    // Data collection
    std::vector<double> potential_energy_data;
    std::vector<double> kinetic_energy_data;
    std::vector<double> total_energy_data;
    std::vector<double> temperature_data;
    std::vector<double> time_data;
    std::vector<std::vector<double>> velocity_dist_data;
    std::vector<double> msd_data;
    std::vector<double> diffusion_coeff;
    std::vector<std::vector<double>> initial_positions;

public:
    MDSimulation(int collision_frequency, int thermo_type = 0, double nose_mass = 1.0) 
        : box_size(L), target_temp(temperature), thermostat_type(thermo_type), 
          collision_freq(collision_frequency), Q(nose_mass), xi(0.0), v_xi(0.0) {
        
        // Calculate number of particles
        N = static_cast<int>(rho * pow(box_size, 3));
        time_step = dt;
        steps_per_tau = static_cast<int>(1.0 / dt);
        total_steps = ::total_steps * steps_per_tau;
        
        // Initialize random number generators
        std::random_device rd;
        gen = std::mt19937(rd());
        uniform_dist = std::uniform_real_distribution<double>(0.0, 1.0);
        normal_dist = std::normal_distribution<double>(0.0, 1.0);
        
        std::cout << "Initializing simulation with " << N << " particles" << std::endl;
        std::cout << "Thermostat: " << (thermostat_type == 0 ? "Anderson" : "Nose-Hoover") << std::endl;
        if (thermostat_type == 0) {
            std::cout << "Collision frequency: " << collision_freq << std::endl;
        } else {
            std::cout << "Nose-Hoover mass: " << Q << std::endl;
        }
    }

    // Initialize particle positions and velocities
    void initialize() {
        particles.resize(N);
        initial_positions.resize(N, std::vector<double>(3));
        neighbor_list.resize(N);
        
        // Initialize positions with minimum distance constraint
        initializePositions();
        
        // Initialize velocities
        initializeVelocities();
        
        // Remove net momentum
        removeNetMomentum();
        
        // Scale velocities to target temperature
        scaleVelocities();
        
        // Store initial positions for MSD calculation
        for (int i = 0; i < N; i++) {
            initial_positions[i][0] = particles[i].x;
            initial_positions[i][1] = particles[i].y;
            initial_positions[i][2] = particles[i].z;
        }
    }

    void initializePositions() {
        int i = 0;
        int attempts = 0;
        const int max_attempts = 1000000;
        
        while (i < N && attempts < max_attempts) {
            double x = uniform_dist(gen) * box_size;
            double y = uniform_dist(gen) * box_size;
            double z = uniform_dist(gen) * box_size;
            
            bool overlap = false;
            for (int j = 0; j < i; j++) {
                double dx = x - particles[j].x;
                double dy = y - particles[j].y;
                double dz = z - particles[j].z;
                
                // Apply periodic boundary conditions
                dx -= box_size * round(dx / box_size);
                dy -= box_size * round(dy / box_size);
                dz -= box_size * round(dz / box_size);
                
                double dist_sq = dx*dx + dy*dy + dz*dz;
                if (dist_sq < min_dist * min_dist) {
                    overlap = true;
                    break;
                }
            }
            
            if (!overlap) {
                particles[i].x = x;
                particles[i].y = y;
                particles[i].z = z;
                particles[i].fx = 0.0;
                particles[i].fy = 0.0;
                particles[i].fz = 0.0;
                i++;
            }
            
            attempts++;
        }
        
        if (attempts >= max_attempts) {
            std::cerr << "Warning: Could not place all particles without overlap. Placed " 
                      << i << " out of " << N << " particles." << std::endl;
            N = i;  // Adjust N to the actual number of particles placed
            particles.resize(N);
            initial_positions.resize(N, std::vector<double>(3));
            neighbor_list.resize(N);
        }
    }

    void initializeVelocities() {
        // Initialize velocities from either Gaussian or uniform distribution
        for (int i = 0; i < N; i++) {
            // Using Gaussian distribution with zero mean and unit variance
            particles[i].vx = normal_dist(gen);
            particles[i].vy = normal_dist(gen);
            particles[i].vz = normal_dist(gen);
            
            // Alternatively, use uniform distribution
            // particles[i].vx = uniform_dist(gen) - 0.5;
            // particles[i].vy = uniform_dist(gen) - 0.5;
            // particles[i].vz = uniform_dist(gen) - 0.5;
        }
    }

    void removeNetMomentum() {
        double px = 0.0, py = 0.0, pz = 0.0;
        
        for (int i = 0; i < N; i++) {
            px += particles[i].vx;
            py += particles[i].vy;
            pz += particles[i].vz;
        }
        
        px /= N;
        py /= N;
        pz /= N;
        
        for (int i = 0; i < N; i++) {
            particles[i].vx -= px;
            particles[i].vy -= py;
            particles[i].vz -= pz;
        }
    }

    void scaleVelocities() {
        double current_temp = calculateTemperature();
        double scale_factor = sqrt(target_temp / current_temp);
        
        for (int i = 0; i < N; i++) {
            particles[i].vx *= scale_factor;
            particles[i].vy *= scale_factor;
            particles[i].vz *= scale_factor;
        }
    }

    double calculateTemperature() {
        double kinetic = 0.0;
        
        for (int i = 0; i < N; i++) {
            kinetic += particles[i].vx * particles[i].vx;
            kinetic += particles[i].vy * particles[i].vy;
            kinetic += particles[i].vz * particles[i].vz;
        }
        
        return kinetic / (3.0 * N);
    }

    void updateNeighborList() {
        // Clear the neighbor list
        for (int i = 0; i < N; i++) {
            neighbor_list[i].clear();
        }
        
        // Build neighbor list with expanded cutoff
        double nl_cutoff = cutoff * 1.2;  // Expanded cutoff for neighbor list
        double nl_cutoff_sq = nl_cutoff * nl_cutoff;
        
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;
                double dz = particles[i].z - particles[j].z;
                
                // Apply periodic boundary conditions
                dx -= box_size * round(dx / box_size);
                dy -= box_size * round(dy / box_size);
                dz -= box_size * round(dz / box_size);
                
                double dist_sq = dx*dx + dy*dy + dz*dz;
                
                if (dist_sq < nl_cutoff_sq) {
                    neighbor_list[i].push_back(j);
                    neighbor_list[j].push_back(i);
                }
            }
        }
    }

    void calculateForces() {
        // Reset forces
        for (int i = 0; i < N; i++) {
            particles[i].fx = 0.0;
            particles[i].fy = 0.0;
            particles[i].fz = 0.0;
        }
        
        double potential = 0.0;
        double cutoff_sq = cutoff * cutoff;
        
        // Calculate forces using neighbor list
        for (int i = 0; i < N; i++) {
            for (const auto& j : neighbor_list[i]) {
                if (j <= i) continue;  // Avoid double counting
                
                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;
                double dz = particles[i].z - particles[j].z;
                
                // Apply periodic boundary conditions
                dx -= box_size * round(dx / box_size);
                dy -= box_size * round(dy / box_size);
                dz -= box_size * round(dz / box_size);
                
                double dist_sq = dx*dx + dy*dy + dz*dz;
                
                if (dist_sq < cutoff_sq) {
                    double dist_6 = dist_sq * dist_sq * dist_sq;
                    double dist_12 = dist_6 * dist_6;
                    double force = 24.0 * (2.0 / dist_12 - 1.0 / dist_6) / dist_sq;
                    
                    particles[i].fx += force * dx;
                    particles[i].fy += force * dy;
                    particles[i].fz += force * dz;
                    
                    particles[j].fx -= force * dx;
                    particles[j].fy -= force * dy;
                    particles[j].fz -= force * dz;
                    
                    potential += 4.0 * (1.0 / dist_12 - 1.0 / dist_6) - 4.0 * (1.0 / pow(cutoff, 12) - 1.0 / pow(cutoff, 6));
                }
            }
        }
        
        // Store potential energy
        potential_energy_data.push_back(potential / N);
    }

    void integrateVelocityVerlet() {
        // Update positions and half-step velocities
        for (int i = 0; i < N; i++) {
            particles[i].vx += 0.5 * time_step * particles[i].fx;
            particles[i].vy += 0.5 * time_step * particles[i].fy;
            particles[i].vz += 0.5 * time_step * particles[i].fz;
            
            particles[i].x += time_step * particles[i].vx;
            particles[i].y += time_step * particles[i].vy;
            particles[i].z += time_step * particles[i].vz;
            
            // Apply periodic boundary conditions
            particles[i].x -= box_size * floor(particles[i].x / box_size);
            particles[i].y -= box_size * floor(particles[i].y / box_size);
            particles[i].z -= box_size * floor(particles[i].z / box_size);
        }
        
        // Recalculate forces
        calculateForces();
        
        // Complete velocity update
        for (int i = 0; i < N; i++) {
            particles[i].vx += 0.5 * time_step * particles[i].fx;
            particles[i].vy += 0.5 * time_step * particles[i].fy;
            particles[i].vz += 0.5 * time_step * particles[i].fz;
        }
    }

    void applyAndersonThermostat() {
        // Anderson thermostat
        std::bernoulli_distribution collision(collision_freq * time_step);
        
        for (int i = 0; i < N; i++) {
            if (collision(gen)) {
                // Reassign velocity from Maxwell-Boltzmann distribution
                particles[i].vx = sqrt(target_temp) * normal_dist(gen);
                particles[i].vy = sqrt(target_temp) * normal_dist(gen);
                particles[i].vz = sqrt(target_temp) * normal_dist(gen);
            }
        }
    }

    void applyNoseHooverThermostat() {
        // Calculate current kinetic energy
        double kinetic = 0.0;
        for (int i = 0; i < N; i++) {
            kinetic += particles[i].vx * particles[i].vx;
            kinetic += particles[i].vy * particles[i].vy;
            kinetic += particles[i].vz * particles[i].vz;
        }
        
        // Update Nose-Hoover thermostat variables
        double G = (kinetic - 3.0 * N * target_temp) / Q;
        v_xi += 0.5 * time_step * G;
        xi += time_step * v_xi;
        
        // Scale velocities
        for (int i = 0; i < N; i++) {
            particles[i].vx *= exp(-time_step * v_xi);
            particles[i].vy *= exp(-time_step * v_xi);
            particles[i].vz *= exp(-time_step * v_xi);
        }
        
        // Update v_xi again
        kinetic = 0.0;
        for (int i = 0; i < N; i++) {
            kinetic += particles[i].vx * particles[i].vx;
            kinetic += particles[i].vy * particles[i].vy;
            kinetic += particles[i].vz * particles[i].vz;
        }
        
        G = (kinetic - 3.0 * N * target_temp) / Q;
        v_xi += 0.5 * time_step * G;
    }

    void collectData(int step) {
        if (step % steps_per_tau == 0) {
            double t = step * time_step;
            time_data.push_back(t);
            
            // Calculate temperature
            double temp = calculateTemperature();
            temperature_data.push_back(temp);
            
            // Calculate kinetic energy
            double kinetic = 0.0;
            for (int i = 0; i < N; i++) {
                kinetic += particles[i].vx * particles[i].vx;
                kinetic += particles[i].vy * particles[i].vy;
                kinetic += particles[i].vz * particles[i].vz;
            }
            kinetic *= 0.5 / N;
            kinetic_energy_data.push_back(kinetic);
            
            // Calculate total energy
            double potential = potential_energy_data.back();
            total_energy_data.push_back(kinetic + potential);
            
            // Calculate MSD
            double msd = 0.0;
            for (int i = 0; i < N; i++) {
                double dx = particles[i].x - initial_positions[i][0];
                double dy = particles[i].y - initial_positions[i][1];
                double dz = particles[i].z - initial_positions[i][2];
                
                // Apply periodic boundary conditions (unwrapped trajectories)
                // This is a simplified approach - actual unwrapping would require tracking crossings
                dx -= box_size * round(dx / box_size);
                dy -= box_size * round(dy / box_size);
                dz -= box_size * round(dz / box_size);
                
                msd += dx*dx + dy*dy + dz*dz;
            }
            msd /= N;
            msd_data.push_back(msd);
            
            // Calculate diffusion coefficient (D = MSD / 6t)
            if (t > 0) {
                diffusion_coeff.push_back(msd / (6.0 * t));
            } else {
                diffusion_coeff.push_back(0.0);
            }
            
            // Collect velocity distribution at specified times
            int tau_step = step / steps_per_tau;
            if (tau_step == 0 || tau_step == 50 || tau_step == 100) {
                std::vector<double> velocities;
                for (int i = 0; i < N; i++) {
                    velocities.push_back(particles[i].vx);
                    velocities.push_back(particles[i].vy);
                    velocities.push_back(particles[i].vz);
                }
                velocity_dist_data.push_back(velocities);
            }
        }
    }

    void run() {
        // Initialize simulation
        initialize();
        updateNeighborList();
        calculateForces();
        
        // Run simulation
        for (int step = 0; step <= total_steps; step++) {
            // Integrate equations of motion
            integrateVelocityVerlet();
            
            // Apply thermostat
            if (thermostat_type == 0) {
                applyAndersonThermostat();
            } else {
                applyNoseHooverThermostat();
            }
            
            // Update neighbor list periodically
            if (step % 10 == 0) {
                updateNeighborList();
            }
            
            // Collect data
            collectData(step);
            
            // Print progress
            if (step % (total_steps / 10) == 0) {
                std::cout << "Progress: " << 100.0 * step / total_steps << "%" << std::endl;
            }
        }
    }

    void saveData(const std::string& prefix) {
        // Save energy data
        std::ofstream energy_file(prefix + "_energy.dat");
        energy_file << "# Time Potential Kinetic Total" << std::endl;
        for (size_t i = 0; i < time_data.size(); i++) {
            energy_file << time_data[i] << " " 
                       << potential_energy_data[i] << " " 
                       << kinetic_energy_data[i] << " " 
                       << total_energy_data[i] << std::endl;
        }
        energy_file.close();
        
        // Save temperature data
        std::ofstream temp_file(prefix + "_temperature.dat");
        temp_file << "# Time Temperature" << std::endl;
        for (size_t i = 0; i < time_data.size(); i++) {
            temp_file << time_data[i] << " " << temperature_data[i] << std::endl;
        }
        temp_file.close();
        
        // Save MSD and diffusion coefficient data
        std::ofstream msd_file(prefix + "_msd.dat");
        msd_file << "# Time MSD DiffusionCoefficient" << std::endl;
        for (size_t i = 0; i < time_data.size(); i++) {
            msd_file << time_data[i] << " " << msd_data[i] << " " << diffusion_coeff[i] << std::endl;
        }
        msd_file.close();
        
        // Save velocity distribution data
        std::ofstream vel_file(prefix + "_velocity_dist.dat");
        vel_file << "# Velocity distributions at t=0, t=50, t=100" << std::endl;
        for (size_t i = 0; i < velocity_dist_data.size(); i++) {
            vel_file << "# Distribution " << i << std::endl;
            for (size_t j = 0; j < velocity_dist_data[i].size(); j++) {
                vel_file << velocity_dist_data[i][j] << std::endl;
            }
            vel_file << std::endl;
        }
        vel_file.close();
    }
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <thermostat_type> [collision_freq/Q_mass]" << std::endl;
        std::cout << "  thermostat_type: 0 = Anderson, 1 = Nose-Hoover" << std::endl;
        std::cout << "  collision_freq: Collision frequency for Anderson thermostat (default: 10)" << std::endl;
        std::cout << "  Q_mass: Effective mass for Nose-Hoover thermostat (default: 5)" << std::endl;
        return 1;
    }
    
    int thermostat_type = std::stoi(argv[1]);
    
    // Default values
    int collision_freq = 10;
    double Q_mass = 5.0;
    
    if (argc >= 3) {
        if (thermostat_type == 0) {
            collision_freq = std::stoi(argv[2]);
        } else {
            Q_mass = std::stod(argv[2]);
        }
    }
    
    std::string prefix;
    if (thermostat_type == 0) {
        prefix = "anderson_" + std::to_string(collision_freq);
        MDSimulation sim(collision_freq, 0);
        sim.run();
        sim.saveData(prefix);
    } else {
        prefix = "nosehoover_" + std::to_string(static_cast<int>(Q_mass));
        MDSimulation sim(0, 1, Q_mass);
        sim.run();
        sim.saveData(prefix);
    }
    
    return 0;
}