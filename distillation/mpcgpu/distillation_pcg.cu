#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <limits>
#include <chrono>

#include "settings_distillation.cuh"
#include "mpcsim_distillation.cuh"
#include "gpu_pcg.cuh"

#include "distillation_plant.cuh"

template <typename T>
void constant_initial_guess(
    std::vector<T>& xu_traj,
    T x_guess,
    T u_guess,
    uint32_t state_size,
    uint32_t knot_points
) {
    uint32_t offset = 0;

    for (uint32_t k = 0; k < knot_points; ++k) {
        for (uint32_t i = 0; i < state_size; ++i) {
            xu_traj[offset++] = x_guess;
        }

        if (k < knot_points - 1) {
            xu_traj[offset++] = u_guess;
        }
    }
}

template <typename T>
void shift_trajectory_warmstart(
    const std::vector<T>& x_current,
    std::vector<T>& xu_traj,
    uint32_t state_size,
    uint32_t control_size,
    uint32_t knot_points
) {
    const uint32_t stride = state_size + control_size;
    std::vector<T> old = xu_traj;

    // new x0 = measured/simulated current state
    for (uint32_t i = 0; i < state_size; ++i) {
        xu_traj[i] = x_current[i];
    }

    // new u_k = old u_{k+1}; last control repeats old last control
    for (uint32_t k = 0; k < knot_points - 1; ++k) {
        uint32_t new_idx = k * stride;

        if (k < knot_points - 2) {
            uint32_t old_idx = (k + 1) * stride;
            xu_traj[new_idx + state_size] = old[old_idx + state_size];
        } else {
            uint32_t old_last_control_idx = (knot_points - 2) * stride + state_size;
            xu_traj[new_idx + state_size] = old[old_last_control_idx];
        }
    }

    // new x_k = old x_{k+1} for k = 1,...,N-1
    for (uint32_t k = 1; k < knot_points; ++k) {
        uint32_t new_idx = k * stride;
        uint32_t old_idx = k * stride;

        for (uint32_t i = 0; i < state_size; ++i) {
            xu_traj[new_idx + i] = old[old_idx + i];
        }
    }
}

int main() {
    using T = float;   // or double, depending on repo settings
    
    // Set NMPC parameters
    constexpr T t0 = static_cast<T>(0.0);
    constexpr T tf = static_cast<T>(300.0);
    constexpr T dt_mpc = static_cast<T>(2.0);
    constexpr uint32_t nmpc_steps = static_cast<uint32_t>((tf - t0) / dt_mpc) + 1;

    // Set problem dimensions and other constants
    constexpr uint32_t state_size = 32;
    constexpr uint32_t control_size = 1;
    constexpr T xbar = static_cast<T>(0.8958);
    constexpr T ubar = static_cast<T>(2.51459);
    constexpr T u_init = static_cast<T>(2.51459);
    constexpr T xf   = static_cast<T>(0.5);
    double total_nmpc_solve_time_s = 0.0;   // For timing purposes

    // Tp = 180, dt = 2
    constexpr uint32_t knot_points = 91;   // 90 intervals + initial point
    constexpr T timestep = static_cast<T>(2.0);

    constexpr uint32_t traj_length =
        (state_size + control_size) * knot_points - control_size;

    constexpr int test_iter = 1;
    constexpr T pcg_exit_tol = static_cast<T>(1e-8);

    // Create CSV file for NMPC results
    std::ofstream nmpc_file("results/MPCGPU/distillation_mpcgpu_nmpc.csv");
    nmpc_file << "iter,t_min,solve_time_s,u";
    for (uint32_t i = 0; i < state_size; ++i) {
        nmpc_file << ",x" << (i + 1);
    }

    for (uint32_t i = 0; i < state_size; ++i) {
        nmpc_file << ",y" << (i + 1);
    }

    nmpc_file << "\n";

    std::string output_prefix = "results/MPCGPU/distillation_pcg";

    std::cout << "Running distillation PCG example\n";
    std::cout << "state_size   = " << state_size << "\n";
    std::cout << "control_size = " << control_size << "\n";
    std::cout << "knot_points  = " << knot_points << "\n";
    std::cout << "traj_length  = " << traj_length << "\n";

    // Initial state: x_i = xf = 0.5
    std::vector<T> h_xs(state_size, xf);

    // Initial trajectory guess:
    // layout assumed from MPCGPU:
    // [x0, u0, x1, u1, ..., x89, u89, x90]
    std::vector<T> h_xu_traj(traj_length, static_cast<T>(0.0));
    std::vector<T> dx(state_size);

    constant_initial_guess(
        h_xu_traj,
        xf,
        u_init,
        state_size,
        knot_points
    );

    T* d_xs = nullptr;
    T* d_xu_traj = nullptr;

    cudaMalloc(&d_xs, state_size * sizeof(T));
    cudaMalloc(&d_xu_traj, traj_length * sizeof(T));

    cudaMemcpy(
        d_xs,
        h_xs.data(),
        state_size * sizeof(T),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_xu_traj,
        h_xu_traj.data(),
        traj_length * sizeof(T),
        cudaMemcpyHostToDevice
    );

    T* d_ref_traj = nullptr;

    // Reference trajectory padded to 6 because simulateMPC expects 6 * knot_points
    std::vector<T> h_ref_traj(6 * knot_points, static_cast<T>(0));

    for (uint32_t k = 0; k < knot_points; ++k) {
        h_ref_traj[6 * k + 0] = xbar;
        h_ref_traj[6 * k + 1] = ubar;
    }

    cudaMalloc(&d_ref_traj, h_ref_traj.size() * sizeof(T));

    cudaMemcpy(
        d_ref_traj,
        h_ref_traj.data(),
        h_ref_traj.size() * sizeof(T),
        cudaMemcpyHostToDevice
    );

    // Main NMPC loop
    for (uint32_t iter = 0; iter < nmpc_steps; ++iter) {
        auto solve_start = std::chrono::high_resolution_clock::now();

        auto result = simulateMPC<T, T>(
            state_size,
            control_size,
            knot_points,
            traj_length,
            timestep,
            d_ref_traj,
            d_xu_traj,
            d_xs,
            h_xs.data(),
            nullptr,
            test_iter,
            pcg_exit_tol,
            output_prefix
        );
        cudaDeviceSynchronize();

        auto solve_end = std::chrono::high_resolution_clock::now();

        // Calculate solve time for this NMPC iteration
        double solve_s =
            std::chrono::duration<double>(solve_end - solve_start).count();

        // Accumulate total solve time
        total_nmpc_solve_time_s += solve_s;
        std::cout << "OCP solve time: " << solve_s << " s\n";

        cudaMemcpy(
            h_xu_traj.data(),
            d_xu_traj,
            traj_length * sizeof(T),
            cudaMemcpyDeviceToHost
        );

        // Extract first optimal control input
        T u_apply = h_xu_traj[state_size];

        // Simulate plant forward one MPC step
        T u_arr[1] = {u_apply};

        const uint32_t n_substeps = 20;
        const T h = timestep / static_cast<T>(n_substeps);

        for (uint32_t sub = 0; sub < n_substeps; ++sub) {
            distill_rhs_host<T>(h_xs.data(), u_arr, dx.data());
            for (uint32_t i = 0; i < state_size; ++i) {
                h_xs[i] += h * dx[i];
            }
        }

        // Log NMPC results
        const T alpha = static_cast<T>(1.6);
        T y32 =
            alpha * h_xs[31] /
            (static_cast<T>(1.0) + (alpha - static_cast<T>(1.0)) * h_xs[31]);

        nmpc_file << iter << ","
            << (iter * timestep / static_cast<T>(60.0)) << ","
            << solve_s << ","
            << u_apply;

        for (uint32_t i = 0; i < state_size; ++i) {
            nmpc_file << "," << h_xs[i];
        }

        for (uint32_t i = 0; i < state_size; ++i) {
            T yi =
                alpha * h_xs[i] /
                (static_cast<T>(1.0) + (alpha - static_cast<T>(1.0)) * h_xs[i]);

            nmpc_file << "," << yi;
        }

        nmpc_file << "\n";

        // Update initial condition on GPU
        cudaMemcpy(
            d_xs,
            h_xs.data(),
            state_size * sizeof(T),
            cudaMemcpyHostToDevice
        );

        // Warmstart by shifting OCP trajectory
        shift_trajectory_warmstart(
            h_xs,
            h_xu_traj,
            state_size,
            control_size,
            knot_points
        );

        cudaMemcpy(
            d_xu_traj,
            h_xu_traj.data(),
            traj_length * sizeof(T),
            cudaMemcpyHostToDevice
        );
    }

    // Print total NMPC problem time
    std::cout << "\nTotal NMPC solve time = "
        << total_nmpc_solve_time_s
        << " s" << std::endl;

    cudaFree(d_ref_traj);
    cudaFree(d_xs);
    cudaFree(d_xu_traj);

    return 0;
}