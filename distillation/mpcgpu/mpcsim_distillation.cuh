#pragma once

#include <iostream>
#include <vector>
#include <tuple>
#include <cstdint>
#include <cuda_runtime.h>

#include "settings_distillation.cuh"
#include "gpuassert.cuh"
#include "gpu_pcg.cuh"
#include "distillation_plant.cuh"

#if LINSYS_SOLVE == 1
#include "sqp_distillation.cuh"
#else
#error "distillation_mpcsim currently supports PCG only. Set LINSYS_SOLVE == 1."
#endif

template <typename T, typename return_type>
std::tuple<std::vector<return_type>, std::vector<T>, T>
simulateMPC(
    const uint32_t state_size,
    const uint32_t control_size,
    const uint32_t knot_points,
    const uint32_t traj_len,
    float timestep,
    T* d_ref_traj,
    T* d_xu_traj,
    T* d_xs,
    T* h_start_state,
    void* unused_reference_state,
    uint32_t test_iter,
    T linsys_exit_tol,
    std::string test_output_prefix
) {
    T* d_lambda = nullptr;
    T* d_xu = nullptr;

    gpuErrchk(cudaMalloc(&d_lambda, state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMalloc(&d_xu, traj_len * sizeof(T)));

    gpuErrchk(cudaMemset(d_lambda, 0, state_size * knot_points * sizeof(T)));
    gpuErrchk(cudaMemcpy(d_xu, d_xu_traj, traj_len * sizeof(T), cudaMemcpyDeviceToDevice));

    void* d_dynmem = gato_plant::initializeDynamicsConstMem<T>();

    pcg_config<T> config;
    config.pcg_block = PCG_NUM_THREADS;
    config.pcg_exit_tol = linsys_exit_tol;
    config.pcg_max_iter = PCG_MAX_ITER;

    T rho = static_cast<T>(1e-3);
    T rho_reset = static_cast<T>(1e-3);

    auto sqp_stats = sqpSolvePcg<T>(
        state_size,
        control_size,
        knot_points,
        timestep,
        d_ref_traj,
        d_lambda,
        d_xu,
        d_dynmem,
        config,
        rho,
        rho_reset
    );

    gpuErrchk(cudaMemcpy(d_xu_traj, d_xu, traj_len * sizeof(T), cudaMemcpyDeviceToDevice));

    gato_plant::freeDynamicsConstMem<T>(d_dynmem);

    gpuErrchk(cudaFree(d_lambda));
    gpuErrchk(cudaFree(d_xu));

    std::vector<return_type> dummy_return;
    std::vector<T> dummy_times;
    return std::make_tuple(dummy_return, dummy_times, rho);
}