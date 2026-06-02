#pragma once

#include <cstdint>
#include <cooperative_groups.h>
#include "distillation_plant.cuh"
#include "distillation_dynamics.cuh"
#include "integrator_distillation.cuh"
#include "glass.cuh"

template <typename T>
size_t get_merit_smem_size(uint32_t state_size, uint32_t control_size)
{
    return sizeof(T) * (3 * state_size + control_size);
}

// cost compute for line search
template <typename T>
__global__
void ls_distillation_compute_merit(uint32_t state_size,
                           uint32_t control_size,
                           uint32_t knot_points,
                           T *d_xs,
                           T *d_xu, 
                           T *d_ref_traj, 
                           T mu, 
                           T dt, 
                           void *d_dynMem_const, 
                           T *d_dz,
                           uint32_t alpha_multiplier, 
                           T *d_merits_out, 
                           T *d_merit_temp)
{

    const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;
    const uint32_t num_blocks = gridDim.x;

    const uint32_t states_s_controls = state_size + control_size;

    extern __shared__ T s_xux_k[];
    __shared__ int s_bound_violation;

    T Jk, ck, pointmerit;

    T alpha = -1.0 / (1 << alpha_multiplier);   // alpha sign
    T *s_temp = s_xux_k + 2 * state_size + control_size;


    for(unsigned knot = block_id; knot < knot_points; knot += num_blocks){

        // Copy x_k
        for (uint32_t i = thread_id; i < state_size; i += num_threads) {
            s_xux_k[i] = d_xu[knot * states_s_controls + i];
        }

        // Copy u_k and x_{k+1}
        if (knot < knot_points - 1) {
            for (uint32_t i = thread_id; i < control_size; i += num_threads) {
                s_xux_k[state_size + i] =
                    d_xu[knot * states_s_controls + state_size + i];
            }

            for (uint32_t i = thread_id; i < state_size; i += num_threads) {
                s_xux_k[state_size + control_size + i] =
                    d_xu[(knot + 1) * states_s_controls + i];
            }
        }

        block.sync();

        bool bound_violation = false;

        // Check x_k bounds 0 <= x <= 1
        for (uint32_t i = thread_id; i < state_size; i += num_threads) {
            T x_candidate = s_xux_k[i] + alpha * d_dz[knot * states_s_controls + i];

            if (x_candidate < static_cast<T>(0.0) ||
                x_candidate > static_cast<T>(1.0)) {
                bound_violation = true;
            }
        }

        // Check u_k bounds 1 <= u <= 5
        if (knot < knot_points - 1) {
            for (uint32_t i = thread_id; i < control_size; i += num_threads) {
                T u_candidate =
                    s_xux_k[state_size + i]
                    + alpha * d_dz[knot * states_s_controls + state_size + i];

                if (u_candidate < static_cast<T>(1.0) ||
                    u_candidate > static_cast<T>(5.0)) {
                    bound_violation = true;
                }
            }
        }

        if (thread_id == 0) {
            s_bound_violation = 0;
        }
        block.sync();

        if (bound_violation) {
            atomicExch(&s_bound_violation, 1);
        }
        block.sync();

        T xbar = d_ref_traj[6*knot + 0];
        T ubar = d_ref_traj[6*knot + 1];

        T x1 = s_xux_k[0];
        T u  = s_xux_k[state_size];

        T ex = x1 - xbar;
        T eu = u - ubar;

        Jk = ex * ex + 0.1 * eu * eu;

        block.sync();
        if(knot < knot_points-1){
            ck = distillationIntegratorError<T>(
                state_size,
                s_xux_k,
                &s_xux_k[state_size + control_size],
                s_temp,
                dt
            );
        }
        else{
            // diff xs vs xs_traj
            for (int i = threadIdx.x; i < state_size; i += blockDim.x) {
                s_temp[i] = abs((d_xu[i] + alpha *d_dz[i]) - d_xs[i]);
            }
            block.sync();
            glass::reduce<T>(state_size, s_temp);
            block.sync();
            ck = s_temp[0];
        }
        block.sync();

        // Compute the merit; reject if the point violates the variable bounds
        if(thread_id == 0){
            if (s_bound_violation) {
                pointmerit = static_cast<T>(1e20);
            }
            else {
                pointmerit = Jk + mu * ck;
            }

            d_merit_temp[alpha_multiplier * knot_points + knot] = pointmerit;
        }
    }
    cooperative_groups::this_grid().sync();
    if(block_id == 0){
        glass::reduce<T>(knot_points, &d_merit_temp[alpha_multiplier*knot_points]);
    
        if(thread_id == 0){
            d_merits_out[alpha_multiplier] = d_merit_temp[alpha_multiplier*knot_points];
        }
    }
}

// cost compute for non line search
template <typename T, unsigned INTEGRATOR_TYPE = 0, bool ANGLE_WRAP = false>
__global__
void compute_merit(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xu, T *d_ref_traj, T mu, T dt, void *d_dynMem_const, T *d_merit_out)
{
    const cooperative_groups::thread_block block = cooperative_groups::this_thread_block();
    const uint32_t thread_id = threadIdx.x;
    const uint32_t num_threads = blockDim.x;
    const uint32_t block_id = blockIdx.x;

    const uint32_t states_s_controls = state_size + control_size;
    extern __shared__ T s_xux_k[];

    T Jk, ck, pointmerit;
    T *s_temp = s_xux_k + 2 * state_size + control_size;

    for(unsigned knot = block_id; knot < knot_points; knot += gridDim.x){

        // Copy x_k
        for (uint32_t i = thread_id; i < state_size; i += num_threads) {
            s_xux_k[i] = d_xu[knot * states_s_controls + i];
        }

        // Copy u_k and x_{k+1}
        if (knot < knot_points - 1) {
            for (uint32_t i = thread_id; i < control_size; i += num_threads) {
                s_xux_k[state_size + i] =
                    d_xu[knot * states_s_controls + state_size + i];
            }

            for (uint32_t i = thread_id; i < state_size; i += num_threads) {
                s_xux_k[state_size + control_size + i] =
                    d_xu[(knot + 1) * states_s_controls + i];
            }
        }

        block.sync();

        T xbar = d_ref_traj[6*knot + 0];
        T ubar = d_ref_traj[6*knot + 1];

        T x1 = s_xux_k[0];
        T u  = s_xux_k[state_size];

        T ex = x1 - xbar;
        T eu = u - ubar;

        Jk = ex * ex + 0.1 * eu * eu;

        block.sync();
        if(knot < knot_points-1){
            ck = distillationIntegratorError<T>(
                state_size,
                s_xux_k,
                &s_xux_k[state_size + control_size],
                s_temp,
                dt
            );
        }
        else{
            ck = 0;
        }
        block.sync();

        if(thread_id == 0){
            pointmerit = Jk + mu*ck;
            atomicAdd(d_merit_out, pointmerit);
        }
    }
}
