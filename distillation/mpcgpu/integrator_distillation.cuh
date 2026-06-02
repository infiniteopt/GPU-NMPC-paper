#pragma once
#include <cooperative_groups.h>
#include <algorithm>
#include <cmath>

namespace cgrps = cooperative_groups;
#include "distillation_dynamics.cuh"

#include "glass.cuh"

template <typename T>
__device__ inline T distillationIntegratorError(
    uint32_t state_size,
    const T* xuk,
    const T* xnext,
    T* temp,
    T dt
) {
    T* rhs = temp;

    const T* xk = xuk;
    const T* uk = xuk + state_size;

    distill_rhs<T>(xk, uk, rhs);

    T err = static_cast<T>(0);

    for (uint32_t i = 0; i < state_size; ++i) {
        T defect = xnext[i] - xk[i] - dt * rhs[i];
        err += defect * defect;
    }

    return err;
}

template <typename T>
__device__ inline void integratorAndGradientDistillation(
    uint32_t state_size,
    uint32_t control_size,
    const T* xuk,
    const T* xnext,
    T* defect,
    T* A,
    T* B,
    T* temp,
    T dt
) {
    const T* x = xuk;
    const T* u = xuk + state_size;

    T* rhs = temp;

    // semi-implicit / backward Euler:
    // defect = xnext - x - dt*f(xnext, u)
    distill_rhs<T>(xnext, u, rhs);
    __syncthreads();

    const uint32_t tid = threadIdx.x;
    const uint32_t nt  = blockDim.x;

    for (uint32_t i = tid; i < state_size; i += nt) {
        defect[i] = xnext[i] - x[i] - dt * rhs[i];
    }

    // After computing defect
    const T eps = static_cast<T>(1e-4);

    // Zero A, B first
    for (uint32_t i = tid; i < state_size * state_size; i += nt) A[i] = static_cast<T>(0);
    for (uint32_t i = tid; i < state_size * control_size; i += nt) B[i] = static_cast<T>(0);
    __syncthreads();

    if (tid == 0) {
        T xp[32], xm[32], up[1], um[1];
        T fp[32], fm[32];

        // A = I + dt * df/dx
        for (uint32_t j = 0; j < state_size; ++j) {
            for (uint32_t i = 0; i < state_size; ++i) {
                xp[i] = xnext[i];
                xm[i] = xnext[i];
            }

            xp[j] += eps;
            xm[j] -= eps;

            distill_rhs<T>(xp, u, fp);
            distill_rhs<T>(xm, u, fm);

            for (uint32_t i = 0; i < state_size; ++i) {
                A[i * state_size + j] =
                    (i == j ? static_cast<T>(1) : static_cast<T>(0))
                    + dt * (fp[i] - fm[i]) / (static_cast<T>(2) * eps);
            }
        }

        // B = dt * df/du
        up[0] = u[0] + eps;
        um[0] = u[0] - eps;

        distill_rhs<T>(xnext, up, fp);
        distill_rhs<T>(xnext, um, fm);

        for (uint32_t i = 0; i < state_size; ++i) {
            B[i] = dt * (fp[i] - fm[i]) / (static_cast<T>(2) * eps);
        }
    }
    __syncthreads();
}

template <typename T>
void just_shift(uint32_t state_size, uint32_t control_size, uint32_t knot_points, T *d_xu){
    for (uint32_t knot = 0; knot < knot_points-1; knot++){
        uint32_t stepsize = (state_size+(knot<knot_points-2)*control_size);
        gpuErrchk(cudaMemcpy(&d_xu[knot*(state_size+control_size)], &d_xu[(knot+1)*(state_size+control_size)], stepsize*sizeof(T), cudaMemcpyDeviceToDevice));
    }
}