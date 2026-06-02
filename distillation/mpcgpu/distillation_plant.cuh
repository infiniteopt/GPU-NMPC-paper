#pragma once

#include "distillation_dynamics.cuh"

namespace gato_plant {

template <typename T>
void* initializeDynamicsConstMem() {
    return nullptr;
}

template <typename T>
void freeDynamicsConstMem(void* d_dynmem) {
    // no-op
}

__host__ __device__ inline int forwardDynamics_TempMemSize_Shared() {
    return 0;
}

__host__ __device__ inline int forwardDynamicsAndGradient_TempMemSize_Shared() {
    return 0;
}

template <typename T>
__device__ inline void forwardDynamics(
    T* qdd,
    const T* q,
    const T* qd,
    const T* u,
    T* temp,
    void* d_dynmem,
    int block = 0
) {
    distill_rhs<T>(q, u, qdd);
}

template <typename T>
__device__ inline void forwardDynamicsAndGradient(
    T* dqdd,
    T* qdd,
    const T* q,
    const T* qd,
    const T* u,
    T* temp,
    void* d_dynmem
) {
    distill_rhs<T>(q, u, qdd);

    for (int i = 0; i < DISTILL_N * (DISTILL_N + 1); ++i) {
        dqdd[i] = static_cast<T>(0);
    }
}

template <typename T>
__device__ inline void trackingCostGradientAndHessian(
    uint32_t state_size,
    uint32_t control_size,
    const T* xuk,
    const T* ref,
    T* Qk,
    T* qk,
    T* Rk,
    T* rk,
    T* temp,
    void* d_dynmem
) {
    const T xbar = ref[0];
    const T ubar = ref[1];

    const T x1 = xuk[0];
    const T u  = xuk[state_size];

    for (uint32_t i = 0; i < state_size * state_size; ++i) Qk[i] = static_cast<T>(0);
    for (uint32_t i = 0; i < state_size; ++i) qk[i] = static_cast<T>(0);
    for (uint32_t i = 0; i < control_size * control_size; ++i) Rk[i] = static_cast<T>(0);
    for (uint32_t i = 0; i < control_size; ++i) rk[i] = static_cast<T>(0);

    qk[0] = static_cast<T>(2.0) * (x1 - xbar);
    Qk[0] = static_cast<T>(2.0);

    rk[0] = static_cast<T>(2.0) * (u - ubar);
    Rk[0] = static_cast<T>(2.0);
}

template <typename T>
__device__ inline void trackingCostGradientAndHessian_lastblock(
    uint32_t state_size,
    uint32_t control_size,
    const T* xk,
    const T* ref,
    T* Qk,
    T* qk,
    T* Rk,
    T* rk,
    T* Qkp1,
    T* qkp1,
    T* temp,
    void* d_dynmem
) {
    const T xbar = ref[0];
    const T x1 = xk[0];

    for (uint32_t i = 0; i < state_size * state_size; ++i) Qkp1[i] = static_cast<T>(0);
    for (uint32_t i = 0; i < state_size; ++i) qkp1[i] = static_cast<T>(0);

    // Also zero these in case kkt.cuh expects them touched.
    for (uint32_t i = 0; i < state_size * state_size; ++i) Qk[i] = static_cast<T>(0);
    for (uint32_t i = 0; i < state_size; ++i) qk[i] = static_cast<T>(0);
    for (uint32_t i = 0; i < control_size * control_size; ++i) Rk[i] = static_cast<T>(0);
    for (uint32_t i = 0; i < control_size; ++i) rk[i] = static_cast<T>(0);

    qkp1[0] = static_cast<T>(2.0) * (x1 - xbar);
    Qkp1[0] = static_cast<T>(2.0);
}
} // namespace gato_plant