#pragma once

#include <cmath>

#pragma once

constexpr int DISTILL_N = 32;
constexpr int DISTILL_FT = 17;

template <typename T>
__host__ __device__ inline T distill_vapor_y(T x) {
    const T alpha = static_cast<T>(1.6);
    return alpha * x / (static_cast<T>(1.0) + (alpha - static_cast<T>(1.0)) * x);
}

template <typename T>
__host__ __device__ inline void distill_rhs_impl(
    const T* x,
    const T* u,
    T* dx
) {
    const T D  = static_cast<T>(0.2);
    const T F  = static_cast<T>(0.4);
    const T xf = static_cast<T>(0.5);

    const T Ac = static_cast<T>(0.5);
    const T At = static_cast<T>(0.25);
    const T Ar = static_cast<T>(1.0);

    T L = u[0] * D;
    T V = L + D;
    T S = L + F;

    dx[0] = V * (distill_vapor_y(x[1]) - x[0]) / Ac;

    for (int i = 1; i <= DISTILL_FT - 2; ++i) {
        dx[i] = (
            L * (x[i - 1] - x[i])
            - V * (distill_vapor_y(x[i]) - distill_vapor_y(x[i + 1]))
        ) / At;
    }

    int f = DISTILL_FT - 1;
    dx[f] = (
        F * xf
        + L * x[f - 1]
        - S * x[f]
        - V * (distill_vapor_y(x[f]) - distill_vapor_y(x[f + 1]))
    ) / At;

    for (int i = DISTILL_FT; i <= DISTILL_N - 2; ++i) {
        dx[i] = (
            S * (x[i - 1] - x[i])
            - V * (distill_vapor_y(x[i]) - distill_vapor_y(x[i + 1]))
        ) / At;
    }

    dx[DISTILL_N - 1] = (
        S * x[DISTILL_N - 2]
        - (F - D) * x[DISTILL_N - 1]
        - V * distill_vapor_y(x[DISTILL_N - 1])
    ) / Ar;
}

template <typename T>
__device__ inline void distill_rhs(
    const T* x,
    const T* u,
    T* dx
) {
    distill_rhs_impl<T>(x, u, dx);
}

template <typename T>
inline void distill_rhs_host(
    const T* x,
    const T* u,
    T* dx
) {
    distill_rhs_impl<T>(x, u, dx);
}