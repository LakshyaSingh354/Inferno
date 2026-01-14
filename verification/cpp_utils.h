/**
 * C++ Numerical Stability Utilities
 * 
 * Low-level utilities for numerical stability checks in CUDA kernels.
 * These can be used for inline verification or debugging.
 */

#ifndef VERIFICATION_CPP_UTILS_H
#define VERIFICATION_CPP_UTILS_H

#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace verification {

/**
 * Compute Units in the Last Place (ULP) difference between two floats
 */
__device__ __forceinline__ int32_t ulp_diff(float a, float b) {
    int32_t a_bits = __float_as_int(a);
    int32_t b_bits = __float_as_int(b);
    
    // Handle special cases
    if (isnan(a) || isnan(b)) return INT32_MAX;
    if (isinf(a) || isinf(b)) {
        if (a == b) return 0;
        return INT32_MAX;
    }
    
    // Compute absolute difference in bit representation
    int32_t diff = a_bits - b_bits;
    return (diff < 0) ? -diff : diff;
}

/**
 * Check if two floats are close within tolerance
 */
__device__ __forceinline__ bool is_close(float a, float b, float rtol = 1e-5f, float atol = 1e-8f) {
    float diff = fabsf(a - b);
    float max_val = fmaxf(fabsf(a), fabsf(b));
    return diff <= atol + rtol * max_val;
}

/**
 * Compute relative error between two floats
 */
__device__ __forceinline__ float relative_error(float actual, float expected) {
    float diff = fabsf(actual - expected);
    float denom = fabsf(expected);
    if (denom < FLT_MIN) {
        return diff;  // Use absolute error for near-zero values
    }
    return diff / denom;
}

/**
 * Check for numerical issues (NaN, Inf) in a value
 */
__device__ __forceinline__ bool is_finite(float val) {
    return isfinite(val);
}

/**
 * Check if value is denormal (subnormal)
 */
__device__ __forceinline__ bool is_denormal(float val) {
    if (!isfinite(val) || val == 0.0f) return false;
    int32_t bits = __float_as_int(val);
    int32_t exp = (bits >> 23) & 0xFF;
    return exp == 0 && (bits & 0x7FFFFF) != 0;
}

/**
 * Atomic add for error tracking (for debugging)
 */
__device__ __forceinline__ void atomic_add_error(float* error_sum, float error) {
    atomicAdd(error_sum, error);
}

/**
 * Compute maximum ULP error across a block
 * Useful for kernel-level verification
 */
__device__ void compute_block_ulp_error(
    const float* actual,
    const float* expected,
    int num_elements,
    int32_t* max_ulp_error
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_elements) {
        int32_t ulp = ulp_diff(actual[tid], expected[tid]);
        atomicMax((int32_t*)max_ulp_error, ulp);
    }
}

} // namespace verification

#endif // VERIFICATION_CPP_UTILS_H

