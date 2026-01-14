# Constrained-Random Verification Suite

A comprehensive verification suite for validating kernel numerical stability, mimicking post-silicon stress testing methodologies.

## Overview

This verification suite provides:

- **Constrained-Random Testing**: Generates test cases with controlled constraints to exercise various numerical conditions
- **Multiple Test Scenarios**: Edge cases, boundary conditions, stress tests, and random tests
- **Statistical Analysis**: Comprehensive error metrics including ULP (Units in Last Place) errors
- **Post-Silicon Style**: Mimics hardware verification methodologies used in chip design

## Features

### Test Generators

- **Value Distributions**: Uniform, Normal, Exponential, Log-Normal, Edge-Biased
- **Edge Cases**: Zeros, ones, identity matrices, extreme values, mixed signs
- **Boundary Conditions**: Various matrix sizes, non-multiples of block size, prime numbers
- **Numerical Stress**: Denormals, Inf, NaN, wide ranges, sparse matrices

### Statistical Metrics

- Maximum absolute error
- Mean absolute error
- RMS error
- Relative error
- ULP (Units in Last Place) error
- Failure rates
- Percentile statistics (P50, P95, P99)

### Test Scenarios

1. **Edge Case Scenario**: Tests zeros, ones, extremes, mixed signs
2. **Boundary Scenario**: Tests various matrix sizes and alignment issues
3. **Stress Scenario**: Tests with extreme numerical conditions
4. **Random Scenario**: Fully random constrained tests

## Usage

### Basic Usage

```python
from verification import VerificationEngine
import torch
import torch.nn.functional as F

# Define your kernel function
def kernel_fn(A, B):
    # Your kernel implementation
    return your_kernel(A, B)

# Define reference function
def reference_fn(A, B):
    return F.relu(torch.matmul(A, B))

# Create verification engine
engine = VerificationEngine(
    kernel_function=kernel_fn,
    reference_function=reference_fn,
    seed=42
)

# Run verification suite
results = engine.run_full_suite(
    num_tests_per_scenario=100,
    base_size=(256, 256, 256),
    verbose=True
)

# Export report
engine.export_report('verification_report.json')
```

### Command Line Usage

```bash
# Run full verification suite
python verification/run_verification.py --tests 1000

# Run with stress test
python verification/run_verification.py --tests 100 --stress-duration 300

# Custom matrix size
python verification/run_verification.py --tests 100 --base-size 512 512 512

# With seed for reproducibility
python verification/run_verification.py --tests 100 --seed 42
```

### Example Script

```bash
python verification/example_verification.py
```

## Test Generators

### ConstrainedRandomGenerator

Generates test matrices with various constraints:

```python
from verification import ConstrainedRandomGenerator

generator = ConstrainedRandomGenerator(seed=42)

# Generate with constraints
constraints = {
    'value_range': (-10.0, 10.0),
    'distribution': ValueDistribution.UNIFORM,
    'sparsity': 0.1,  # 10% zeros
    'edge_case_prob': 0.2,  # 20% chance of edge case
}

A, B = generator.generate_matrix_pair(256, 256, 256, constraints=constraints)
```

### Available Distributions

- `ValueDistribution.UNIFORM`: Uniform distribution
- `ValueDistribution.NORMAL`: Normal distribution
- `ValueDistribution.EXPONENTIAL`: Exponential distribution
- `ValueDistribution.LOG_NORMAL`: Log-normal distribution
- `ValueDistribution.EDGE_BIASED`: Biased towards edge values

## C++ Utilities

The suite includes C++ utilities for low-level numerical checks:

```cpp
#include "verification/cpp_utils.h"

// Compute ULP difference
int32_t ulp = verification::ulp_diff(actual, expected);

// Check if values are close
bool close = verification::is_close(actual, expected, rtol=1e-5f, atol=1e-8f);

// Compute relative error
float rel_err = verification::relative_error(actual, expected);
```

## Report Format

The verification suite generates JSON reports with:

- **Summary Statistics**: Overall pass rate, error statistics
- **Size Statistics**: Error metrics grouped by matrix size
- **Worst Cases**: Test cases with highest errors
- **Failure Cases**: All failed test cases

Example report structure:

```json
{
  "summary": {
    "total_tests": 400,
    "total_passed": 398,
    "total_failed": 2,
    "pass_rate": 0.995,
    "max_abs_error": {
      "min": 1.2e-7,
      "max": 5.3e-5,
      "mean": 2.1e-6,
      "p95": 1.8e-5,
      "p99": 4.2e-5
    }
  },
  "size_statistics": { ... },
  "worst_cases": [ ... ],
  "failure_cases": [ ... ]
}
```

## Integration with Inferno

The verification suite is designed to work seamlessly with Inferno's compilation pipeline:

```python
from inferno import compile_model
from verification import VerificationEngine

# Compile your model
@compile_model(example_inputs=[torch.randn(256, 256, device='cuda')])
class MyModel(nn.Module):
    ...

# Create verification engine
engine = VerificationEngine(
    kernel_function=compiled_model,
    reference_function=reference_fn
)

# Run verification
results = engine.run_full_suite()
```

## Best Practices

1. **Start Small**: Begin with a small number of tests to verify the setup
2. **Use Seeds**: Set a seed for reproducibility during development
3. **Check Worst Cases**: Always review worst cases to understand failure modes
4. **Stress Testing**: Run long-duration stress tests before deployment
5. **Monitor Metrics**: Track error statistics over time to detect regressions

## Architecture

The verification suite follows post-silicon verification principles:

- **Constrained Random**: Tests are random but constrained to exercise specific conditions
- **Coverage**: Multiple scenarios ensure comprehensive coverage
- **Statistical Analysis**: Quantitative metrics for numerical stability
- **Reproducibility**: Seed-based generation for reproducible results
- **Scalability**: Can run thousands of tests efficiently

## Future Enhancements

Potential future additions:

- Coverage metrics (code coverage, value coverage)
- Property-based testing (invariants, properties)
- Performance regression testing
- Multi-GPU verification
- Custom constraint definitions
- Integration with CI/CD pipelines

