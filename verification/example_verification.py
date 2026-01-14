#!/usr/bin/env python3
"""
Example script demonstrating how to use the verification suite.

This script shows how to verify a kernel implementation using
constrained-random testing.
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification import VerificationEngine, ConstrainedRandomGenerator
from inferno import compile


def example_basic_verification():
    """Basic example of running verification"""
    print("="*60)
    print("Basic Verification Example")
    print("="*60)
    
    # Define a simple model
    class SimpleModel(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(input_size, output_size, device='cuda', dtype=torch.float32)
            )
        
        def forward(self, x):
            x = torch.matmul(x, self.weight)
            x = F.relu(x)
            return x
    
    # Create model and compile it
    M, K, N = 256, 256, 256
    model = SimpleModel(K, N).cuda()
    example_input = torch.randn(M, K, device='cuda', dtype=torch.float32)
    
    print("\nCompiling model...")
    compiled_model = compile(model, [example_input], 'src/fused_kernel.cu')
    
    # Define kernel function (wrapper around compiled model)
    def kernel_fn(A, B):
        # Create a temporary model with B as weight
        temp_model = SimpleModel(A.size(1), B.size(1)).cuda()
        temp_model.weight.data = B.t()
        compiled_temp = compile(temp_model, [A], 'src/fused_kernel.cu')
        return compiled_temp(A)
    
    # Reference function (PyTorch)
    def reference_fn(A, B):
        return F.relu(torch.matmul(A, B))
    
    # Create verification engine
    engine = VerificationEngine(
        kernel_function=kernel_fn,
        reference_function=reference_fn,
        seed=42
    )
    
    # Run a small verification suite
    print("\nRunning verification suite (10 tests per scenario)...")
    results = engine.run_full_suite(
        num_tests_per_scenario=10,
        base_size=(M, K, N),
        verbose=True
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)
    summary = results['summary']
    print(f"Total tests: {summary['total_tests']}")
    print(f"Pass rate: {summary['pass_rate']*100:.2f}%")
    print(f"Max error: {summary['max_abs_error']['max']:.2e}")
    
    # Export report
    engine.export_report('verification_report.json')
    print("\nDetailed report saved to verification_report.json")
    
    return results


def example_custom_test():
    """Example of creating custom test cases"""
    print("\n" + "="*60)
    print("Custom Test Example")
    print("="*60)
    
    generator = ConstrainedRandomGenerator(seed=42)
    
    # Generate matrices with specific constraints
    constraints = {
        'value_range': (-10.0, 10.0),
        'distribution': 'uniform',
        'sparsity': 0.1,  # 10% zeros
        'edge_case_prob': 0.2,  # 20% chance of edge case
    }
    
    A, B = generator.generate_matrix_pair(128, 128, 128, constraints=constraints)
    
    print(f"Generated matrices:")
    print(f"  A shape: {A.shape}, range: [{A.min():.2f}, {A.max():.2f}]")
    print(f"  B shape: {B.shape}, range: [{B.min():.2f}, {B.max():.2f}]")
    
    # Test with reference
    expected = F.relu(torch.matmul(A, B))
    print(f"\nReference output shape: {expected.shape}")
    print(f"Output range: [{expected.min():.2f}, {expected.max():.2f}]")


def example_edge_cases():
    """Example of testing edge cases"""
    print("\n" + "="*60)
    print("Edge Cases Example")
    print("="*60)
    
    generator = ConstrainedRandomGenerator(seed=42)
    
    # Test various edge cases
    edge_cases = [
        ('zeros', {'edge_case_prob': 1.0}),
        ('ones', {'edge_case_prob': 1.0}),
        ('large_values', {'value_range': (1e6, 1e7), 'edge_case_prob': 0.0}),
        ('small_values', {'value_range': (1e-6, 1e-5), 'edge_case_prob': 0.0}),
    ]
    
    for name, constraints in edge_cases:
        A, B = generator.generate_matrix_pair(64, 64, 64, constraints=constraints)
        result = F.relu(torch.matmul(A, B))
        print(f"\n{name}:")
        print(f"  A range: [{A.min():.2e}, {A.max():.2e}]")
        print(f"  Output range: [{result.min():.2e}, {result.max():.2e}]")
        print(f"  Output sum: {result.sum():.2e}")


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This example requires CUDA.")
        sys.exit(1)
    
    try:
        # Run examples
        example_basic_verification()
        example_custom_test()
        example_edge_cases()
        
        print("\n" + "="*60)
        print("Examples completed successfully!")
        print("="*60)
    
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

