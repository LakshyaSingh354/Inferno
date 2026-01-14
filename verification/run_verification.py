#!/usr/bin/env python3
"""
Main script to run the constrained-random verification suite.

Usage:
    python verification/run_verification.py [options]

Example:
    python verification/run_verification.py --tests 1000 --stress-duration 300
"""

import argparse
import sys
import os
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verification import VerificationEngine
from inferno import compile_model


def create_test_kernel_function(compiled_model):
    """Create a kernel function wrapper from compiled model"""
    def kernel_fn(A, B):
        # For matmul+relu, we need to pass input and weight
        # This assumes the model takes input and uses internal weight
        return compiled_model(A)
    return kernel_fn


def main():
    parser = argparse.ArgumentParser(
        description='Run constrained-random verification suite for Inferno kernels'
    )
    parser.add_argument(
        '--tests',
        type=int,
        default=100,
        help='Number of tests per scenario (default: 100)'
    )
    parser.add_argument(
        '--stress-duration',
        type=int,
        default=0,
        help='Duration of stress test in seconds (0 to skip, default: 0)'
    )
    parser.add_argument(
        '--base-size',
        type=int,
        nargs=3,
        default=[256, 256, 256],
        metavar=('M', 'K', 'N'),
        help='Base matrix size MxKxN (default: 256 256 256)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='verification_report.json',
        help='Output file for detailed report (default: verification_report.json)'
    )
    parser.add_argument(
        '--kernel-file',
        type=str,
        default='src/fused_kernel.cu',
        help='Path to kernel file (default: src/fused_kernel.cu)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This verification suite requires CUDA.")
        sys.exit(1)
    
    print("="*60)
    print("Inferno Kernel Verification Suite")
    print("="*60)
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Tests per scenario: {args.tests}")
    print(f"Base size: {args.base_size}")
    print("="*60)
    
    # Create a test model
    class TestModel(torch.nn.Module):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(input_size, output_size, device='cuda', dtype=torch.float32)
            )
        
        def forward(self, x):
            x = torch.matmul(x, self.weight)
            x = F.relu(x)
            return x
    
    # Compile the model
    print("\nCompiling model...")
    M, K, N = args.base_size
    model = TestModel(K, N).cuda()
    example_input = torch.randn(M, K, device='cuda', dtype=torch.float32)
    
    try:
        compiled_model = compile_model(
            example_inputs=[example_input],
            kernel_filepath=args.kernel_file
        )(TestModel)(K, N).cuda()
        
        # Create kernel function
        def kernel_fn(A, B):
            # For this model, A is input, weight is B
            # But our model uses internal weight, so we need to adapt
            # Create a temporary model with B as weight
            temp_model = TestModel(A.size(1), B.size(1)).cuda()
            temp_model.weight.data = B.t()
            compiled_temp = compile_model(
                example_inputs=[A],
                kernel_filepath=args.kernel_file
            )(TestModel)(A.size(1), B.size(1)).cuda()
            compiled_temp.weight.data = B.t()
            return compiled_temp(A)
        
        # Reference function
        def reference_fn(A, B):
            return F.relu(torch.matmul(A, B))
        
        # Create verification engine
        engine = VerificationEngine(
            kernel_function=kernel_fn,
            reference_function=reference_fn,
            seed=args.seed
        )
        
        # Run verification suite
        print("\nRunning verification suite...")
        results = engine.run_full_suite(
            num_tests_per_scenario=args.tests,
            base_size=tuple(args.base_size),
            verbose=args.verbose
        )
        
        # Run stress test if requested
        if args.stress_duration > 0:
            print("\nRunning stress test...")
            stress_results = engine.run_stress_test(
                duration_seconds=args.stress_duration,
                base_size=tuple(args.base_size),
                verbose=args.verbose
            )
            results['stress_test'] = stress_results
        
        # Export report
        print(f"\nExporting detailed report to {args.output}...")
        engine.export_report(args.output)
        
        # Print worst cases
        print("\nTop 5 Worst Cases:")
        worst = engine.get_worst_cases(5)
        for i, case in enumerate(worst, 1):
            print(f"\n  {i}. {case['test_info'].get('test_name', 'unknown')}")
            print(f"     Max error: {case['metrics']['max_abs_error']:.2e}")
            print(f"     Size: {case['test_info'].get('size', 'unknown')}")
        
        # Final status
        summary = results['summary']
        if summary['pass_rate'] >= 0.99:
            print("\n✅ VERIFICATION PASSED (>=99% pass rate)")
            return 0
        elif summary['pass_rate'] >= 0.95:
            print("\n⚠️  VERIFICATION WARNING (<99% but >=95% pass rate)")
            return 1
        else:
            print("\n❌ VERIFICATION FAILED (<95% pass rate)")
            return 2
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == '__main__':
    sys.exit(main())

