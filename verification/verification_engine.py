"""
Main Verification Engine

Orchestrates constrained-random verification testing similar to
post-silicon verification methodologies.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional, Dict, List
import time
from tqdm import tqdm

from .test_generators import ConstrainedRandomGenerator
from .test_scenarios import TestScenario, EdgeCaseScenario, BoundaryScenario, StressScenario, RandomScenario
from .statistics import StatisticsCollector


class VerificationEngine:
    """
    Main engine for running constrained-random verification tests.
    
    Mimics post-silicon verification by:
    - Running multiple test scenarios
    - Collecting statistical metrics
    - Generating comprehensive reports
    - Supporting long-running stress tests
    """
    
    def __init__(
        self,
        kernel_function: Callable,
        reference_function: Optional[Callable] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize verification engine.
        
        Args:
            kernel_function: Function to test (takes A, B, returns C)
            reference_function: Reference implementation (default: PyTorch matmul + relu)
            seed: Random seed for reproducibility
        """
        self.kernel_function = kernel_function
        self.reference_function = reference_function or self._default_reference
        self.generator = ConstrainedRandomGenerator(seed=seed)
        self.stats = StatisticsCollector()
    
    def _default_reference(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Default reference: PyTorch matmul + ReLU"""
        return F.relu(torch.matmul(A, B))
    
    def run_scenario(
        self,
        scenario: TestScenario,
        num_tests: int,
        verbose: bool = True
    ) -> Dict:
        """
        Run a single test scenario.
        
        Args:
            scenario: TestScenario instance
            num_tests: Number of tests to run
            verbose: Print progress
        
        Returns:
            Dictionary with results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Scenario: {scenario.name}")
            print(f"Description: {scenario.get_description()}")
            print(f"Number of tests: {num_tests}")
            print(f"{'='*60}")
        
        tests = scenario.generate_tests(num_tests)
        passed = 0
        failed = 0
        
        iterator = tqdm(tests, desc=f"Testing {scenario.name}") if verbose else tests
        
        for test_case in iterator:
            try:
                # Run kernel under test
                actual = self.kernel_function(test_case['A'], test_case['B'])
                
                # Run reference
                expected = self.reference_function(test_case['A'], test_case['B'])
                
                # Record metrics
                test_info = {
                    'size': test_case['size'],
                    'scenario': test_case['scenario'],
                    'test_name': test_case['test_name'],
                    'tolerance': test_case.get('tolerance', 1e-5),
                    'distribution': test_case.get('distribution', 'unknown'),
                }
                
                metrics = self.stats.record_test(actual, expected, test_info)
                
                if metrics.num_failures == 0:
                    passed += 1
                else:
                    failed += 1
                    if verbose:
                        print(f"\n  ❌ FAILED: {test_case['test_name']}")
                        print(f"     Max error: {metrics.max_abs_error:.2e}")
                        print(f"     Failures: {metrics.num_failures}/{metrics.total_elements}")
                
            except Exception as e:
                failed += 1
                if verbose:
                    print(f"\n  ❌ ERROR in {test_case['test_name']}: {e}")
        
        result = {
            'scenario': scenario.name,
            'total_tests': num_tests,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / num_tests if num_tests > 0 else 0.0
        }
        
        if verbose:
            print(f"\nScenario Results:")
            print(f"  Passed: {passed}/{num_tests}")
            print(f"  Failed: {failed}/{num_tests}")
            print(f"  Pass Rate: {result['pass_rate']*100:.2f}%")
        
        return result
    
    def run_full_suite(
        self,
        num_tests_per_scenario: int = 100,
        base_size: tuple = (256, 256, 256),
        verbose: bool = True
    ) -> Dict:
        """
        Run the full verification suite with all scenarios.
        
        Args:
            num_tests_per_scenario: Number of tests per scenario
            base_size: Base matrix size (M, K, N)
            verbose: Print progress
        
        Returns:
            Dictionary with comprehensive results
        """
        if verbose:
            print("\n" + "="*60)
            print(" " * 15 + "INFERNO VERIFICATION SUITE")
            print("="*60)
            print("Post-Silicon Style Constrained-Random Verification")
            print("="*60)
        
        scenarios = [
            EdgeCaseScenario(self.generator, base_size),
            BoundaryScenario(self.generator, base_size),
            StressScenario(self.generator, base_size),
            RandomScenario(self.generator, base_size),
        ]
        
        results = []
        start_time = time.time()
        
        for scenario in scenarios:
            result = self.run_scenario(scenario, num_tests_per_scenario, verbose)
            results.append(result)
        
        elapsed_time = time.time() - start_time
        
        # Get summary statistics
        summary = self.stats.get_summary_statistics()
        
        full_results = {
            'scenario_results': results,
            'summary': summary,
            'elapsed_time': elapsed_time,
        }
        
        if verbose:
            self._print_summary(full_results)
        
        return full_results
    
    def run_stress_test(
        self,
        duration_seconds: int = 300,
        base_size: tuple = (256, 256, 256),
        verbose: bool = True
    ) -> Dict:
        """
        Run a long-duration stress test.
        
        Args:
            duration_seconds: How long to run (in seconds)
            base_size: Base matrix size
            verbose: Print progress
        
        Returns:
            Dictionary with stress test results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Stress Test for {duration_seconds} seconds")
            print(f"{'='*60}")
        
        scenario = RandomScenario(self.generator, base_size)
        start_time = time.time()
        test_count = 0
        
        # Create a temporary stats collector for stress test
        stress_stats = StatisticsCollector()
        original_stats = self.stats
        self.stats = stress_stats
        
        try:
            while time.time() - start_time < duration_seconds:
                # Generate and run a batch of tests
                tests = scenario.generate_tests(10)
                for test_case in tests:
                    if time.time() - start_time >= duration_seconds:
                        break
                    
                    try:
                        actual = self.kernel_function(test_case['A'], test_case['B'])
                        expected = self.reference_function(test_case['A'], test_case['B'])
                        
                        test_info = {
                            'size': test_case['size'],
                            'scenario': 'stress_test',
                            'test_name': f'stress_{test_count}',
                            'tolerance': 1e-5,
                        }
                        
                        stress_stats.record_test(actual, expected, test_info)
                        test_count += 1
                    except Exception as e:
                        if verbose:
                            print(f"Error in stress test: {e}")
        
        finally:
            self.stats = original_stats
        
        elapsed_time = time.time() - start_time
        summary = stress_stats.get_summary_statistics()
        
        result = {
            'duration_seconds': elapsed_time,
            'total_tests': test_count,
            'summary': summary,
        }
        
        if verbose:
            print(f"\nStress Test Complete:")
            print(f"  Duration: {elapsed_time:.2f} seconds")
            print(f"  Total tests: {test_count}")
            print(f"  Pass rate: {summary.get('pass_rate', 0.0)*100:.2f}%")
            print(f"  Max error: {summary.get('max_abs_error', {}).get('max', 0.0):.2e}")
        
        return result
    
    def _print_summary(self, results: Dict):
        """Print summary of verification results"""
        print("\n" + "="*60)
        print(" " * 20 + "VERIFICATION SUMMARY")
        print("="*60)
        
        summary = results['summary']
        print(f"\nOverall Statistics:")
        print(f"  Total Tests: {summary['total_tests']}")
        print(f"  Passed: {summary['total_passed']}")
        print(f"  Failed: {summary['total_failed']}")
        print(f"  Pass Rate: {summary['pass_rate']*100:.2f}%")
        
        print(f"\nError Statistics:")
        max_err = summary['max_abs_error']
        print(f"  Max Absolute Error:")
        print(f"    Min:  {max_err['min']:.2e}")
        print(f"    Mean: {max_err['mean']:.2e}")
        print(f"    Max:  {max_err['max']:.2e}")
        print(f"    P95:  {max_err['p95']:.2e}")
        print(f"    P99:  {max_err['p99']:.2e}")
        
        print(f"\nElapsed Time: {results['elapsed_time']:.2f} seconds")
        print("="*60)
    
    def export_report(self, filepath: str):
        """Export detailed verification report"""
        return self.stats.export_report(filepath)
    
    def get_worst_cases(self, top_n: int = 10) -> List[Dict]:
        """Get worst test cases"""
        return self.stats.get_worst_cases(top_n)

