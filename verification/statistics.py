"""
Statistical Analysis for Numerical Stability Verification

Collects and analyzes numerical errors, providing statistical metrics
similar to post-silicon verification methodologies.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class ErrorMetrics:
    """Container for error metrics for a single test case"""
    max_abs_error: float
    mean_abs_error: float
    rms_error: float
    relative_error: float
    max_relative_error: float
    ulp_error: float  # Units in the Last Place
    num_failures: int  # Number of elements exceeding tolerance
    total_elements: int
    
    def to_dict(self) -> Dict:
        return {
            'max_abs_error': self.max_abs_error,
            'mean_abs_error': self.mean_abs_error,
            'rms_error': self.rms_error,
            'relative_error': self.relative_error,
            'max_relative_error': self.max_relative_error,
            'ulp_error': self.ulp_error,
            'num_failures': self.num_failures,
            'total_elements': self.total_elements,
            'failure_rate': self.num_failures / self.total_elements if self.total_elements > 0 else 0.0
        }


class StatisticsCollector:
    """
    Collects and analyzes statistical metrics across multiple test runs.
    
    Mimics post-silicon verification by tracking:
    - Error distributions
    - Failure rates
    - Worst-case scenarios
    - Coverage metrics
    """
    
    def __init__(self):
        self.metrics_history: List[ErrorMetrics] = []
        self.test_results: List[Dict] = []
        self.failure_cases: List[Dict] = []
        self.size_statistics: Dict[str, List[ErrorMetrics]] = defaultdict(list)
        self.distribution_statistics: Dict[str, List[ErrorMetrics]] = defaultdict(list)
    
    def record_test(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        test_info: Optional[Dict] = None
    ) -> ErrorMetrics:
        """
        Record a test case and compute error metrics.
        
        Args:
            actual: Output from kernel under test
            expected: Reference output (PyTorch)
            test_info: Optional dictionary with test metadata
        
        Returns:
            ErrorMetrics object
        """
        # Compute various error metrics
        abs_error = torch.abs(actual - expected)
        max_abs_error = float(torch.max(abs_error).item())
        mean_abs_error = float(torch.mean(abs_error).item())
        rms_error = float(torch.sqrt(torch.mean(abs_error ** 2)).item())
        
        # Relative error (avoid division by zero)
        with torch.no_grad():
            relative_error_tensor = abs_error / (torch.abs(expected) + 1e-10)
            relative_error = float(torch.mean(relative_error_tensor).item())
            max_relative_error = float(torch.max(relative_error_tensor).item())
        
        # ULP (Units in the Last Place) error
        ulp_error = self._compute_ulp_error(actual, expected)
        
        # Count failures (elements exceeding tolerance)
        tolerance = test_info.get('tolerance', 1e-5) if test_info else 1e-5
        failures = (abs_error > tolerance).sum().item()
        total_elements = actual.numel()
        
        metrics = ErrorMetrics(
            max_abs_error=max_abs_error,
            mean_abs_error=mean_abs_error,
            rms_error=rms_error,
            relative_error=relative_error,
            max_relative_error=max_relative_error,
            ulp_error=ulp_error,
            num_failures=failures,
            total_elements=total_elements
        )
        
        self.metrics_history.append(metrics)
        
        # Store test result with metadata
        result = {
            'metrics': metrics.to_dict(),
            'test_info': test_info or {},
            'passed': failures == 0
        }
        self.test_results.append(result)
        
        # Track failures
        if failures > 0:
            self.failure_cases.append(result)
        
        # Group by size
        if test_info and 'size' in test_info:
            size_key = f"{test_info['size']}"
            self.size_statistics[size_key].append(metrics)
        
        # Group by distribution
        if test_info and 'distribution' in test_info:
            dist_key = test_info['distribution']
            self.distribution_statistics[dist_key].append(metrics)
        
        return metrics
    
    def _compute_ulp_error(self, actual: torch.Tensor, expected: torch.Tensor) -> float:
        """
        Compute Units in the Last Place (ULP) error.
        ULP measures the number of representable floating-point values between two numbers.
        """
        if actual.dtype != torch.float32:
            # For other types, use relative error as proxy
            return self._compute_ulp_error_float32(actual.float(), expected.float())
        
        return self._compute_ulp_error_float32(actual, expected)
    
    def _compute_ulp_error_float32(self, actual: torch.Tensor, expected: torch.Tensor) -> float:
        """Compute ULP error for float32"""
        # Convert to numpy for bit manipulation
        actual_np = actual.detach().cpu().numpy()
        expected_np = expected.detach().cpu().numpy()
        
        # Get bit representations
        actual_bits = actual_np.view(np.int32)
        expected_bits = expected_np.view(np.int32)
        
        # Compute ULP difference
        ulp_diff = np.abs(actual_bits.astype(np.int64) - expected_bits.astype(np.int64))
        
        # Handle special cases (NaN, Inf)
        valid_mask = np.isfinite(actual_np) & np.isfinite(expected_np)
        if valid_mask.sum() == 0:
            return float('inf')
        
        max_ulp = float(np.max(ulp_diff[valid_mask]))
        return max_ulp
    
    def get_summary_statistics(self) -> Dict:
        """Get overall summary statistics"""
        if not self.metrics_history:
            return {}
        
        max_errors = [m.max_abs_error for m in self.metrics_history]
        mean_errors = [m.mean_abs_error for m in self.metrics_history]
        rms_errors = [m.rms_error for m in self.metrics_history]
        failure_rates = [m.num_failures / m.total_elements for m in self.metrics_history]
        
        total_tests = len(self.metrics_history)
        total_failures = sum(1 for r in self.test_results if not r['passed'])
        
        return {
            'total_tests': total_tests,
            'total_passed': total_tests - total_failures,
            'total_failed': total_failures,
            'pass_rate': (total_tests - total_failures) / total_tests if total_tests > 0 else 0.0,
            'max_abs_error': {
                'min': float(np.min(max_errors)),
                'max': float(np.max(max_errors)),
                'mean': float(np.mean(max_errors)),
                'std': float(np.std(max_errors)),
                'p50': float(np.percentile(max_errors, 50)),
                'p95': float(np.percentile(max_errors, 95)),
                'p99': float(np.percentile(max_errors, 99)),
            },
            'mean_abs_error': {
                'min': float(np.min(mean_errors)),
                'max': float(np.max(mean_errors)),
                'mean': float(np.mean(mean_errors)),
                'std': float(np.std(mean_errors)),
            },
            'rms_error': {
                'min': float(np.min(rms_errors)),
                'max': float(np.max(rms_errors)),
                'mean': float(np.mean(rms_errors)),
                'std': float(np.std(rms_errors)),
            },
            'failure_rate': {
                'min': float(np.min(failure_rates)),
                'max': float(np.max(failure_rates)),
                'mean': float(np.mean(failure_rates)),
            }
        }
    
    def get_size_statistics(self) -> Dict:
        """Get statistics grouped by matrix size"""
        stats = {}
        for size_key, metrics_list in self.size_statistics.items():
            if not metrics_list:
                continue
            
            max_errors = [m.max_abs_error for m in metrics_list]
            failure_rates = [m.num_failures / m.total_elements for m in metrics_list]
            
            stats[size_key] = {
                'num_tests': len(metrics_list),
                'max_abs_error': {
                    'max': float(np.max(max_errors)),
                    'mean': float(np.mean(max_errors)),
                    'p95': float(np.percentile(max_errors, 95)),
                },
                'failure_rate': {
                    'mean': float(np.mean(failure_rates)),
                    'max': float(np.max(failure_rates)),
                }
            }
        
        return stats
    
    def get_worst_cases(self, top_n: int = 10) -> List[Dict]:
        """Get the worst N test cases by maximum absolute error"""
        sorted_results = sorted(
            self.test_results,
            key=lambda x: x['metrics']['max_abs_error'],
            reverse=True
        )
        return sorted_results[:top_n]
    
    def export_report(self, filepath: str):
        """Export detailed report to JSON file"""
        report = {
            'summary': self.get_summary_statistics(),
            'size_statistics': self.get_size_statistics(),
            'worst_cases': self.get_worst_cases(20),
            'failure_cases': [
                {
                    'metrics': case['metrics'],
                    'test_info': case['test_info']
                }
                for case in self.failure_cases[:100]  # Limit to first 100 failures
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

