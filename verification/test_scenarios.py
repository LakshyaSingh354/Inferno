"""
Test Scenarios for Constrained-Random Verification

Defines various test scenarios that exercise different aspects of
numerical stability and correctness.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
import torch
from .test_generators import ConstrainedRandomGenerator, ValueDistribution


class TestScenario(ABC):
    """Base class for test scenarios"""
    
    def __init__(self, name: str, generator: ConstrainedRandomGenerator):
        self.name = name
        self.generator = generator
    
    @abstractmethod
    def generate_tests(self, num_tests: int) -> List[Dict]:
        """Generate a list of test cases"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get description of this scenario"""
        pass


class EdgeCaseScenario(TestScenario):
    """Tests edge cases: zeros, ones, extremes, etc."""
    
    def __init__(self, generator: ConstrainedRandomGenerator, base_size: Tuple[int, int, int] = (256, 256, 256)):
        super().__init__("Edge Cases", generator)
        self.base_size = base_size
    
    def generate_tests(self, num_tests: int) -> List[Dict]:
        tests = []
        M, K, N = self.base_size
        
        # Fixed edge cases
        edge_cases = [
            {'name': 'zeros', 'constraints': {'edge_case_prob': 1.0}},
            {'name': 'ones', 'constraints': {'edge_case_prob': 1.0}},
            {'name': 'negative_ones', 'constraints': {'edge_case_prob': 1.0}},
            {'name': 'large_values', 'constraints': {'value_range': (1e6, 1e7), 'edge_case_prob': 0.0}},
            {'name': 'small_values', 'constraints': {'value_range': (1e-6, 1e-5), 'edge_case_prob': 0.0}},
            {'name': 'mixed_signs', 'constraints': {'value_range': (-100, 100), 'edge_case_prob': 0.0}},
        ]
        
        for case in edge_cases[:num_tests]:
            A, B = self.generator.generate_matrix_pair(M, K, N, constraints=case['constraints'])
            tests.append({
                'A': A,
                'B': B,
                'size': (M, K, N),
                'scenario': self.name,
                'test_name': case['name'],
                'tolerance': 1e-5,
            })
        
        return tests
    
    def get_description(self) -> str:
        return "Tests edge cases including zeros, ones, extreme values, and mixed signs"


class BoundaryScenario(TestScenario):
    """Tests boundary conditions: various matrix sizes, alignment issues"""
    
    def __init__(self, generator: ConstrainedRandomGenerator, base_size: Tuple[int, int, int] = (256, 256, 256)):
        super().__init__("Boundary Conditions", generator)
        self.base_size = base_size
    
    def generate_tests(self, num_tests: int) -> List[Dict]:
        tests = []
        base_M, base_K, base_N = self.base_size
        
        # Generate size variations
        sizes = self.generator.generate_size_variations(base_M, base_K, base_N, num_tests)
        
        for M, K, N in sizes:
            A, B = self.generator.generate_matrix_pair(
                M, K, N,
                constraints={'value_range': (-10.0, 10.0), 'edge_case_prob': 0.0}
            )
            tests.append({
                'A': A,
                'B': B,
                'size': (M, K, N),
                'scenario': self.name,
                'test_name': f'size_{M}x{K}x{N}',
                'tolerance': 1e-5,
            })
        
        return tests
    
    def get_description(self) -> str:
        return "Tests various matrix sizes including non-multiples of block size, primes, and extremes"


class StressScenario(TestScenario):
    """Stress tests with extreme numerical conditions"""
    
    def __init__(self, generator: ConstrainedRandomGenerator, base_size: Tuple[int, int, int] = (256, 256, 256)):
        super().__init__("Numerical Stress", generator)
        self.base_size = base_size
    
    def generate_tests(self, num_tests: int) -> List[Dict]:
        tests = []
        M, K, N = self.base_size
        
        stress_configs = [
            {
                'name': 'high_precision',
                'constraints': {
                    'value_range': (-1.0, 1.0),
                    'distribution': ValueDistribution.UNIFORM,
                    'edge_case_prob': 0.0
                },
                'tolerance': 1e-6
            },
            {
                'name': 'wide_range',
                'constraints': {
                    'value_range': (-1e6, 1e6),
                    'distribution': ValueDistribution.UNIFORM,
                    'edge_case_prob': 0.0
                },
                'tolerance': 1e-3
            },
            {
                'name': 'sparse',
                'constraints': {
                    'value_range': (-10.0, 10.0),
                    'sparsity': 0.5,
                    'edge_case_prob': 0.0
                },
                'tolerance': 1e-5
            },
            {
                'name': 'normal_distribution',
                'constraints': {
                    'value_range': (-100.0, 100.0),
                    'distribution': ValueDistribution.NORMAL,
                    'edge_case_prob': 0.0
                },
                'tolerance': 1e-5
            },
            {
                'name': 'edge_biased',
                'constraints': {
                    'value_range': (-10.0, 10.0),
                    'distribution': ValueDistribution.EDGE_BIASED,
                    'edge_case_prob': 0.0
                },
                'tolerance': 1e-5
            },
        ]
        
        # Repeat configurations to reach num_tests
        for i in range(num_tests):
            config = stress_configs[i % len(stress_configs)]
            A, B = self.generator.generate_matrix_pair(M, K, N, constraints=config['constraints'])
            tests.append({
                'A': A,
                'B': B,
                'size': (M, K, N),
                'scenario': self.name,
                'test_name': f"{config['name']}_{i}",
                'tolerance': config['tolerance'],
                'distribution': config['name'],
            })
        
        return tests
    
    def get_description(self) -> str:
        return "Stress tests with various distributions, sparsity, and numerical ranges"


class RandomScenario(TestScenario):
    """Fully random constrained tests"""
    
    def __init__(self, generator: ConstrainedRandomGenerator, base_size: Tuple[int, int, int] = (256, 256, 256)):
        super().__init__("Constrained Random", generator)
        self.base_size = base_size
    
    def generate_tests(self, num_tests: int) -> List[Dict]:
        tests = []
        M, K, N = self.base_size
        
        for i in range(num_tests):
            # Random constraints
            constraints = {
                'value_range': (-10.0, 10.0),
                'distribution': self.generator.rng.choice(list(ValueDistribution)),
                'sparsity': self.generator.rng.uniform(0.0, 0.3),
                'edge_case_prob': self.generator.rng.uniform(0.0, 0.2),
            }
            
            A, B = self.generator.generate_matrix_pair(M, K, N, constraints=constraints)
            dist_value = constraints['distribution']
            if isinstance(dist_value, ValueDistribution):
                dist_value = dist_value.value
            tests.append({
                'A': A,
                'B': B,
                'size': (M, K, N),
                'scenario': self.name,
                'test_name': f'random_{i}',
                'tolerance': 1e-5,
                'distribution': dist_value,
            })
        
        return tests
    
    def get_description(self) -> str:
        return "Fully random constrained tests with varying distributions and constraints"

