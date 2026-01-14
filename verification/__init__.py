"""
Constrained-Random Verification Suite for Inferno Kernels

This module provides post-silicon-style stress testing for numerical stability
validation of fused CUDA kernels.
"""

from .test_generators import ConstrainedRandomGenerator
from .verification_engine import VerificationEngine
from .statistics import StatisticsCollector
from .test_scenarios import TestScenario, EdgeCaseScenario, BoundaryScenario, StressScenario

__all__ = [
    'ConstrainedRandomGenerator',
    'VerificationEngine',
    'StatisticsCollector',
    'TestScenario',
    'EdgeCaseScenario',
    'BoundaryScenario',
    'StressScenario',
]

