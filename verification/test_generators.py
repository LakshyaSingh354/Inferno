"""
Constrained-Random Test Generators

Generates test cases with constraints mimicking post-silicon verification methodologies.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Callable
from enum import Enum
import random


class ValueDistribution(Enum):
    """Distribution types for random value generation"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    EXPONENTIAL = "exponential"
    LOG_NORMAL = "log_normal"
    EDGE_BIASED = "edge_biased"  # Biased towards edge cases


class ConstrainedRandomGenerator:
    """
    Generates constrained-random test cases for kernel verification.
    
    Mimics post-silicon verification by generating test cases that:
    - Cover edge cases (zeros, ones, extremes)
    - Exercise boundary conditions
    - Stress numerical precision
    - Include pathological cases
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate_matrix_pair(
        self,
        M: int,
        K: int,
        N: int,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda',
        constraints: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate a constrained-random matrix pair (A, B) for GEMM.
        
        Args:
            M, K, N: Matrix dimensions (A: MxK, B: KxN)
            dtype: Data type
            device: Device ('cuda' or 'cpu')
            constraints: Dictionary with constraints:
                - 'value_range': (min, max) tuple
                - 'distribution': ValueDistribution enum
                - 'sparsity': float (0-1, probability of zero)
                - 'include_denormals': bool
                - 'include_inf': bool
                - 'include_nan': bool
                - 'edge_case_prob': float (0-1, probability of edge case)
        
        Returns:
            Tuple of (A, B) tensors
        """
        constraints = constraints or {}
        
        # Check if we should generate an edge case
        edge_prob = constraints.get('edge_case_prob', 0.1)
        if self.rng.random() < edge_prob:
            return self._generate_edge_case_matrix_pair(M, K, N, dtype, device)
        
        # Generate normal constrained-random matrices
        value_range = constraints.get('value_range', (-10.0, 10.0))
        distribution = constraints.get('distribution', ValueDistribution.UNIFORM)
        # Handle string distribution names
        if isinstance(distribution, str):
            distribution = ValueDistribution(distribution)
        sparsity = constraints.get('sparsity', 0.0)
        
        A = self._generate_matrix(M, K, value_range, distribution, sparsity, dtype, device, constraints)
        B = self._generate_matrix(K, N, value_range, distribution, sparsity, dtype, device, constraints)
        
        return A, B
    
    def _generate_matrix(
        self,
        rows: int,
        cols: int,
        value_range: Tuple[float, float],
        distribution: ValueDistribution,
        sparsity: float,
        dtype: torch.dtype,
        device: str,
        constraints: Dict
    ) -> torch.Tensor:
        """Generate a single matrix with specified constraints"""
        
        # Start with base distribution
        if distribution == ValueDistribution.UNIFORM:
            matrix = torch.empty(rows, cols, dtype=dtype, device=device)
            matrix.uniform_(value_range[0], value_range[1])
        elif distribution == ValueDistribution.NORMAL:
            mean = (value_range[0] + value_range[1]) / 2
            std = (value_range[1] - value_range[0]) / 6
            matrix = torch.normal(mean, std, size=(rows, cols), dtype=dtype, device=device)
        elif distribution == ValueDistribution.EXPONENTIAL:
            # Scale exponential to fit range
            scale = (value_range[1] - value_range[0]) / 5.0
            matrix = torch.empty(rows, cols, dtype=dtype, device=device)
            matrix.exponential_(scale)
            matrix = matrix.clamp(value_range[0], value_range[1])
        elif distribution == ValueDistribution.LOG_NORMAL:
            mean = np.log((value_range[0] + value_range[1]) / 2)
            std = np.log((value_range[1] - value_range[0]) / 2)
            matrix = torch.empty(rows, cols, dtype=dtype, device=device)
            matrix.log_normal_(mean, std)
            matrix = matrix.clamp(value_range[0], value_range[1])
        elif distribution == ValueDistribution.EDGE_BIASED:
            # Mix of uniform with bias towards edges
            matrix = torch.empty(rows, cols, dtype=dtype, device=device)
            uniform_part = torch.empty(rows, cols, dtype=dtype, device=device)
            uniform_part.uniform_(value_range[0], value_range[1])
            
            # 30% chance of edge values
            edge_mask = torch.rand(rows, cols, device=device) < 0.3
            edge_values = torch.where(
                torch.rand(rows, cols, device=device) < 0.5,
                torch.full((rows, cols), value_range[0], dtype=dtype, device=device),
                torch.full((rows, cols), value_range[1], dtype=dtype, device=device)
            )
            matrix = torch.where(edge_mask, edge_values, uniform_part)
        else:
            matrix = torch.randn(rows, cols, dtype=dtype, device=device)
        
        # Apply sparsity
        if sparsity > 0:
            zero_mask = torch.rand(rows, cols, device=device) < sparsity
            matrix[zero_mask] = 0.0
        
        # Inject special values if requested
        if constraints.get('include_denormals', False):
            matrix = self._inject_denormals(matrix, dtype)
        
        if constraints.get('include_inf', False):
            matrix = self._inject_inf(matrix)
        
        if constraints.get('include_nan', False):
            matrix = self._inject_nan(matrix)
        
        return matrix
    
    def _generate_edge_case_matrix_pair(
        self,
        M: int,
        K: int,
        N: int,
        dtype: torch.dtype,
        device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate edge case matrices"""
        case_type = self.rng.choice([
            'zeros', 'ones', 'identity_like', 'negative', 'large', 'small',
            'mixed_signs', 'diagonal_dominant'
        ])
        
        if case_type == 'zeros':
            A = torch.zeros(M, K, dtype=dtype, device=device)
            B = torch.zeros(K, N, dtype=dtype, device=device)
        elif case_type == 'ones':
            A = torch.ones(M, K, dtype=dtype, device=device)
            B = torch.ones(K, N, dtype=dtype, device=device)
        elif case_type == 'identity_like':
            # Create identity-like matrices
            A = torch.eye(min(M, K), dtype=dtype, device=device)
            if M > K:
                A = torch.cat([A, torch.zeros(M - K, K, dtype=dtype, device=device)], dim=0)
            elif K > M:
                A = torch.cat([A, torch.zeros(M, K - M, dtype=dtype, device=device)], dim=1)
            
            B = torch.eye(min(K, N), dtype=dtype, device=device)
            if K > N:
                B = torch.cat([B, torch.zeros(K - N, N, dtype=dtype, device=device)], dim=0)
            elif N > K:
                B = torch.cat([B, torch.zeros(K, N - K, dtype=dtype, device=device)], dim=1)
        elif case_type == 'negative':
            A = torch.full((M, K), -1.0, dtype=dtype, device=device)
            B = torch.full((K, N), -1.0, dtype=dtype, device=device)
        elif case_type == 'large':
            max_val = torch.finfo(dtype).max / 1000.0  # Avoid overflow
            A = torch.full((M, K), max_val, dtype=dtype, device=device)
            B = torch.full((K, N), max_val, dtype=dtype, device=device)
        elif case_type == 'small':
            min_val = torch.finfo(dtype).tiny * 10.0
            A = torch.full((M, K), min_val, dtype=dtype, device=device)
            B = torch.full((K, N), min_val, dtype=dtype, device=device)
        elif case_type == 'mixed_signs':
            A = torch.randn(M, K, dtype=dtype, device=device)
            B = torch.randn(K, N, dtype=dtype, device=device)
        elif case_type == 'diagonal_dominant':
            # Create diagonally dominant matrices
            A = torch.randn(M, K, dtype=dtype, device=device)
            A = A + torch.diag(torch.ones(min(M, K), dtype=dtype, device=device) * 10.0)
            if M > K:
                A = torch.cat([A, torch.zeros(M - K, K, dtype=dtype, device=device)], dim=0)
            
            B = torch.randn(K, N, dtype=dtype, device=device)
            B = B + torch.diag(torch.ones(min(K, N), dtype=dtype, device=device) * 10.0)
            if K > N:
                B = torch.cat([B, torch.zeros(K - N, N, dtype=dtype, device=device)], dim=0)
        else:
            A = torch.randn(M, K, dtype=dtype, device=device)
            B = torch.randn(K, N, dtype=dtype, device=device)
        
        return A, B
    
    def _inject_denormals(self, tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Inject denormal (subnormal) floating-point values"""
        if dtype != torch.float32:
            return tensor  # Denormals mainly relevant for float32
        
        # Create denormal values (very small, non-zero)
        denormal_mask = torch.rand_like(tensor) < 0.01  # 1% chance
        # Denormal range for float32: ~1.4e-45 to ~1.2e-38
        denormal_values = torch.tensor(1e-40, dtype=dtype, device=tensor.device)
        tensor = torch.where(denormal_mask, denormal_values, tensor)
        return tensor
    
    def _inject_inf(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inject infinity values"""
        inf_mask = torch.rand_like(tensor) < 0.01  # 1% chance
        inf_values = torch.tensor(float('inf'), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.where(inf_mask, inf_values, tensor)
        return tensor
    
    def _inject_nan(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inject NaN values"""
        nan_mask = torch.rand_like(tensor) < 0.01  # 1% chance
        nan_values = torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
        tensor = torch.where(nan_mask, nan_values, tensor)
        return tensor
    
    def generate_size_variations(
        self,
        base_M: int,
        base_K: int,
        base_N: int,
        num_variations: int = 20
    ) -> List[Tuple[int, int, int]]:
        """
        Generate size variations around base dimensions.
        Includes edge cases like non-multiples of block size, prime numbers, etc.
        """
        sizes = []
        
        # Standard sizes
        sizes.append((base_M, base_K, base_N))
        
        # Powers of 2
        for exp in range(3, 13):
            size = 2 ** exp
            sizes.append((size, size, size))
        
        # Non-multiples of block size (16)
        for offset in [1, 7, 15, 17, 31]:
            sizes.append((base_M + offset, base_K + offset, base_N + offset))
        
        # Prime numbers (stress alignment)
        primes = [17, 31, 61, 127, 251, 509, 1021]
        for p in primes[:3]:
            sizes.append((p, p, p))
        
        # Rectangular matrices
        sizes.extend([
            (32, 128, 64),
            (128, 32, 256),
            (256, 64, 128),
            (1, 1024, 1),  # Extreme rectangular
            (1024, 1, 1024),
        ])
        
        # Very small sizes
        sizes.extend([
            (1, 1, 1),
            (2, 2, 2),
            (3, 3, 3),
            (4, 4, 4),
        ])
        
        return sizes[:num_variations]

