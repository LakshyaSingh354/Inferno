import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_generation import compile

def test_simple_case():
    """Test with a simple 2x2 case to debug the kernel"""
    print("=== Testing Simple 2x2 Case ===")
    
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        
        def forward(self, x):
            x = torch.matmul(x, self.weight)
            x = F.relu(x)
            return x
    
    # Create tensors with known values
    model = SimpleModel().cuda()
    input_tensor = torch.tensor([[1.0, -1.0], [2.0, -2.0]], device='cuda')
    
    print(f"Input: {input_tensor}")
    print(f"Weight: {model.weight}")
    
    # Original PyTorch computation
    original_output = model(input_tensor)
    print(f"Original output: {original_output}")
    
    # Manual computation for verification
    manual_matmul = torch.matmul(input_tensor, model.weight)
    print(f"Manual matmul: {manual_matmul}")
    manual_relu = F.relu(manual_matmul)
    print(f"Manual ReLU: {manual_relu}")
    
    # Try to compile and test
    try:
        compiled_model = compile(model, [input_tensor], 'src/fused_matmul_relu_kernel.cu')
        compiled_output = compiled_model(input_tensor)
        print(f"Compiled output: {compiled_output}")
        
        is_close = torch.allclose(original_output, compiled_output, atol=1e-5)
        print(f"Verification: {'✅ PASSED' if is_close else '❌ FAILED'}")
        
        if not is_close:
            print(f"Difference: {torch.abs(original_output - compiled_output)}")
            
    except Exception as e:
        print(f"Compilation failed: {e}")

def test_larger_case():
    """Test with a larger case to see if the issue persists"""
    print("\n=== Testing Larger 4x4 Case ===")
    
    class LargerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(4, 4))
        
        def forward(self, x):
            x = torch.matmul(x, self.weight)
            x = F.relu(x)
            return x
    
    model = LargerModel().cuda()
    input_tensor = torch.randn(4, 4, device='cuda')
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {model.weight.shape}")
    
    original_output = model(input_tensor)
    print(f"Original output shape: {original_output.shape}")
    
    try:
        compiled_model = compile(model, [input_tensor], 'src/fused_matmul_relu_kernel.cu')
        compiled_output = compiled_model(input_tensor)
        print(f"Compiled output shape: {compiled_output.shape}")
        
        is_close = torch.allclose(original_output, compiled_output, atol=1e-5)
        print(f"Verification: {'✅ PASSED' if is_close else '❌ FAILED'}")
        
        if not is_close:
            print(f"Max difference: {torch.max(torch.abs(original_output - compiled_output))}")
            print(f"Mean difference: {torch.mean(torch.abs(original_output - compiled_output))}")
            
    except Exception as e:
        print(f"Compilation failed: {e}")

if __name__ == "__main__":
    test_simple_case()
    test_larger_case()
