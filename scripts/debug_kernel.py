import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.code_generation import compile

def debug_kernel():
    """Debug the kernel with a simple case and detailed output"""
    print("=== Detailed Kernel Debug ===")
    
    class DebugModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Use simple, known values
            self.weight = torch.nn.Parameter(torch.tensor([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]
            ], dtype=torch.float32))
        
        def forward(self, x):
            x = torch.matmul(x, self.weight)
            x = F.relu(x)
            return x
    
    # Create input with known values
    model = DebugModel().cuda()
    input_tensor = torch.tensor([
        [1.0, -1.0],
        [2.0, -2.0],
        [3.0, -3.0]
    ], dtype=torch.float32, device='cuda')
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Input: {input_tensor}")
    print(f"Weight shape: {model.weight.shape}")
    print(f"Weight: {model.weight}")
    
    # Original PyTorch computation
    original_output = model(input_tensor)
    print(f"Original output shape: {original_output.shape}")
    print(f"Original output: {original_output}")
    
    # Manual step-by-step computation
    manual_matmul = torch.matmul(input_tensor, model.weight)
    print(f"Manual matmul: {manual_matmul}")
    manual_relu = F.relu(manual_matmul)
    print(f"Manual ReLU: {manual_relu}")
    
    # Try compiled version
    try:
        compiled_model = compile(model, [input_tensor], 'src/fused_matmul_relu_kernel.cu')
        compiled_output = compiled_model(input_tensor)
        print(f"Compiled output shape: {compiled_output.shape}")
        print(f"Compiled output: {compiled_output}")
        
        # Check if shapes match
        print(f"Shape match: {original_output.shape == compiled_output.shape}")
        
        # Check if values match
        is_close = torch.allclose(original_output, compiled_output, atol=1e-5)
        print(f"Value match: {'✅ PASSED' if is_close else '❌ FAILED'}")
        
        if not is_close:
            diff = torch.abs(original_output - compiled_output)
            print(f"Max difference: {torch.max(diff)}")
            print(f"Mean difference: {torch.mean(diff)}")
            print(f"All differences: {diff}")
            
            # Check if it's just a scaling issue
            if torch.allclose(original_output, compiled_output * 2.0, atol=1e-5):
                print("⚠️  Possible scaling issue: compiled output is 2x the expected value")
            elif torch.allclose(original_output, compiled_output * 0.5, atol=1e-5):
                print("⚠️  Possible scaling issue: compiled output is 0.5x the expected value")
            
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()

def test_different_sizes():
    """Test with different matrix sizes to see if the issue is size-dependent"""
    print("\n=== Testing Different Sizes ===")
    
    sizes = [(256, 256), (2, 2), (3, 3), (4, 4), (8, 8), (16, 16)]
    
    for M, N in sizes:
        print(f"\n--- Testing {M}x{N} ---")
        
        class SizeTestModel(torch.nn.Module):
            def __init__(self, size):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(size[1], size[1]))
            
            def forward(self, x):
                x = torch.matmul(x, self.weight)
                x = F.relu(x)
                return x
        
        model = SizeTestModel((M, N)).cuda()
        input_tensor = torch.randn(M, N, device='cuda')
        
        original_output = model(input_tensor)
        
        try:
            compiled_model = compile(model, [input_tensor], 'src/fused_matmul_relu_kernel.cu')
            compiled_output = compiled_model(input_tensor)
            
            is_close = torch.allclose(original_output, compiled_output, atol=1e-5)
            status = "✅ PASSED" if is_close else "❌ FAILED"
            print(f"{M}x{N}: {status}")
            
            if not is_close:
                max_diff = torch.max(torch.abs(original_output - compiled_output))
                print(f"    Max diff: {max_diff:.6f}")
                
        except Exception as e:
            print(f"{M}x{N}: FAILED - {e}")

if __name__ == "__main__":
    debug_kernel()
    test_different_sizes() 