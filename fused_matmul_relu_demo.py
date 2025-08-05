import torch
from inferno import InfernoCompiler

# ================================================================================
# DEMONSTRATION
# ================================================================================

# Create a global instance of our compiler
inferno = InfernoCompiler()

# --- Define a model using the pattern ---
@inferno.compile
def simple_fused_model(A, B):
    # This code will be analyzed by the decorator.
    # If the pattern is found, this body will be replaced.
    return torch.relu(torch.matmul(A, B))

# --- Define another model WITHOUT the pattern ---
@inferno.compile
def simple_unfused_model(A, B):
    x = torch.matmul(A, B)
    # Some other operation in between
    # x = x * 2.0 
    y = torch.relu(x)
    return y


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" " * 10 + "INFERNO COMPILER DECORATOR DEMO")
    print("="*60)

    # --- Setup ---
    A = torch.randn(256, 512, device='cuda')
    B = torch.randn(512, 128, device='cuda')

    # --- Test the compiled function ---
    print("\n[1] Testing the model that SHOULD be fused...")
    
    # When we call this, we're actually calling the `compiled_wrapper`
    # that our decorator returned.
    output_fused = simple_fused_model(A, B)
    
    # For verification, let's call the PyTorch version directly
    output_pytorch = torch.relu(torch.matmul(A, B))
    
    is_close = torch.allclose(output_fused, output_pytorch, atol=1e-5)
    status = "✅ PASSED" if is_close else "❌ FAILED"
    print(f"\n    Verification against PyTorch: {status}")
    print(f"    Output shape: {output_fused.shape}")


    # --- Test the un-compiled function ---
    print("\n[2] Testing the model that should NOT be fused...")
    
    # This should have returned the original function
    output_unfused = simple_unfused_model(A, B)
    
    # Verification
    output_pytorch_unfused = torch.relu(torch.matmul(A, B) * 2.0)
    
    is_close_unfused = torch.allclose(output_unfused, output_pytorch_unfused, atol=1e-5)
    status_unfused = "✅ PASSED" if is_close_unfused else "❌ FAILED"
    print(f"\n    Verification against PyTorch: {status_unfused}")
    print(f"    Output shape: {output_unfused.shape}")
    
    print("\n" + "="*60)

