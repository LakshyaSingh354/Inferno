import inferno_relu

_KERNELS = {
    "relu": inferno_relu.relu,
    "relu_vec4": inferno_relu.relu_vec4,
}

def kernel(op_name):
    def decorator(f):
        def wrapper(*args, **kwargs):
            # We assume: args = [input_tensor, output_tensor]
            return _KERNELS[op_name](*args, **kwargs)
        return wrapper
    return decorator