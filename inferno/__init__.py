import inspect
import ast
import inferno_fused

class InfernoCompiler:
    """
    The main class for the Inferno compiler.
    This will eventually house the full compilation pipeline. For now,
    it holds our pattern-matching decorator.
    """

    def __init__(self):
        # This could eventually hold compiler flags, registered patterns, etc.
        pass

    def compile(self, fn):
        """
        The decorator that acts as our pattern-matching compiler stub.
        It inspects the AST of the decorated function for a specific pattern
        and replaces it with a call to a fused kernel if found.
        """
        print(f"--- [Inferno] Analyzing function: {fn.__name__} ---")
        
        # 1. Get the source code of the function
        source_code = inspect.getsource(fn)
        
        # 2. Parse the source code into an Abstract Syntax Tree (AST)
        tree = ast.parse(source_code)

        # 3. Analyze the tree for our target pattern
        analyzer = FusedGemmReluAnalyzer()
        analyzer.visit(tree)

        if analyzer.pattern_found:
            print(f"    ✅ Pattern Found: torch.relu(torch.matmul(...))")
            print(f"    🔥 Replacing with fused kernel: inferno_fused.fused_gemm_relu")

            # 4. If the pattern is found, return a new function that *only*
            #    calls our fused kernel. This is the "compilation" step.
            #    The original Python code is discarded.
            def compiled_wrapper(*args, **kwargs):
                # We assume the arguments match the matmul call, e.g., (A, B)
                return inferno_fused.fused_gemm_relu(*args, **kwargs)
            
            return compiled_wrapper
        else:
            print(f"    ❌ Pattern Not Found. Returning original function.")
            # If no specific pattern is found, just return the original function
            return fn


class FusedGemmReluAnalyzer(ast.NodeVisitor):
    """
    An AST NodeVisitor that walks the syntax tree to find the specific
    pattern: a `relu` call whose argument is a `matmul` call.
    """
    def __init__(self):
        self.pattern_found = False

    def visit_Call(self, node):
        # This method is called for every function call in the code.
        
        # Is this a call to `torch.relu`?
        # We check if the function being called is an attribute (`relu`) of an
        # attribute (`torch`) of a name. A bit complex, but robust.
        is_relu_call = (
            isinstance(node.func, ast.Attribute) and
            node.func.attr == 'relu' and
            isinstance(node.func.value, ast.Name) and
            node.func.value.id == 'torch'
        )

        if is_relu_call:
            # OK, we found a relu call. Now check its argument.
            # We expect exactly one argument.
            if len(node.args) == 1 and isinstance(node.args[0], ast.Call):
                inner_call_node = node.args[0]
                
                # Is the inner call a `torch.matmul`?
                is_matmul_call = (
                    isinstance(inner_call_node.func, ast.Attribute) and
                    inner_call_node.func.attr == 'matmul' and
                    isinstance(inner_call_node.func.value, ast.Name) and
                    inner_call_node.func.value.id == 'torch'
                )

                if is_matmul_call:
                    # We found our exact pattern!
                    self.pattern_found = True
        
        # Continue walking the tree for other nodes
        self.generic_visit(node)


