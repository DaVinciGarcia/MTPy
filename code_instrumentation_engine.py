import importlib.util
from io import StringIO
import sys
import signal
from contextlib import contextmanager
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import docker  # For sandboxing (optional)

@dataclass
class ExecutionResult:
    status: str  # "pass", "fail", "timeout"
    output: Any
    error: Optional[str] = None
    logs: Optional[str] = None  # Captured stdout/stderr

class CodeInstrumentationEngine:
    def __init__(self, target_module_path: str, function_name: str):
        self.target_module_path = target_module_path
        self.function_name = function_name
        self.function = self._load_function()

    def _load_function(self):
        """Dynamically load the target function from the module."""
        spec = importlib.util.spec_from_file_location("target_module", self.target_module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["target_module"] = module
        spec.loader.exec_module(module)
        return getattr(module, self.function_name)

    @contextmanager
    def _timeout(self, seconds: int):
        """Timeout context manager to prevent infinite loops."""
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timeout after {seconds} seconds")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    def _capture_output(self, func, *args, **kwargs) -> Tuple[Any, str]:
        """Capture stdout/stderr during execution."""
        original_stdout = sys.stdout
        sys.stdout = captured_stdout = StringIO()
        try:
            result = func(*args, **kwargs)
            logs = captured_stdout.getvalue()
            return result, logs
        finally:
            sys.stdout = original_stdout

    def execute_test_case(
        self, 
        input_data: Dict[str, Any], 
        timeout: int = 5,
        sandbox: bool = False
    ) -> ExecutionResult:
        """
        Execute the function under test with the given input.
        - `sandbox`: If True, run in a Docker container (for untrusted code).
        """
        if sandbox:
            return self._execute_in_docker(input_data, timeout)
        
        try:
            # Capture output and enforce timeout
            with self._timeout(timeout):
                output, logs = self._capture_output(self.function, **input_data)
            return ExecutionResult("pass", output, logs=logs)
        except TimeoutError as e:
            return ExecutionResult("timeout", None, error=str(e))
        except Exception as e:
            return ExecutionResult("fail", None, error=str(e), logs=logs)

    def _execute_in_docker(self, input_data: Dict, timeout: int) -> ExecutionResult:
        """Run the code in an isolated Docker container (for security)."""
        client = docker.from_env()
        try:
            # Mount the code and run in a minimal Python container
            container = client.containers.run(
                "python:3.9-slim",
                command=f"python -c 'from target_module import {self.function_name}; print({self.function_name}(**{input_data}))'",
                volumes={self.target_module_path: {"bind": "/app/target_module.py", "mode": "ro"}},
                working_dir="/app",
                detach=True,
                mem_limit="100m",  # Limit memory
            )
            container.wait(timeout=timeout)
            logs = container.logs().decode("utf-8")
            output = eval(logs.strip())  # Extract output from logs
            return ExecutionResult("pass", output, logs=logs)
        except docker.errors.ContainerError as e:
            return ExecutionResult("fail", None, error=str(e), logs=e.logs.decode())
        except Exception as e:
            return ExecutionResult("error", None, error=str(e))

    def reset_state(self):
        """Reset global state between test cases (if applicable)."""
        # Re-import the module to reset global variables
        self.function = self._load_function()

# Load the engine
engine = CodeInstrumentationEngine(
    target_module_path="path/to/program.py",
    function_name="square"
)

# Execute a test case
result = engine.execute_test_case({"x": 5})
print(result)  # ExecutionResult(status='pass', output=25, error=None, logs='')

# Reset state between tests (if needed)
engine.reset_state()
