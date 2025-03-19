import json
from jsonschema import validate
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional, Tuple
from hypothesis import strategies as st
from hypothesis import given, settings
import inspect
import importlib.util
import sys
import signal
from contextlib import contextmanager
from io import StringIO
import docker 

def load_mrdl(json_path, schema_path):
    with open(json_path) as f:
        mrdl_data = json.load(f) # json.load(f) converts json into python dict 
    with open(schema_path) as f:
        schema = json.load(f)
    validate(mrdl_data, schema)  # Validate against schema
    return mrdl_data

# Create dataclasses to represent MRDL Objects
@dataclass
class Variable:
    name: str  # e.g., "source.x"

@dataclass
class Operation:
    operator: str
    left: Union['Expression', int, float, str]
    right: Union['Expression', int, float, str]

@dataclass
class Quantifier:
    type: str  # "forall" or "exists"
    variable: str
    collection: str
    condition: 'Relation'

@dataclass
class Relation:
    negation: bool
    expression_follow_up: Union[Variable, Operation, Quantifier]
    operator: str
    expression_source: Union[Variable, Operation, Quantifier]

@dataclass
class RelationGroup:
    relations: List[Relation]
    connective: str  # "AND" or "OR"

@dataclass
class MR:
    R: RelationGroup
    Rf: RelationGroup

@dataclass
class MRSet:
    mr_list: List[MR]

# Parser, class MRDLParser is responsible for converting a JSON structure into Python objects
class MRDLParser:
    def parse_mrset(self, mrdl_data):
        mr_list = [self.parse_mr(mr_json) for mr_json in mrdl_data["mrSet"]]
        return MRSet(mr_list)

    def parse_mr(self, mr_json):
        R = self.parse_relation_group(mr_json["R"])
        Rf = self.parse_relation_group(mr_json["Rf"])
        return MR(R, Rf)

    def parse_relation_group(self, rg_json):
        relations = [self.parse_relation(rel) for rel in rg_json["relations"]]
        connective = rg_json.get("connective", "AND")
        return RelationGroup(relations, connective)

    def parse_relation(self, rel_json):
        negation = rel_json.get("negation", False)
        expr_follow_up = self.parse_expression(rel_json["expressionFollowUp"])
        operator = rel_json["operator"]
        expr_source = self.parse_expression(rel_json["expressionSource"])
        return Relation(negation, expr_follow_up, operator, expr_source)

    def parse_expression(self, expr_json):
        if isinstance(expr_json, (int, float)):
            # Handle numeric literals (e.g., -1, 3.14)
            return expr_json
        elif isinstance(expr_json, str):
            # Handle Variables (e.g., "source.x")
            return Variable(expr_json)
        elif isinstance(expr_json, dict):
            # Handle Operations or Quantifiers
            if "operator" in expr_json:
                return Operation(
                    expr_json["operator"],
                    self.parse_expression(expr_json["left"]),
                    self.parse_expression(expr_json["right"])
                )
            elif "type" in expr_json:
                return Quantifier(
                    expr_json["type"],
                    expr_json["variable"],
                    expr_json["collection"],
                    self.parse_relation(expr_json["condition"])
                )
            else:
                raise ValueError(f"Invalid expression: {expr_json}")
        else:
            raise ValueError(f"Unsupported expression type: {type(expr_json)}")
        

class AEMRTestCaseGenerator:
    def __init__(self, mrset: MRSet, func):
        self.mrset = mrset
        self.func = func
        self.param_names = inspect.getfullargspec(func).args  # Get function parameters

    def generate_test_cases(self) -> List[Tuple[Dict, Dict]]:
        """
        Generates source and follow-up test cases for all MRs in the MRSet.
        Returns a list of tuples: (source_input_dict, followup_input_dict)
        """
        test_cases = []

        for mr in self.mrset.mr_list:
            # Assume each MR.R has exactly one relation for simplicity
            relation = mr.R.relations[0]
            
            # Get Hypothesis strategy for source inputs
            source_strategy = self._get_source_strategy()
            
            # Generate source inputs using Hypothesis
            @given(source_strategy)
            @settings(max_examples=100)
            def _generate(source_input):
                followup_input = self._apply_relation(relation, source_input)
                test_cases.append((source_input, followup_input))

            _generate()  # Trigger Hypothesis generation

        return test_cases

    def _get_source_strategy(self) -> st.SearchStrategy:
        """
        Create a Hypothesis strategy for generating valid source inputs.
        Defaults to integer values for all parameters.
        """
        strategies = {}
        for param in self.param_names:
            strategies[param] = st.integers()
        return st.fixed_dictionaries(strategies)

    def _apply_relation(self, relation: Relation, source_input: Dict) -> Dict:
        """
        Apply the MR's R relation to generate follow-up input.
        """
        followup_input = {}

        # Evaluate expression_source for each parameter
        for param in self.param_names:
            source_var = Variable(f"source.{param}")
            followup_var = Variable(f"followUp.{param}")
            
            # Check if this parameter is transformed in the relation
            if relation.expression_follow_up.name == followup_var.name:
                # Evaluate the expression using source input values
                followup_value = self._evaluate_expression(
                    relation.expression_source, 
                    source_input
                )
                followup_input[param] = followup_value
            else:
                # Copy unchanged parameter
                followup_input[param] = source_input[param]

        return followup_input

    def _evaluate_expression(self, expr: Any, source_input: Dict) -> Any:
        """
        Recursively evaluate an expression using source input values.
        """
        # Handle numeric literals (int/float)
        if isinstance(expr, (int, float)):
            return expr
        # Handle Variables
        elif isinstance(expr, Variable):
            param = expr.name.split(".")[1]
            return source_input[param]
        # Handle Operations
        elif isinstance(expr, Operation):
            left = self._evaluate_expression(expr.left, source_input)
            right = self._evaluate_expression(expr.right, source_input)
            
            if expr.operator == '+':
                return left + right
            elif expr.operator == '-':
                return left - right
            elif expr.operator == '*':
                return left * right
            elif expr.operator == '/':
                return left / right
            else:
                raise ValueError(f"Unsupported operator: {expr.operator}")
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")


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

if __name__ == "__main__":
    # Step 1: Load and parse MRDL
    mrdl_data = load_mrdl("example.mrdl.json", "mrdl_schema.json")
    parser = MRDLParser()
    mrset = parser.parse_mrset(mrdl_data)
    # print("----------------------MR SET---------------------------")
    # print(mrset)

    # Step 2: Initialize the Code Instrumentation Engine
    engine = CodeInstrumentationEngine(  # <-- This was missing!
        target_module_path="program.py",
        function_name="square"
    )

    # Step 3: Initialize the Test Case Generator with the engine's function
    generator = AEMRTestCaseGenerator(mrset, engine.function)

    # Step 4: Generate test cases
    test_cases = generator.generate_test_cases()
    # print("--------------------TEST CASES----------------------------")
    # print(test_cases)

    # Step 5: Execute tests and check MRs
    for source_input, followup_input in test_cases:
        # Execute source test case
        source_result = engine.execute_test_case(source_input)
        if source_result.status != "pass":
            print(f"Source test failed: {source_input} → {source_result.error}")
            continue

        # Execute follow-up test case
        followup_result = engine.execute_test_case(followup_input)
        if followup_result.status != "pass":
            print(f"Follow-up test failed: {followup_input} → {followup_result.error}")
            continue

        # Verify MR (example: followup.output == source.output)
        if followup_result.output != source_result.output:
            print(f"MR Violated: {source_input} → {followup_input}")