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
class Function:
    operator: str  # e.g., "LEN"
    list: str      # e.g., "source.input"

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
            return expr_json
        elif isinstance(expr_json, str):
            return Variable(expr_json)
        elif isinstance(expr_json, dict):
            # Handle Functions (e.g., LEN)
            if "operator" in expr_json and "list" in expr_json:
                return Function(
                    operator=expr_json["operator"],
                    list=expr_json["list"]
                )
            # Handle Operations (e.g., +, ⊆)
            elif "operator" in expr_json:
                return Operation(
                    operator=expr_json["operator"],
                    left=self.parse_expression(expr_json["left"]),
                    right=self.parse_expression(expr_json["right"])
                )
            # Handle Quantifiers (∀, ∃)
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
        Generates lists for parameters named "lst", "arr", or "list".
        """
        strategies = {}
        list_params = {"lst", "arr", "list"}  # Add more names as needed
        for param in self.param_names:
            if param.lower() in list_params:
                # Generate non-empty lists of integers
                strategies[param] = st.lists(st.integers(), min_size=1)
            else:
                strategies[param] = st.integers()  # Default to integers
        return st.fixed_dictionaries(strategies)

    def _apply_relation(self, relation: Relation, source_input: Dict) -> Dict:
        """
        Generate follow-up input using the R relation.
        """
        followup_input = {}
        context = {f"source.{param}": value for param, value in source_input.items()}
        
        for param in self.param_names:
            followup_var = f"followUp.{param}"
            if relation.expression_follow_up.name == followup_var:
                # Evaluate the expression using the context (source.x, etc.)
                followup_value = self._evaluate_expression(
                    relation.expression_source,
                    context
                )
                followup_input[param] = followup_value
            else:
                followup_input[param] = source_input[param]
        
        return followup_input


    def _evaluate_expression(self, expr: Any, context: Dict[str, Any]) -> Any:

        """
        Evaluate an expression with access to variables in `context`.
        - `context`: A dictionary with keys like `source.x` or `source.output`.
        """

        if isinstance(expr, (int, float)):
            return expr  # Return numeric literals directly
        elif isinstance(expr, Variable):
            # Handle variables like "source.x" or "source.output"
            parts = expr.name.split(".")
            if parts[0] == "source" or parts[0] == "followUp":
                return context.get(expr.name)  # Get from context
            else:
                raise ValueError(f"Invalid variable: {expr.name}")
        elif isinstance(expr, Function):
            if expr.operator == "LEN":
                # Evaluate the list and return its length
                list_value = context.get(expr.list)
                return len(list_value)
            else:
                raise ValueError(f"Unsupported function: {expr.operator}")
        elif isinstance(expr, Operation):
            # Recursively evaluate left and right with the same context
            left = self._evaluate_expression(expr.left, context)
            right = self._evaluate_expression(expr.right, context)
            
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
            
    def verify_mr(self, source_output: Any, followup_output: Any, rf_relation: RelationGroup) -> bool:

        """
        Verify if the follow-up output satisfies the Rf relation.
        """

        # Assume one relation in Rf for simplicity
        relation = rf_relation.relations[0]
        
        # Context includes source.output and followUp.output
        context = {
            "source.output": source_output,
            "followUp.output": followup_output
        }
        
        # Evaluate the expected follow-up output from the Rf relation
        expected_followup_output = self._evaluate_expression(
            relation.expression_source,
            context
        )
        
        # Apply the operator to check the condition
        return self._apply_operator(
            followup_output,
            relation.operator,
            expected_followup_output
        )
    
    def _apply_operator(self, left: Any, operator: str, right: Any) -> bool:
        """
        Evaluate a comparison operator (e.g., "==", ">=").
        """
        if operator == "==":
            return left == right
        elif operator == ">=":
            return left >= right
        elif operator == "⊆":
            return set(left).issubset(set(right))
        # Add other operators as needed
        else:
            raise ValueError(f"Unsupported operator: {operator}")


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
        original_stdout = sys.stdout
        sys.stdout = captured_stdout = StringIO()
        logs = ""  # Initialize logs here
        try:
            result = func(*args, **kwargs)
            logs = captured_stdout.getvalue()
            return result, logs
        except Exception as e:
            logs = captured_stdout.getvalue()
            raise e
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
    # "/home/leonardo/Documentos/MTPy/MTPy/MRDLs/example.mrdl.json"
    # "/home/leonardo/Documentos/MTPy/MTPy/MRDLs/cube.mrdl.json"

    mrdl_data = load_mrdl("/home/leonardo/Documentos/MTPy/MTPy/MRDLs/robustsorter.mrdl.json", 
                          "/home/leonardo/Documentos/MTPy/MTPy/MRDLs/mrdl_schema.json")
    
    parser = MRDLParser()
    mrset = parser.parse_mrset(mrdl_data)
    
    # Step 2: Initialize the Code Instrumentation Engine
    # "/home/leonardo/Documentos/MTPy/MTPy/SUTs/program.py"
    # "/home/leonardo/Documentos/MTPy/MTPy/SUTs/cube.py"

    engine = CodeInstrumentationEngine(  
        target_module_path="/home/leonardo/Documentos/MTPy/MTPy/SUTs/merge_sort.py",
        function_name="mergesort"
    )

    # Step 3: Initialize the Test Case Generator with the engine's function
    generator = AEMRTestCaseGenerator(mrset, engine.function)

    # Step 4: Generate test cases
    test_cases = generator.generate_test_cases()

    # Step 5: Execute tests and check MRs
    # Inside your test workflow loop:
    for source_input, followup_input in test_cases:
        # Execute source test case
        print(source_input == followup_input)

        source_result = engine.execute_test_case(source_input)
        if source_result.status != "pass":
            print(f"Source test failed: {source_input} → {source_result.error}")
            continue

        # Execute follow-up test case
        followup_result = engine.execute_test_case(followup_input)
        if followup_result.status != "pass":
            print(f"Follow-up test failed: {followup_input} → {followup_result.error}")
            continue

        # Verify MR using the parsed Rf relation
        mr = mrset.mr_list[0]  # Assuming one MR for simplicity
        is_valid = generator.verify_mr(
            source_result.output,
            followup_result.output,
            mr.Rf
        )
        
        if not is_valid:
            print(f"MR Violated: {source_input} → {followup_input}")