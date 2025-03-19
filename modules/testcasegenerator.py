from dataclasses import dataclass
from hypothesis import strategies as st
from hypothesis import given, settings
import inspect
from typing import Dict, Any, List, Tuple


@dataclass
class Variable:
    name: str

@dataclass
class Operation:
    operator: str
    left: Any
    right: Any

@dataclass
class Relation:
    negation: bool
    expression_follow_up: Any  # Variable (e.g., "followUp.x")
    operator: str
    expression_source: Any  # Expression (e.g., "source.x * -1")

@dataclass
class RelationGroup:
    relations: List[Relation]
    connective: str

@dataclass
class MR:
    R: RelationGroup
    Rf: RelationGroup

@dataclass
class MRSet:
    mr_list: List[MR]

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

if __name__ == "__main__":

    # 1. Define a sample MR (followUp.x = source.x * -1)
    sample_mr = MR(
        R=RelationGroup(
            relations=[
                Relation(
                    negation=False,
                    expression_follow_up=Variable("followUp.x"),
                    operator="==",
                    expression_source=Operation(
                        operator="*",
                        left=Variable("source.x"),
                        right=-1
                    )
                )
            ],
            connective="AND"
        ),
        Rf=RelationGroup(relations=[], connective="AND")  # Simplified for example
    )

    mrset = MRSet(mr_list=[sample_mr])

    # 2. Define a sample function to test
    def sample_function(x: int):
        return x ** 2

    # 3. Generate test cases
    generator = AEMRTestCaseGenerator(mrset, sample_function)
    
    test_cases = generator.generate_test_cases()

    # 4. Print generated test cases
    for source, followup in test_cases[:10]:  # Print first 5
        print(f"Source: {source}, Follow-up: {followup}")

        