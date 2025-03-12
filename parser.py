import json
from jsonschema import validate
from dataclasses import dataclass
from typing import List, Union

# Load and Validate Parser
def load_mrdl(json_path, schema_path):
    with open(json_path) as f:
        mrdl_data = json.load(f)
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

# Parser
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
        
mrdl_data = load_mrdl("example2.mrdl.json", "mrdl_schema.json")
parser = MRDLParser()
mrset = parser.parse_mrset(mrdl_data)

# Access parsed data
mr = mrset.mr_list[0]
print(mr.R.relations[0].expression_source)  # Operation(operator='*', left=Variable('source.x'), right=-1)