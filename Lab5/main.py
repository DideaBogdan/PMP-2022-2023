from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt


#Ex1-Model
bet_model = BayesianNetwork(
    [
        ("c1", "c2"),
        ("c1", "p1"),
        ("c1", "p3"),
        ("c2", "p2"),
        ("p1", "p2"),
        ("p1", "p3"),
        ("p2", "p3")
    ]
)

CPD_c1 = TabularCPD(variable='c1', variable_card=5, values=[[1/5], [1/5], [1/5], [1/5], [1/5]])
print(CPD_c1)

CPD_c2 = TabularCPD(
    variable='c2',
    variable_card=5,
    values=[[0, 1/4, 1/4, 1/4, 1/4], [1/4, 0, 1/4, 1/4, 1/4], [1/4, 1/4, 0, 1/4, 1/4], [1/4, 1/4, 1/4, 0, 1/4], [1/4, 1/4, 1/4, 1/4, 0]],
    evidence=['c1'],
    evidence_card=[5]
)
print(CPD_c2)

CPD_p1 = TabularCPD(
    variable='p1',
    variable_card=2,
    values=[[0, 1/4, 2/4, 3/4, 1], [1, 3/4, 2/4, 1/4, 0]],
    evidence=['c1'],
    evidence_card=[5]
)
print(CPD_p1)

CPD_p2 = TabularCPD(
    variable='p2',
    variable_card=2,
    values=[[0, 1/4, 2/4, 3/4, 1, 1, 3/4, 2/4, 1/4, 0], [1, 3/4, 2/4, 1/4, 0, 0, 1/4, 2/4, 3/4, 1]],
    evidence=['c2', 'p1'],
    evidence_card=[5, 2]
)
print(CPD_p2)


CPD_p3 = TabularCPD(
    variable='p3',
    variable_card=2,
    values=[[1, 1, 1, 1, 3/4, 3/4, 3/4, 3/4, 2/4, 2/4, 2/4, 2/4, 1/4, 1/4, 1/4, 1/4, 0, 0, 0, 0],
            [0, 0, 0, 0, 1/4, 1/4, 1/4, 1/4, 2/4, 2/4, 2/4, 2/4, 3/4, 3/4, 3/4, 3/4, 1, 1, 1, 1]],
    evidence=['c1', 'p1', 'p2'],
    evidence_card=[5, 2, 2]
)
print(CPD_p3)


bet_model.add_cpds(CPD_c1, CPD_c2)
bet_model.get_cpds()
bet_model.check_model()