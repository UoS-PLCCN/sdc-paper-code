import itertools
from typing import List, Tuple

import numpy as np

from .logic.eval import LogicExpressionEvaluator


def logic_funcs_to_PBN_data(nodes: List[str], node_functions: List[Tuple[str, int]]):
    logic_eval = LogicExpressionEvaluator({})  # Don't need a value dict yet
    PBN_data = []

    for i, node in enumerate(nodes):
        # Input Mask
        input_mask = np.zeros(len(nodes), dtype=bool)
        for function, _ in node_functions[i]:
            symbols = logic_eval.get_symbols(function)
            for symbol in symbols:
                j = nodes.index(symbol)
                input_mask[j] = True

        # Truth Table
        if (node_functions[i][0][0] == "True"):
            truth_table = "True"
        elif (node_functions[i][0][0] == "False"):
            truth_table = "False"
        else:
            truth_table = np.zeros([2] * sum(input_mask))

        all_states = itertools.product([0, 1], repeat=sum(input_mask))
        input_nodes = np.array(nodes)[input_mask]

        for state in all_states:
            for function, prob in node_functions[i]:
                if ((function != "True") and (function != "False")):
                    logic_eval.dictionary = {
                        node: value for node, value in zip(input_nodes, state)
                    }
                    value = int(logic_eval.evaluate(function))
                    if value == 1:
                        truth_table[state] += prob

        if ((sum(input_mask) == 0) and (not ((node_functions[i][0][0] == "True") or (node_functions[i][0][0] == "False")))):
            control = True
        else:
            control = False

        PBN_data.append((input_mask, truth_table, node, control))

    return PBN_data

def slave_logic_funcs_to_PBN_data(nodes: List[str], node_functions: List[Tuple[str, int]]):
    logic_eval = LogicExpressionEvaluator({})  # Don't need a value dict yet
    PBN_data = []

    i = 0
    for node in nodes:

        if (node.startswith("y") or node.startswith("u")):
            # Input Mask
            input_mask = np.zeros(len(nodes), dtype=bool)
            for function, _ in node_functions[i]:
                symbols = logic_eval.get_symbols(function)
                for symbol in symbols:
                    j = nodes.index(symbol)
                    input_mask[j] = True

            # Truth Table
            truth_table = np.zeros([2] * sum(input_mask))
            all_states = itertools.product([0, 1], repeat=sum(input_mask))
            input_nodes = np.array(nodes)[input_mask]

            for state in all_states:
                for function, prob in node_functions[i]:
                    logic_eval.dictionary = {
                        node: value for node, value in zip(input_nodes, state)
                    }
                    value = int(logic_eval.evaluate(function))
                    if value == 1:
                        truth_table[state] += prob

            control = sum(input_mask) == 0

            PBN_data.append((input_mask, truth_table, node, control))

            i = i + 1

    return PBN_data