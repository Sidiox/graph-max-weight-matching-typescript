from networkx import Graph as XGraph
from networkx.algorithms.matching import max_weight_matching as nx_max_weight_matching
import random

from max_weight import max_weight_matching, Graph

def compare_matchings(num_tests=100, num_nodes_range=(4, 25), num_edges_range=(5, 20), weight_range=(1, 10)):
    """
    Compare the outputs of max_weight_matching (pure Python) and nx_max_weight_matching (NetworkX-based)
    over randomly generated graphs.
    """
    results = []

    for i in range(num_tests):
        num_nodes = random.randint(*num_nodes_range)
        num_edges = random.randint(*num_edges_range)
        weight_min, weight_max = weight_range

        # Generate a list of unique node pairs
        possible_edges = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)]
        edges = random.sample(possible_edges, min(num_edges, len(possible_edges)))
        weighted_edges = [(u, v, random.randint(weight_min, weight_max)) for u, v in edges]
        # convert into str, str, int
        weighted_edges = [(chr(u + 65), chr(v + 65), int(w)) for u, v, w in weighted_edges]


        # Construct Graph using your implementation
        g1 = Graph()
        g1.add_weighted_edges_from(weighted_edges)
        result1 = max_weight_matching(g1)
        normalized_result1 = {frozenset(edge) for edge in result1}

        # Construct Graph using NetworkX-compatible implementation
        g2 = XGraph()
        g2.add_weighted_edges_from(weighted_edges)
        result2 = nx_max_weight_matching(g2)
        normalized_result2 = {frozenset(edge) for edge in result2}

        if normalized_result1 != normalized_result2:
            print(f"Test {i + 1} FAILED")
            print("Edges:", weighted_edges)
            print("Pure Python:", result1)
            print("NetworkX:", result2)
            return  # Early exit on first failure
        else:
            print("Edges:", weighted_edges)
            print("Pure Python:", result1)
            print("NetworkX:", result2)

        results.append(
            {"edges": weighted_edges,
             "result": list(result1)
             }
        )

    print("All tests passed!")
    return results

if __name__ == "__main__":
    results = compare_matchings()

