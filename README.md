# Maximum weight matching in non-bipartite graphs in TypeScript
Experiment in translation NetworkX source code https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html of an algorithm from Zvi Galil to TypeScript

This repo contains:
- `max_weight.py`: extracted source code from NetworkX
- `max_weight_minimal.py`: minimized version of the NetworkX implementation with type annotations
- `max-weight-matching.ts`
- `chat-transcripts`: folder with the original LLM chat transcripts
- `transcrypt-attempt`: folder with generated code by [transcrypt](https://www.transcrypt.org/)
- `nx_vs_pure_python.py`: test script to ensure my NetworkX code extraction went well, also used to dump graphs for TypeScript testing
- `ts_vs_py.test.ts`: test script to ensure the TypeScript implementation matches the Python solutions, compares solution and cost


# References
- [Zvi Galil, Efficient algorithms for finding maximum matching in graphs, ACM Computing Surveys, 1986.](https://dl.acm.org/doi/10.1145/6462.6502)
- [NetworkX source](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html)
- [Joris van Rantwijk on Maximum Weighted Matching](https://jorisvr.nl/article/maximum-matching)
- [Blog post for this repo](https://portegi.es/blog/translating-max-weight-graph-matching-llms)