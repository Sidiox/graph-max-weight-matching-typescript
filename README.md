# Maximum weight matching in non-bipartite graphs in TypeScript
Experiment in translation NetworkX source code https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html of an algorithm from Zvi Galil to TypeScript

This repo contains:
- `max_weight.py`: extracted source code from NetworkX
- `max_weight_minimal.py`: minimized version of the NetworkX implementation with type annotations
- `max-weight-matching.ts`: TypeScript version of the algorithm
- `chat-transcripts`: folder with the original LLM chat transcripts
- `transcrypt-attempt`: folder with generated code by [transcrypt](https://www.transcrypt.org/)
- `nx_vs_pure_python.py`: test script to ensure my NetworkX code extraction went well, also used to dump graphs for TypeScript testing
- `ts_vs_py.test.ts`: test script to ensure the TypeScript implementation matches the Python solutions, compares solution and cost

## Using the implementation
```typescript
import { Graph } from './max-weight-matching.ts';
import { maxWeightMatching } from './max-weight-matching.ts';
if (require.main === module) {
    const g = new Graph();
    console.log("Graph:");
    const edges: [string, string, number][] = [
        ["A", "B", 6],
        ["A", "C", 2],
        ["B", "C", 1],
        ["B", "D", 2],
        ["C", "E", 9],
        ["D", "E", 3],
    ];

    g.addWeightedEdgesFrom(edges);
    const res = maxWeightMatching(g);
    console.log("TypeScript Result:");
    console.log(res);
}
```


## Testing NetworkX vs Python vs TypeScript
To test the Python vs TypeScript implementations:
```
python nx_vs_pure_python.py
```
Yields 3 json files filled with 10 graphs.
Which can then be tested against TypeScript:
```
bun test ts_vs_py.test.ts
```

# References
- [Zvi Galil, Efficient algorithms for finding maximum matching in graphs, ACM Computing Surveys, 1986.](https://dl.acm.org/doi/10.1145/6462.6502)
- [NetworkX source](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html)
- [Joris van Rantwijk on Maximum Weighted Matching](https://jorisvr.nl/article/maximum-matching)
- [Blog post for this repo](https://portegi.es/blog/translating-max-weight-graph-matching-llms)