/**
 * https://bun.sh/docs/cli/test 
 */
import { Graph } from './gemini-2.5-pro-max-weight.ts';
import { maxWeightMatching } from './gemini-2.5-pro-max-weight.ts';
import { expect, test } from "bun:test";

// function to compare the result set against the reference set
function compareResults(res, expected: Set<[string, string]>): boolean {
    let match = true;
    if (res.size !== expected.size) {
        match = false;
    } else {
        // Convert Set<[string,string]> to something comparable like Set<string> ("u,v")
        const resComparable = new Set(Array.from(res).map(p => `${p[0]},${p[1]}`.split(',').sort().join(',')));
        const expComparable = new Set(Array.from(expected).map(p => `${p[0]},${p[1]}`.split(',').sort().join(',')));

        for (const item of resComparable) {
            if (!expComparable.has(item)) {
                match = false;
                break;
            }
        }
    }
    return match;
}

async function runTests(filePath: string) {
    const file = Bun.file(filePath);
    const contents = await file.json();

    var results = [];
    var counter = 0;
    for (const item of contents) {
        const expected = new Set<[string, string]>();

        const refedges: [string, string, number][] = item.result;
        for (const edge of refedges) {
            expected.add([edge[0], edge[1]]);
        }
        const refCost = Number(item.cost);

        const g = new Graph();
        const edges: [string, string, number][] = item.edges;
        g.addWeightedEdgesFrom(edges);

        var match = false;
        var message = "";
        try {
            const res = maxWeightMatching(g);

            // now we also want to know the weight of the answer
            var resultCost = 0;
            for (const resEdge of res) {
                console.log(resEdge);
                for (const edge of edges) {
                    if ((edge[0] === resEdge[0] && edge[1] === resEdge[1]) || (edge[1] === resEdge[0] && edge[0] === resEdge[1])) {
                        resultCost += Number(edge[2]);
                    }
                }
            }
            // console.log("TypeScript Result:");
            // console.log(res);
            match = compareResults(res, expected);
            // expect(match).toBe(true);
            if (!match && resultCost != refCost) {
                // bad
                const resComparable = Array.from(res).map(p => `${p[0]},${p[1]}`.split(',').sort().join(','));
                const expComparable = Array.from(expected).map(p => `${p[0]},${p[1]}`.split(',').sort().join(','));
                message = `${resComparable} vs ${expComparable}`;
            } else if (!match && resultCost == refCost) {
                // not that bad
                message = "Cost matched";
                match = true;
            }
        } catch (e) {
            console.warn(e);
            message = e.message;
        }

        results.push({
            index: counter,
            edges: item.edges,
            result: match,
            message: message,
        });
        counter += 1;
    }

    var trueCount = 0;
    var falseCount = 0;
    for (const result of results) {
        if (result.result) {
            trueCount++;
        } else {
            falseCount++;
            // console.log(`{index: ${result.index} result: ${result.result}, message: ${result.message}`);
        }
    }
    console.log(`Matching Python Reference ${trueCount} Erroring:${falseCount}`);
}

test("Graph Tests for graph_int_medium.json", async () => {
    await runTests("./graph_int_medium.json");
});

test("Graph Tests for graph_float_large.json", async () => {
    await runTests("./graph_float_large.json");
});

test("Graph Tests for graph_float_medium.json", async () => {
    await runTests("./graph_float_medium.json");
});
