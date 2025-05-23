function assert(condition, message) {
    if (!condition) {
        throw new Error(message || "Assertion failed");
    }
}

/**
 * We are translating a very complicated algorithm
 * We have a source code in Python.
 * I am providing you iterative parts of the
 * maxWeightMatching function
 * that have to be translated matching the Python
 * as close as possible, but making sure the JS is correct
 */
class Graph {
    constructor() {
        this.adj = new Map();
        this.nodeList = new Set();
    }

    addEdge(u, v, weight = 1) {
        if (!this.adj.has(u)) {
            this.adj.set(u, new Map());
            this.nodeList.add(u);
        }
        if (!this.adj.has(v)) {
            this.adj.set(v, new Map());
            this.nodeList.add(v);
        }
        this.adj.get(u).set(v, weight);
        this.adj.get(v).set(u, weight); // undirected graph
    }

    neighbors(u) {
        return [...(this.adj.get(u)?.keys() ?? [])];
    }

    *edges() {
        const seen = new Set();
        for (let [u, neighbors] of this.adj.entries()) {
            for (let [v, weight] of neighbors.entries()) {
                const key = [u, v].sort().join(',');
                if (!seen.has(key)) {
                    seen.add(key);
                    yield [u, v, { weight }];
                }
            }
        }
    }

    nodes() {
        return Array.from(this.nodeList);
    }

    get(u) {
        return this.adj.get(u) || new Map();
    }

    // Enables g[u][v] syntax using Proxy
    static withProxy(graph) {
        return new Proxy(graph, {
            get(target, prop) {
                if (typeof prop === 'string' && target.adj.has(prop)) {
                    return new Proxy({}, {
                        get(_, v) {
                            return target.adj.get(prop).get(v);
                        }
                    });
                }
                return target[prop];
            }
        });
    }

    static maxWeightMatching(G, maxcardinality = false, weight = "weight") {
        class NoNode {
            // Dummy value which is different from any node.
        }

        class Blossom {
            // Representation of a non-trivial blossom or sub-blossom.
            constructor() {
                this.childs = [];
                this.edges = [];
                this.mybestedges = [];
            }

            *leaves() {
                const stack = [...this.childs];
                while (stack.length > 0) {
                    const t = stack.pop();
                    if (t instanceof Blossom) {
                        stack.push(...t.childs);
                    } else {
                        yield t;
                    }
                }
            }
        }

        const gnodes = G.nodes();
        if (gnodes.length === 0) {
            return new Set();
        }

        let maxweight = 0;
        let allinteger = true;
        for (let [i, j, d] of G.edges()) {

            let wt = d['weight']
            if (i !== j && wt > maxweight) {
                maxweight = wt;
                allinteger = allinteger && (typeof wt === 'number' && Number.isInteger(wt));
            }
        }

        let mate = {};
        let label = new Map();
        let labeledge = new Map(); // Use Map for labeledge
        let inblossom = Object.fromEntries(G.nodes().map(node => [node, node]));
        let blossomparent = Object.fromEntries(G.nodes().map(node => [node, null]));
        let blossombase = Object.fromEntries(G.nodes().map(node => [node, node]));
        let bestedge = new Map();
        let dualvar = Object.fromEntries(G.nodes().map(node => [node, maxweight]));
        console.log(`dualvar=${JSON.stringify(dualvar)}`);
        let blossomdual = {};
        let allowedge = {};
        let queue = [];
        function slack(v, w) {
            const dualvar_v = dualvar[v];
            const dualvar_w = dualvar[w];
            const weight = 2 * G.get(v).get(w);
            console.log(`dualvar_v=${dualvar_v} dualvar_w=${dualvar_w} weight=${weight}`);
            return dualvar_v + dualvar_w - weight;
        }

        function assignLabel(w, t, v) {
            const b = inblossom[w];
            // assert(label.get(w) === undefined && label.get(b) === undefined);
            label.set(w, t)
            label.set(b, t)
            if (v !== undefined) {
                labeledge.set(w, [v, w]);
                labeledge.set(b, [v, w]);
            } else {
                labeledge.set(w, null);
                labeledge.set(b, null);
            }
            bestedge.set(w, null);
            bestedge.set(b, null);
            if (t === 1) {
                if (b instanceof Blossom) {
                    queue.push(...b.leaves());
                } else {
                    queue.push(b);
                }
            } else if (t === 2) {
                const base = blossombase[b];
                assignLabel(mate[base], 1, base);
            }
        }
        function scanBlossom(v, w) {
            let path = [];
            let base = new NoNode();

            while (v !== new NoNode()) {
                console.log(`v: ${v}`)
                
                const b = inblossom[v];
                console.log(`label b: ${label.get(b)}`);
                if (label.get(b) & 4) {
                    base = blossombase[b];
                    break;
                }
                console.log(`b: ${b}`);
                assert(label.get(b) === 1);
                path.push(b);
                label.set(b, 5);
                if (labeledge.get(b) === null) {
                    assert(blossombase[b] !== mate[blossombase[b]]);
                    v = new NoNode();
                } else {
                    assert(labeledge.get(b)[0] === mate[blossombase[b]]);
                    v = labeledge.get(b)[0];
                    const b2 = inblossom[v];
                    assert(label.get(b2) === 2);
                    v = labeledge.get(b2)[0];
                }
            }

            if (w !== new NoNode()) {
                [v, w] = [w, v];
            }

            for (const b of path) {
                label.set(b, 1);
            }

            return base;
        }

        function addBlossom(base, v, w) {
            var bb = inblossom[base];
            var bv = inblossom[v];
            var bw = inblossom[w];
            var b = new Blossom();
            blossombase[b] = base;
            blossomparent[b] = null;
            blossomparent[bb] = b;
            b.childs = [];
            b.edges = [(v, w)];

            while (bv !== bb) {
                blossomparent[bv] = b;
                b.childs.push(bv);
                b.edges.push(labeledge[bv]);
                assert(label[bv] === 2 || (label[bv] === 1 && labeledge[bv][0] === mate[blossombase[bv]]));
                v = labeledge[bv][0];
                bv = inblossom[v];
            }

            b.childs.push(bb);
            b.childs.reverse();
            b.edges.reverse();

            while (bw !== bb) {
                blossomparent[bw] = b;
                b.childs.push(bw);
                b.edges.push([labeledge[bw][1], labeledge[bw][0]]);
                assert(label[bw] === 2 || (label[bw] === 1 && labeledge[bw][0] === mate[blossombase[bw]]));
                w = labeledge[bw][0];
                bw = inblossom[w];
            }

            assert(label[bb] === 1);
            label[b] = 1;
            labeledge[b] = labeledge[bb];
            blossomdual[b] = 0;

            for (const v of b.leaves()) {
                if (label[inblossom[v]] === 2) {
                    queue.push(v);
                }
                inblossom[v] = b;
            }

            const bestedgeto = {};
            for (const bv of b.childs) {
                let nblist;
                if (bv instanceof Blossom) {
                    if (bv.mybestedges !== null) {
                        nblist = bv.mybestedges;
                        bv.mybestedges = null;
                    } else {
                        nblist = [...bv.leaves()].flatMap(v => G.neighbors(v).filter(w => v !== w).map(w => [v, w]));
                    }
                } else {
                    nblist = G.neighbors(bv).filter(w => bv !== w).map(w => [bv, w]);
                }

                for (var [i, j] of nblist) {
                    if (inblossom[j] === b) {
                        [i, j] = [j, i];
                    }
                    const bj = inblossom[j];
                    if (bj !== b && label.get(bj) === 1 && (!bestedgeto[bj] || slack(i, j) < slack(...bestedgeto[bj]))) {
                        bestedgeto[bj] = [i, j];
                    }
                }

                bestedge[bv] = null;
            }

            b.mybestedges = Object.values(bestedgeto);
            let mybestedge = null;
            let mybestslack = Infinity;

            for (const [i, j] of b.mybestedges) {
                const kslack = slack(i, j);
                if (mybestedge === null || kslack < mybestslack) {
                    mybestedge = [i, j];
                    mybestslack = kslack;
                }
            }

            bestedge[b] = mybestedge;
        }
        function expandBlossom(b, endstage) {
            function* _recurse(b, endstage) {
                for (const s of b.childs) {
                    blossomparent[s] = null;
                    if (s instanceof Blossom) {
                        if (endstage && blossomdual[s] === 0) {
                            yield s;
                        } else {
                            for (const v of s.leaves()) {
                                inblossom[v] = s;
                            }
                        }
                    } else {
                        inblossom[s] = s;
                    }
                    if (!endstage && label.get(s) === 2) {
                        const entrychild = inblossom[labeledge[b][1]];
                        let j = b.childs.indexOf(entrychild);
                        let jstep;
                        if (j & 1) {
                            j -= b.childs.length;
                            jstep = 1;
                        } else {
                            jstep = -1;
                        }
                        let [v, w] = labeledge[b];
                        while (j !== 0) {
                            if (jstep === 1) {
                                const [p, q] = b.edges[j];
                                label[w] = null;
                                label[q] = null;
                                assignLabel(w, 2, v);
                                allowedge[[p, q]] = allowedge[[q, p]] = true;
                            } else {
                                const [q, p] = b.edges[j - 1];
                                label[w] = null;
                                label[q] = null;
                                assignLabel(w, 2, v);
                                allowedge[[p, q]] = allowedge[[q, p]] = true;
                            }
                            j += jstep;
                            if (jstep === 1) {
                                [v, w] = b.edges[j];
                            } else {
                                [w, v] = b.edges[j - 1];
                            }
                            allowedge[[v, w]] = allowedge[[w, v]] = true;
                            j += jstep;
                            const bw = b.childs[j];
                            label[w] = label[bw] = 2;
                            labeledge[w] = labeledge[bw] = [v, w];
                            bestedge[bw] = null;
                            j += jstep;
                            while (b.childs[j] !== entrychild) {
                                const bv = b.childs[j];
                                if (label.get(bv) === 1) {
                                    j += jstep;
                                    continue;
                                }
                                if (bv instanceof Blossom) {
                                    for (const v of bv.leaves()) {
                                        if (label.get(v)) {
                                            break;
                                        }
                                    }
                                } else {
                                    if (label.get(bv)) {
                                        assert(label[bv] === 2);
                                        assert(inblossom[bv] === bv);
                                        label[bv] = null;
                                        label[mate[blossombase[bv]]] = null;
                                        assignLabel(bv, 2, labeledge[bv][0]);
                                    }
                                }
                                j += jstep;
                            }
                        }
                    }
                }
                label.delete(b);
                labeledge.delete(b);
                bestedge.delete(b);
                delete blossomparent[b];
                delete blossombase[b];
                delete blossomdual[b];
            }

            const stack = [_recurse(b, endstage)];
            while (stack.length > 0) {
                const top = stack[stack.length - 1];
                for (const s of top) {
                    stack.push(_recurse(s, endstage));
                    break;
                }
                if (top.length === 0) {
                    stack.pop();
                }
            }
        }
        function *augmentBlossom(b, v) {
            function *_recurse(b, v) {
                let t = v;
                while (blossomparent[t] !== b) {
                    t = blossomparent[t];
                    if (t instanceof Blossom) {
                        yield[t, v];
                    }
                }
                let i = j = b.childs.indexOf(t);
                let jstep;
                if (i & 1) {
                    j -= b.childs.length;
                    jstep = 1;
                } else {
                    jstep = -1;
                }
                while (j !== 0) {
                    j += jstep;
                    t = b.childs[j];
                    let w, x;
                    if (jstep === 1) {
                        [w, x] = b.edges[j];
                    } else {
                        [x, w] = b.edges[j - 1];
                    }
                    if (t instanceof Blossom) {
                        yield[t, w];
                    }
                    j += jstep;
                    t = b.childs[j];
                    if (t instanceof Blossom) {
                        yield[t, x];
                    }
                    mate[w] = x;
                    mate[x] = w;
                }
                b.childs = b.childs.slice(i).concat(b.childs.slice(0, i));
                b.edges = b.edges.slice(i).concat(b.edges.slice(0, i));
                blossombase[b] = blossombase[b.childs[0]];
                assert(blossombase[b] === v);
            }

            let stack = [_recurse(b, v)];
            while (stack.length > 0) {
                const top = stack[stack.length - 1];
                for (const [t, v] of top) {
                    stack.push(_recurse(t, v));
                    break;
                }
                if (top.length === 0) {
                    stack.pop();
                }
            }
        }
        function augmentMatching(v, w) {
            for (var [s, j] of [[v, w], [w, v]]) {
                while (true) {
                    const bs = inblossom[s];
                    assert(label.get(bs) === 1);
                    assert(
                        (labeledge.get(bs) === null && blossombase[bs] !== mate[blossombase[bs]]) ||
                        (labeledge.get(bs)[0] === mate[blossombase[bs]])
                    );
                    if (bs instanceof Blossom) {
                        augmentBlossom(bs, s);
                    }
                    mate[s] = j;
                    if (labeledge.get(bs) === null) {
                        break;
                    }
                    const t = labeledge.get(bs)[0];
                    const bt = inblossom[t];
                    assert(label.get(bt) === 2);
                    [s, j] = labeledge.get(bt);
                    assert(blossombase[bt] === t);
                    if (bt instanceof Blossom) {
                        augmentBlossom(bt, j);
                    }
                    mate[j] = s;
                }
            }
        }

        function verifyOptimum() {
            if (maxcardinality) {
                const vdualoffset = Math.max(0, -Math.min(...Object.values(dualvar)));
            } else {
                const vdualoffset = 0;
            }
            assert(Math.min(...Object.values(dualvar)) + vdualoffset >= 0);
            assert(Object.keys(blossomdual).length === 0 || Math.min(...Object.values(blossomdual)) >= 0);

            for (const [i, j, d] of G.edges()) {
                const wt = d.get(weight, 1);
                if (i === j) continue;

                let s = dualvar[i] + dualvar[j] - 2 * wt;
                let iblossoms = [i];
                let jblossoms = [j];

                while (blossomparent[iblossoms[iblossoms.length - 1]] !== null) {
                    iblossoms.push(blossomparent[iblossoms[iblossoms.length - 1]]);
                }

                while (blossomparent[jblossoms[jblossoms.length - 1]] !== null) {
                    jblossoms.push(blossomparent[jblossoms[jblossoms.length - 1]]);
                }

                iblossoms.reverse();
                jblossoms.reverse();

                for (const [bi, bj] of zip(iblossoms, jblossoms)) {
                    if (bi !== bj) break;
                    s += 2 * blossomdual[bi];
                }

                assert(s >= 0);
                if (mate[i] === j || mate[j] === i) {
                    assert(mate[i] === j && mate[j] === i);
                    assert(s === 0);
                }
            }

            for (const v of gnodes) {
                assert((v in mate) || dualvar[v] + vdualoffset === 0);
            }

            for (const b of Object.keys(blossomdual)) {
                if (blossomdual[b] > 0) {
                    assert(b.edges.length % 2 === 1);
                    for (const [i, j] of b.edges.slice(1, null, 2)) {
                        assert(mate[i] === j && mate[j] === i);
                    }
                }
            }
        }
        while (true) {
            console.log("OUTER");
            label = new Map();
            labeledge = new Map();
            bestedge = new Map();
            for (let b of Object.keys(blossomdual)) {
                blossomdual[b].mybestedges = null;
            }
            allowedge = {};
            queue = [];
            for (let v of gnodes) {
                if (!(v in mate) && !(inblossom[v] in label)) {
                    assignLabel(v, 1, null);
                }
            }
            let augmented = 0;
            while (true) {
                console.log(`Inner que length: ${queue.length}`);

                while (queue.length > 0 && !augmented) {
                    console.log(`Inner queue ${queue.length} ${augmented}`);
                    let v = queue.pop();
                    console.log(`v: ${v}`);
                    assert(label.get(inblossom[v]) === 1);
                    for (let w of G.neighbors(v)) {
                        if (w === v) {
                            continue;
                        }
                        let bv = inblossom[v];
                        let bw = inblossom[w];
                        if (bv === bw) {
                            continue;
                        }
                        if (!allowedge.hasOwnProperty(`${v},${w}`)) {
                            let kslack = slack(v, w);
                            console.log(`kslack ${kslack}`);
                            if (kslack <= 0) {
                                allowedge[`${v},${w}`] = allowedge[`${w},${v}`] = true;
                            }
                        }
                        console.log(`negh ${JSON.stringify(allowedge)}`);
                        if (allowedge.hasOwnProperty(`${v},${w}`)) {
                            console.log(`${v} ${w}`)
                            if (!(bw in label.keys())) {
                                console.log("c1")
                                assignLabel(w, 2, v);
                            } else if (label[bw] === 1) {
                                console.log("c2");
                                let base = scanBlossom(v, w);
                                if (base !== NoNode) {
                                    addBlossom(base, v, w);
                                } else {
                                    augmentMatching(v, w);
                                    augmented = 1;
                                    break;
                                }
                            } else if (!(w in label)) {
                                assert(label[bw] === 2);
                                label[w] = 2;
                                labeledge[w] = [v, w];
                            }
                        } else if (label[bw] === 1) {
                            if (bestedge[bv] === undefined || kslack < slack(...bestedge[bv])) {
                                bestedge[bv] = [v, w];
                            }
                        } else if (!(w in label)) {
                            if (bestedge[w] === undefined || kslack < slack(...bestedge[w])) {
                                bestedge[w] = [v, w];
                            }
                        }
                    }
                }
                if (augmented) {
                    console.log("Break!");
                    break;
                }
                let deltatype = -1;
                let delta = deltaedge = deltablossom = null;
                if (!maxcardinality) {
                    deltatype = 1;
                    delta = Math.min(...Object.values(dualvar));
                }
                for (let v of G.nodes()) {
                    if (!(inblossom[v] in label) && bestedge.hasOwnProperty(v)) {
                        let d = slack(...bestedge[v]);
                        if (deltatype === -1 || d < delta) {
                            delta = d;
                            deltatype = 2;
                            deltaedge = bestedge[v];
                        }
                    }
                }
                for (let b of Object.keys(blossomparent)) {
                    if (blossomparent[b] === null && label[b] === 1 && bestedge.hasOwnProperty(b)) {
                        let kslack = slack(...bestedge[b]);
                        let d = allinteger ? Math.floor(kslack / 2) : kslack / 2.0;
                        if (deltatype === -1 || d < delta) {
                            delta = d;
                            deltatype = 3;
                            deltaedge = bestedge[b];
                        }
                    }
                }
                for (let b of Object.keys(blossomdual)) {
                    if (blossomparent[b] === null && label[b] === 2 && (deltatype === -1 || blossomdual[b] < delta)) {
                        delta = blossomdual[b];
                        deltatype = 4;
                        deltablossom = b;
                    }
                }
                if (deltatype === -1) {
                    assert(maxcardinality);
                    deltatype = 1;
                    delta = Math.max(0, Math.min(...Object.values(dualvar)));
                }
                for (let v of gnodes) {
                    if (label[inblossom[v]] === 1) {
                        dualvar[v] -= delta;
                    } else if (label[inblossom[v]] === 2) {
                        dualvar[v] += delta;
                    }
                }
                for (let b of Object.keys(blossomdual)) {
                    if (blossomparent[b] === null) {
                        if (label[b] === 1) {
                            blossomdual[b] += delta;
                        } else if (label[b] === 2) {
                            blossomdual[b] -= delta;
                        }
                    }
                }
                if (deltatype === 1) {
                    break;
                } else if (deltatype === 2) {
                    let [v, w] = deltaedge;
                    assert(label[inblossom[v]] === 1);
                    allowedge[`${v},${w}`] = allowedge[`${w},${v}`] = true;
                    queue.push(v);
                } else if (deltatype === 3) {
                    let [v, w] = deltaedge;
                    allowedge[`${v},${w}`] = allowedge[`${w},${v}`] = true;
                    assert(label[inblossom[v]] === 1);
                    queue.push(v);
                } else if (deltatype === 4) {
                    expandBlossom(deltablossom, false);
                }
            }
            for (let v of Object.keys(mate)) {
                assert(mate[mate[v]] === v);
            }
            if (!augmented) {
                break;
            }
            for (let b of Object.keys(blossomdual)) {
                if (blossomparent[b] === null && label[b] === 1 && blossomdual[b] === 0) {
                    expandBlossom(b, true);
                }
            }
        }
        if (allinteger) {
            verifyOptimum()
        }

        return matching_dict_to_set(mate)
    }
}
/**
 * Convert matching represented as a dict to a set of tuples.
 * The keys of mate are vertices in the graph, and mate[v] is v's partner
 * vertex in the matching.
 */
function matchingDictToSet(mate) {
    let matching = new Set();
    for (let [v, w] of Object.entries(mate)) {
        if (w !== null && v < w) {
            matching.add([v, w]);
        }
    }
    return matching;
}

console.log("Starting")
var gBase = new Graph();
var edges = [[1, 2, 6], [1, 3, 2], [2, 3, 1], [2, 4, 7], [3, 5, 9], [4, 5, 3]];

for (var edge in edges) {
    var [u, v, weight] = edges[edge];
    console.log("Adding edge:", u, v, weight);
    gBase.addEdge(u.toString(), v.toString(), weight)
}

const g = Graph.withProxy(gBase);

console.log(g.nodes())
console.log(g["1"]);
console.log(g["1"]["2"]);

var result = Graph.maxWeightMatching(g)
console.log(result)