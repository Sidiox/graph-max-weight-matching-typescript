### USER
```py  
from itertools import repeat
from numbers import Number
from typing import Any
  
  
class NoNode:
    """Dummy value which is different from any node."""
  
  
class Blossom:
    """Representation of a non-trivial blossom or sub-blossom."""
  
    **slots** = ["childs", "edges", "mybestedges"]
  
    def leaves(self):
        stack = [*self.childs]
        while stack:
            t = stack.pop()
            if isinstance(t, Blossom):
                stack.extend(t.childs)
            else:
                yield t
  
  
class Graph:
    def **init**(self):
        self.adj: dict[str, dict[str, int]] = {}
        self.node_list: list[str] = []
  
    def add_edge(self, u: str, v: str, weight=1):
        """Adds an edge between nodes u and v with the specified weight."""
        if u not in self.adj:
            self.adj[u] = {}
            self.node_list.append(u)
        if v not in self.adj:
            self.adj[v] = {}
            self.node_list.append(v)
        self.adj[u][v] = weight
        self.adj[v][u] = weight
  
    def add_weighted_edges_from(self, edge_list):
        """Adds multiple weighted edges from a list of (u, v, weight) tuples."""
        for u, v, weight in edge_list:
            self.add_edge(u, v, weight)
  
    def **iter**(self):
        """Allows iteration over the nodes in the graph."""
        return iter(self.node_list)
  
    def edges(self):
        """Iterates through edges,  including weights."""
        seen = set()
        for u in self.adj:
            for v in self.adj[u]:
                if (u, v) not in seen and (v, u) not in seen:
                    seen.add((u, v))
                    yield u, v, self.adj[u][v]
  
    def neighbors(self, node):
        """Returns a list of neighbors for the given node."""
        return list(self.adj[node].keys())
  
    def **getitem**(self, node: str) -> tuple[dict[str, int], int]:
        """Enables accessing neighbors and their weights using graph[node][neighbor]."""
        return self.adj[node]
  
    def nodes(self):
        """Returns an iterator over the nodes in the graph."""
        return iter(self.node_list)
  
  
# custom type hint for something that is either a str a Blossom or NoNode
NodeType = str | Blossom | type[NoNode]
NullableNodeType = NodeType | None
  
  
def max_weight_matching(G: "Graph", maxcardinality=False):
    gnodes: list[str] = list(G)
    if not gnodes:
        return set()
    maxweight = 0
    allinteger = True
    for i, j, d in G.edges():
        wt = d
        if i != j and wt > maxweight:
            maxweight = wt
        allinteger = allinteger and (str(type(wt)).split("'")[1] in ("int", "long"))
    mate = {}
    label: dict[NullableNodeType, int] = {}
    labeledge: dict[NodeType, tuple[NodeType, NodeType] | None] = {}
    inblossom: dict[NodeType, NodeType] = dict(zip(gnodes, gnodes))
    blossomparent: dict[NullableNodeType, NullableNodeType] = {}
    for node in gnodes:
        blossomparent[node] = None
    blossombase: dict[NodeType, NodeType] = dict(zip(gnodes, gnodes))
  
    bestedge = {}
    dualvar: dict[NodeType, int] = {node: maxweight for node in gnodes}
    blossomdual: dict[NullableNodeType, int] = {}
    allowedge: dict[tuple[NodeType, NodeType], bool] = {}
    queue: list[NodeType] = []
  
    def slack(v: str, w: str) -> int:
        dualvar_v = dualvar[v]
        dualvar_w = dualvar[w]
        weight = 2 * G[v][w]
        print(f"{dualvar_v=} {dualvar_w=} {weight=}")
        return dualvar_v + dualvar_w - weight
  
    def assignLabel(w: NodeType, t: int, v: NullableNodeType):
        b = inblossom[w]
        assert label.get(w) is None and label.get(b) is None
        label[w] = t
        label[b] = t
        if v is not None:
            labeledge[w] = (v, w)
            labeledge[b] = (v, w)
        else:
            labeledge[w] = None
            labeledge[b] = None
        bestedge[w] = bestedge[b] = None
        if t == 1:
            if isinstance(b, Blossom):
                queue.extend(b.leaves())
            else:
                queue.append(b)
        elif t == 2:
            base = blossombase[b]
            assignLabel(mate[base], 1, base)
  
    def scanBlossom(v: NullableNodeType, w: NullableNodeType) -> NullableNodeType:
        """
        Side effecting, touching label
        """
        path = []
        base = NoNode
        while v is not NoNode:
            b = inblossom[v]
            if label[b] & 4:
                base = blossombase[b]
                break
            assert label[b] == 1
            path.append(b)
            label[b] = 5
            if labeledge[b] is None:
                assert blossombase[b] not in mate
                v = NoNode
            else:
                assert labeledge[b][0] == mate[blossombase[b]]
                v = labeledge[b][0]
                b = inblossom[v]
                assert label[b] == 2
                v = labeledge[b][0]
            if w is not NoNode:
                v, w = w, v
        for b in path:
            label[b] = 1
        return base
  
    def addBlossom(base, v, w):
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        b = Blossom()
        blossombase[b] = base
        blossomparent[b] = None
        blossomparent[bb] = b
        b.childs = []
        path = []
        b.edges = [(v, w)]
        edgs = [(v, w)]
        while bv != bb:
            blossomparent[bv] = b
            path.append(bv)
            edgs.append(labeledge[bv])
            assert label[bv] == 2 or (
                label[bv] 1 and labeledge[bv][0] mate[blossombase[bv]]
            )
            v = labeledge[bv][0]
            bv = inblossom[v]
        path.append(bb)
        path.reverse()
        edgs.reverse()
        while bw != bb:
            blossomparent[bw] = b
            path.append(bw)
            edgs.append((labeledge[bw][1], labeledge[bw][0]))
            assert label[bw] == 2 or (
                label[bw] 1 and labeledge[bw][0] mate[blossombase[bw]]
            )
            w = labeledge[bw][0]
            bw = inblossom[w]
        assert label[bb] == 1
        label[b] = 1
        labeledge[b] = labeledge[bb]
        blossomdual[b] = 0
        for v in b.leaves():
            if label[inblossom[v]] == 2:
                queue.append(v)
            inblossom[v] = b
        bestedgeto = {}
        for bv in path:
            if isinstance(bv, Blossom):
                if bv.mybestedges is not None:
                    nblist = bv.mybestedges
                    bv.mybestedges = None
                else:
                    nblist = [
                        (v, w) for v in bv.leaves() for w in G.neighbors(v) if v != w
                    ]
            else:
                nblist = [(bv, w) for w in G.neighbors(bv) if bv != w]
            for k in nblist:
                (i, j) = k
                if inblossom[j] == b:
                    i, j = j, i
                bj = inblossom[j]
                if (
                    bj != b
                    and label.get(bj) == 1
                    and ((bj not in bestedgeto) or slack(i, j) < slack(*bestedgeto[bj]))
                ):
                    bestedgeto[bj] = k
            bestedge[bv] = None
        b.mybestedges = list(bestedgeto.values())
        mybestedge = None
        bestedge[b] = None
        for k in b.mybestedges:
            kslack = slack(*k)
            if mybestedge is None or kslack < mybestslack:
                mybestedge = k
                mybestslack = kslack
        bestedge[b] = mybestedge
  
    def expandBlossom(b: Blossom, endstage: bool):
        def _recurse(b: Blossom, endstage: bool):
            for s in b.childs:
                blossomparent[s] = None
                if isinstance(s, Blossom):
                    if endstage and blossomdual[s] == 0:
                        yield s
                    else:
                        for v in s.leaves():
                            inblossom[v] = s
                else:
                    inblossom[s] = s
            if (not endstage) and label.get(b) == 2:
                entrychild = inblossom[labeledge[b][1]]
                j = b.childs.index(entrychild)
                if j & 1:
                    j -= len(b.childs)
                    jstep = 1
                else:
                    jstep = -1
                v, w = labeledge[b]
                while j != 0:
                    if jstep == 1:
                        p, q = b.edges[j]
                    else:
                        q, p = b.edges[j - 1]
                    label[w] = None
                    label[q] = None
                    assignLabel(w, 2, v)
                    allowedge[(p, q)] = allowedge[(q, p)] = True
                    j += jstep
                    if jstep == 1:
                        v, w = b.edges[j]
                    else:
                        w, v = b.edges[j - 1]
                    allowedge[(v, w)] = allowedge[(w, v)] = True
                    j += jstep
                bw = b.childs[j]
                label[w] = label[bw] = 2
                labeledge[w] = labeledge[bw] = (v, w)
                bestedge[bw] = None
                j += jstep
                while b.childs[j] != entrychild:
                    bv = b.childs[j]
                    if label.get(bv) == 1:
                        j += jstep
                        continue
                    if isinstance(bv, Blossom):
                        for v in bv.leaves():
                            if label.get(v):
                                break
                    else:
                        v = bv
                    if label.get(v):
                        assert label[v] == 2
                        assert inblossom[v] == bv
                        label[v] = None
                        label[mate[blossombase[bv]]] = None
                        assignLabel(v, 2, labeledge[v][0])
                    j += jstep
            label.pop(b, None)
            labeledge.pop(b, None)
            bestedge.pop(b, None)
            del blossomparent[b]
            del blossombase[b]
            del blossomdual[b]
  
        stack = [_recurse(b, endstage)]
        while stack:
            top = stack[-1]
            for s in top:
                stack.append(_recurse(s, endstage))
                break
            else:
                stack.pop()
  
    def augmentBlossom(b: Blossom, v):
        def _recurse(b: Blossom, v):
            t = v
            while blossomparent[t] != b:
                t = blossomparent[t]
            if isinstance(t, Blossom):
                yield (t, v)
            i = j = b.childs.index(t)
            if i & 1:
                j -= len(b.childs)
                jstep = 1
            else:
                jstep = -1
            while j != 0:
                j += jstep
                t = b.childs[j]
                if jstep == 1:
                    w, x = b.edges[j]
                else:
                    x, w = b.edges[j - 1]
                if isinstance(t, Blossom):
                    yield (t, w)
                j += jstep
                t = b.childs[j]
                if isinstance(t, Blossom):
                    yield (t, x)
                mate[w] = x
                mate[x] = w
            b.childs = b.childs[i:] + b.childs[:i]
            b.edges = b.edges[i:] + b.edges[:i]
            blossombase[b] = blossombase[b.childs[0]]
            assert blossombase[b] == v
  
        stack = [_recurse(b, v)]
        while stack:
            top = stack[-1]
            for args in top:
                stack.append(_recurse(*args))
                break
            else:
                stack.pop()
  
    def augmentMatching(v, w):
        for s, j in ((v, w), (w, v)):
            while 1:
                bs = inblossom[s]
                assert label[bs] == 1
                assert (labeledge[bs] is None and blossombase[bs] not in mate) or (
                    labeledge[bs][0] == mate[blossombase[bs]]
                )
                if isinstance(bs, Blossom):
                    augmentBlossom(bs, s)
                mate[s] = j
                if labeledge[bs] is None:
                    break
                t = labeledge[bs][0]
                bt = inblossom[t]
                assert label[bt] == 2
                s, j = labeledge[bt]
                assert blossombase[bt] == t
                if isinstance(bt, Blossom):
                    augmentBlossom(bt, j)
                mate[j] = s
  
    def verifyOptimum():
        if maxcardinality:
            vdualoffset = max(0, -min(dualvar.values()))
        else:
            vdualoffset = 0
        assert min(dualvar.values()) + vdualoffset >= 0
        assert len(blossomdual) == 0 or min(blossomdual.values()) >= 0
        for i, j, d in G.edges():
            wt = d
            if i == j:
                continue
            s = dualvar[i] + dualvar[j] - 2 * wt
            iblossoms: list[NullableNodeType] = [i]
            jblossoms: list[NullableNodeType] = [j]
            while blossomparent[iblossoms[-1]] is not None:
                iblossoms.append(blossomparent[iblossoms[-1]])
            while blossomparent[jblossoms[-1]] is not None:
                jblossoms.append(blossomparent[jblossoms[-1]])
            iblossoms.reverse()
            jblossoms.reverse()
            for bi, bj in zip(iblossoms, jblossoms):
                if bi != bj:
                    break
                s += 2 * blossomdual[bi]
            assert s >= 0
            if mate.get(i) j or mate.get(j) i:
                assert mate[i] j and mate[j] i
                assert s == 0
        for v in gnodes:
            assert (v in mate) or dualvar[v] + vdualoffset == 0
        for b in blossomdual:
            if blossomdual[b] > 0:
                assert len(b.edges) % 2 == 1
                for i, j in b.edges[1::2]:
                    assert mate[i] j and mate[j] i
  
    while 1:
        print("outer")
        label.clear()
        labeledge.clear()
        bestedge.clear()
        for b in blossomdual:
            b.mybestedges = None
        allowedge.clear()
        queue[:] = []
        for v in gnodes:
            if (v not in mate) and label.get(inblossom[v]) is None:
                assignLabel(v, 1, None)
        augmented = 0
        while 1:
            print(f"Inner 1 {len(queue)}")
  
            while queue and not augmented:
                print("Inner queue")
                v = queue.pop()
                print(v)
                assert label[inblossom[v]] == 1
                for w in G.neighbors(v):
                    if w == v:
                        continue
                    bv = inblossom[v]
                    bw = inblossom[w]
                    if bv == bw:
                        continue
                    if (v, w) not in allowedge:
                        kslack = slack(v, w)
                        print(f"kslack {kslack}")
                        if kslack <= 0:
                            allowedge[(v, w)] = allowedge[(w, v)] = True
                    print(f"negh ${allowedge}")
                    if (v, w) in allowedge:
                        print("c2")
                        if label.get(bw) is None:
                            assignLabel(w, 2, v)
                        elif label.get(bw) == 1:
                            print("Scan")
                            base = scanBlossom(v, w)
                            if base is not NoNode:
                                addBlossom(base, v, w)
                            else:
                                augmentMatching(v, w)
                                augmented = 1
                                break
                        elif label.get(w) is None:
                            assert label[bw] == 2
                            label[w] = 2
                            labeledge[w] = (v, w)
                    elif label.get(bw) == 1:
                        if bestedge.get(bv) is None or kslack < slack(*bestedge[bv]):
                            bestedge[bv] = (v, w)
                    elif label.get(w) is None:
                        if bestedge.get(w) is None or kslack < slack(*bestedge[w]):
                            bestedge[w] = (v, w)
            if augmented:
                print("Break!")
                break
            deltatype = -1
            delta: None | int | float = None
            deltaedge: tuple[NodeType, NodeType] | None = None
            deltablossom = None
            if not maxcardinality:
                deltatype = 1
                delta = min(dualvar.values())
            for v in G.nodes():
                if label.get(inblossom[v]) is None and bestedge.get(v) is not None:
                    d = slack(*bestedge[v])
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 2
                        deltaedge = bestedge[v]
            for b in blossomparent:
                if (
                    blossomparent[b] is None
                    and label.get(b) == 1
                    and bestedge.get(b) is not None
                ):
                    kslack = slack(*bestedge[b])
                    if allinteger:
                        assert (kslack % 2) == 0
                        d = kslack // 2
                    else:
                        d = kslack / 2.0
                    if deltatype == -1 or d < delta:
                        delta = d
                        deltatype = 3
                        deltaedge = bestedge[b]
            for b in blossomdual:
                if (
                    blossomparent[b] is None
                    and label.get(b) == 2
                    and (deltatype == -1 or blossomdual[b] < delta)
                ):
                    delta = blossomdual[b]
                    deltatype = 4
                    deltablossom = b
            if deltatype == -1:
                assert maxcardinality
                deltatype = 1
                delta = max(0, min(dualvar.values()))
            for v in gnodes:
                if label.get(inblossom[v]) == 1:
                    dualvar[v] -= delta
                elif label.get(inblossom[v]) == 2:
                    dualvar[v] += delta
            for b in blossomdual:
                if blossomparent[b] is None:
                    if label.get(b) == 1:
                        blossomdual[b] += delta
                    elif label.get(b) == 2:
                        blossomdual[b] -= delta
            if deltatype == 1:
                break
            elif deltatype == 2:
                (v, w) = deltaedge
                assert label[inblossom[v]] == 1
                allowedge[(v, w)] = True
                allowedge[(w, v)] = True
                queue.append(v)
            elif deltatype == 3:
                (v, w) = deltaedge
                allowedge[(v, w)] = True
                allowedge[(w, v)] = True
                assert label[inblossom[v]] == 1
                queue.append(v)
            elif deltatype == 4:
                assert isinstance(deltablossom, Blossom)
                expandBlossom(deltablossom, False)
        for v in mate:
            assert mate[mate[v]] == v
        if not augmented:
            break
        for b in list(blossomdual.keys()):
            if b not in blossomdual:
                continue
            if blossomparent[b] is None and label.get(b) 1 and blossomdual[b] 0:
                assert isinstance(b, Blossom)
                expandBlossom(b, True)
    if allinteger:
        verifyOptimum()
    return matching_dict_to_set(mate)
  
  
def matching_dict_to_set(mate: dict):
    """Convert matching represented as a dict to a set of tuples.
    The keys of mate are vertices in the graph, and mate[v] is v's partner
    vertex in the matching.
    """
    matching = set()
    for v, w in mate.items():
        if w is not None and v < w:
            matching.add((v, w))
    return matching
  
  
if **name** == "__main__":
    g = Graph()
    print("Graph:")
    edges = [
        ("A", "B", 6),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 7),
        ("C", "E", 9),
        ("D", "E", 3),
    ]
  
    g.add_weighted_edges_from(edges)
    res = max_weight_matching(g)
    print("Pure Python:")
    print(res)
    assert res == frozenset({('C', 'E'), ('B', 'D')})
  
    g = Graph()
    print("Graph:")
    edges = [
        ("A", "B", 6),
        ("A", "C", 2),
        ("B", "C", 1),
        ("B", "D", 2),
        ("C", "E", 9),
        ("D", "E", 3),
    ]
  
    g.add_weighted_edges_from(edges)
    res = max_weight_matching(g)
    print("Pure Python:")
    print(res)
    assert res == frozenset({('C', 'E'), ('A', 'B')})  
```  
Translate the above code to TypeScript. Since it is a very complicated algorithm, do not change the structure.

### ASSISTANT
Translating the provided Python code to TypeScript while maintaining the same structure is a complex task due to the differences in language syntax and features. Below is a TypeScript version of the code:

```typescript
class NoNode {
    // Dummy value which is different from any node.
}

class Blossom {
    childs: (Blossom | string)[];
    edges: [string, string][];
    mybestedges?: [string, string][];

    constructor() {
        this.childs = [];
        this.edges = [];
    }

    *leaves(): Generator<string> {
        let stack = [...this.childs];
        while (stack.length > 0) {
            let t = stack.pop()!;
            if (t instanceof Blossom) {
                stack.push(...t.childs);
            } else {
                yield t;
            }
        }
    }
}

class Graph {
    adj: { [key: string]: { [key: string]: number } };
    nodeList: string[];

    constructor() {
        this.adj = {};
        this.nodeList = [];
    }

    addEdge(u: string, v: string, weight = 1): void {
        if (!(u in this.adj)) {
            this.adj[u] = {};
            this.nodeList.push(u);
        }
        if (!(v in this.adj)) {
            this.adj[v] = {};
            this.nodeList.push(v);
        }
        this.adj[u][v] = weight;
        this.adj[v][u] = weight;
    }

    addWeightedEdgesFrom(edgeList: [string, string, number][]): void {
        for (let [u, v, weight] of edgeList) {
            this.addEdge(u, v, weight);
        }
    }

    *[Symbol.iterator](): Generator<string> {
        return this.nodeList[Symbol.iterator]();
    }

    *edges(): Generator<[string, string, number]> {
        let seen = new Set<string>();
        for (let u in this.adj) {
            for (let v in this.adj[u]) {
                if (!seen.has(`${u}-${v}`) && !seen.has(`${v}-${u}`)) {
                    seen.add(`${u}-${v}`);
                    yield [u, v, this.adj[u][v]];
                }
            }
        }
    }

    neighbors(node: string): string[] {
        return Object.keys(this.adj[node]);
    }

    get(node: string): { [key: string]: number } {
        return this.adj[node];
    }

    *nodes(): Generator<string> {
        return this.nodeList[Symbol.iterator]();
    }
}

type NodeType = string | Blossom | typeof NoNode;
type NullableNodeType = NodeType | null;

function maxWeightMatching(G: Graph, maxcardinality = false): Set<[string, string]> {
    let gnodes: string[] = Array.from(G);
    if (gnodes.length === 0) {
        return new Set();
    }
    let maxweight = 0;
    let allinteger = true;
    for (let [i, j, d] of G.edges()) {
        let wt = d;
        if (i !== j && wt > maxweight) {
            maxweight = wt;
        }
        allinteger = allinteger && (typeof wt === "number" && Number.isInteger(wt));
    }
    let mate: { [key: string]: string | null } = {};
    let label: { [key: string]: number } = {};
    let labeledge: { [key: string]: [string, string] | null } = {};
    let inblossom: { [key: string]: NodeType } = Object.fromEntries(gnodes.map(node => [node, node]));
    let blossomparent: { [key: string]: NodeType | null } = Object.fromEntries(gnodes.map(node => [node, null]));
    let blossombase: { [key: string]: NodeType } = Object.fromEntries(gnodes.map(node => [node, node]));

    let bestedge: { [key: string]: [string, string] | null } = {};
    let dualvar: { [key: string]: number } = Object.fromEntries(gnodes.map(node => [node, maxweight]));
    let blossomdual: { [key: string]: number } = {};
    let allowedge: { [key: string]: boolean } = {};
    let queue: string[] = [];

    function slack(v: string, w: string): number {
        let dualvar_v = dualvar[v];
        let dualvar_w = dualvar[w];
        let weight = 2 * G.get(v)[w];
        console.log(`${dualvar_v=} ${dualvar_w=} ${weight=}`);
        return dualvar_v + dualvar_w - weight;
    }

    function assignLabel(w: NodeType, t: number, v: NullableNodeType): void {
        let b = inblossom[w as string];
        if (label[b as string] !== undefined || label[w as string] !== undefined) {
            throw new Error("Label already assigned");
        }
        label[b as string] = t;
        label[w as string] = t;
        if (v !== null) {
            labeledge[w as string] = [v as string, w as string];
            labeledge[b as string] = [v as string, w as string];
        } else {
            labeledge[w as string] = null;
            labeledge[b as string] = null;
        }
        bestedge[w as string] = bestedge[b as string] = null;
        if (t === 1) {
            if (b instanceof Blossom) {
                queue.push(...b.leaves());
            } else {
                queue.push(b as string);
            }
        } else if (t === 2) {
            let base = blossombase[b as string];
            assignLabel(mate[base as string], 1, base);
        }
    }

    function scanBlossom(v: NullableNodeType, w: NullableNodeType): NullableNodeType {
        let path: NodeType[] = [];
        let base: NodeType = NoNode;
        while (v !== NoNode) {
            let b = inblossom[v as string];
            if (label[b as string] & 4) {
                base = blossombase[b as string];
                break;
            }
            if (label[b as string] !== 1) {
                throw new Error("Label is not 1");
            }
            path.push(b);
            label[b as string] = 5;
            if (labeledge[b as string] === null) {
                if (!(blossombase[b as string] in mate)) {
                    v = NoNode;
                }
            } else {
                if (labeledge[b as string][0] !== mate[blossombase[b as string]]) {
                    throw new Error("Label edge mismatch");
                }
                v = labeledge[b as string][0];
                b = inblossom[v];
                if (label[b as string] !== 2) {
                    throw new Error("Label is not 2");
                }
                v = labeledge[b as string][0];
            }
            if (w !== NoNode) {
                [v, w] = [w, v];
            }
        }
        for (let b of path) {
            label[b as string] = 1;
        }
        return base;
    }

    function addBlossom(base: NodeType, v: NodeType, w: NodeType): void {
        let bb = inblossom[base as string];
        let bv = inblossom[v as string];
        let bw = inblossom[w as string];
        let b = new Blossom();
        blossombase[b] = base;
        blossomparent[b] = null;
        blossomparent[bb as string] = b;
        b.childs = [];
        b.edges = [[v as string, w as string]];
        let edgs = [[v as string, w as string]];
        while (bv !== bb) {
            blossomparent[bv as string] = b;
            edgs.push(labeledge[bv as string]);
            if (label[bv as string] !== 2 && (label[bv as string] !== 1 || labeledge[bv as string][0] !== mate[blossombase[bv as string]])) {
                throw new Error("Label mismatch");
            }
            v = labeledge[bv as string][0];
            bv = inblossom[v];
        }
        edgs.reverse();
        while (bw !== bb) {
            blossomparent[bw as string] = b;
            edgs.push([labeledge[bw as string][1], labeledge[bw as string][0]]);
            if (label[bw as string] !== 2 && (label[bw as string] !== 1 || labeledge[bw as string][0] !== mate[blossombase[bw as string]])) {
                throw new Error("Label mismatch");
            }
            w = labeledge[bw as string][0];
            bw = inblossom[w];
        }
        if (label[bb as string] !== 1) {
            throw new Error("Label is not 1");
        }
        label[b] = 1;
        labeledge[b] = labeledge[bb as string];
        blossomdual[b] = 0;
        for (let v of b.leaves()) {
            if (label[inblossom[v] as string] === 2) {
                queue.push(v);
            }
            inblossom[v] = b;
        }
        let bestedgeto: { [key: string]: [string, string] } = {};
        for (let bv of b.childs) {
            let nblist: [string, string][] = [];
            if (bv instanceof Blossom) {
                if (bv.mybestedges !== undefined) {
                    nblist = bv.mybestedges;
                    bv.mybestedges = undefined;
                } else {
                    nblist = Array.from(bv.leaves()).flatMap(v => G.neighbors(v).map(w => [v, w] as [string, string]));
                }
            } else {
                nblist = G.neighbors(bv as string).map(w => [bv as string, w] as [string, string]);
            }
            for (let [i, j] of nblist) {
                if (inblossom[j] === b) {
                    [i, j] = [j, i];
                }
                let bj = inblossom[j];
                if (bj !== b && label[bj as string] === 1 && (bestedgeto[bj as string] === undefined || slack(i, j) < slack(...bestedgeto[bj as string]))) {
                    bestedgeto[bj as string] = [i, j];
                }
            }
            bestedge[bv as string] = null;
        }
        b.mybestedges = Object.values(bestedgeto);
        let mybestedge: [string, string] | null = null;
        let mybestslack: number | null = null;
        for (let k of b.mybestedges) {
            let kslack = slack(...k);
            if (mybestedge === null || kslack < mybestslack) {
                mybestedge = k;
                mybestslack = kslack;
            }
        }
        bestedge[b] = mybestedge;
    }

    function expandBlossom(b: Blossom, endstage: boolean): void {
        function _recurse(b: Blossom, endstage: boolean): Generator<Blossom> {
            for (let s of b.childs) {
                blossomparent[s as string] = null;
                if (s instanceof Blossom) {
                    if (endstage && blossomdual[s] === 0) {
                        yield s;
                    } else {
                        for (let v of s.leaves()) {
                            inblossom[v] = s;
                        }
                    }
                } else {
                    inblossom[s as string] = s as string;
                }
            }
            if (!endstage && label[b] === 2) {
                let entrychild = inblossom[labeledge[b][1]];
                let j = b.childs.indexOf(entrychild);
                let jstep = j & 1 ? 1 : -1;
                let [v, w] = labeledge[b];
                while (j !== 0) {
                    let [p, q] = jstep === 1 ? b.edges[j] : b.edges[j - 1];
                    label[w] = null;
                    label[q] = null;
                    assignLabel(w, 2, v);
                    allowedge[`${p}-${q}`] = allowedge[`${q}-${p}`] = true;
                    j += jstep;
                    [v, w] = jstep === 1 ? b.edges[j] : b.edges[j - 1];
                    allowedge[`${v}-${w}`] = allowedge[`${w}-${v}`] = true;
                    j += jstep;
                }
                let bw = b.childs[j];
                label[w] = label[bw] = 2;
                labeledge[w] = labeledge[bw] = [v, w];
                bestedge[bw] = null;
                j += jstep;
                while (b.childs[j] !== entrychild) {
                    let bv = b.childs[j];
                    if (label[bv as string] === 1) {
                        j += jstep;
                        continue;
                    }
                    let v = bv instanceof Blossom ? Array.from(bv.leaves()).find(v => label[v]) : bv as string;
                    if (v && label[v] === 2 && inblossom[v] === bv) {
                        label[v] = null;
                        label[mate[blossombase[bv as string]]] = null;
                        assignLabel(v, 2, labeledge[v][0]);
                    }
                    j += jstep;
                }
            }
            delete label[b];
            delete labeledge[b];
            delete bestedge[b];
            delete blossomparent[b];
            delete blossombase[b];
            delete blossomdual[b];
        }

        let stack = [_recurse(b, endstage)];
        while (stack.length > 0) {
            let top = stack.pop()!;
            for (let s of top) {
                stack.push(_recurse(s, endstage));
                break;
            }
        }
    }

    function augmentBlossom(b: Blossom, v: string): void {
        function _recurse(b: Blossom, v: string): Generator<[Blossom, string]> {
            let t = v;
            while (blossomparent[t] !== b) {
                t = blossomparent[t] as string;
            }
            if (t instanceof Blossom) {
                yield [t, v];
            }
            let i = j = b.childs.indexOf(t);
            let jstep = i & 1 ? 1 : -1;
            while (j !== 0) {
                j += jstep;
                let t = b.childs[j];
                let [w, x] = jstep === 1 ? b.edges[j] : b.edges[j - 1];
                if (t instanceof Blossom) {
                    yield [t, w];
                }
                j += jstep;
                t = b.childs[j];
                if (t instanceof Blossom) {
                    yield [t, x];
                }
                mate[w] = x;
                mate[x] = w;
            }
            b.childs = b.childs.slice(i).concat(b.childs.slice(0, i));
            b.edges = b.edges.slice(i).concat(b.edges.slice(0, i));
            blossombase[b] = blossombase[b.childs[0]];
        }

        let stack = [_recurse(b, v)];
        while (stack.length > 0) {
            let top = stack.pop()!;
            for (let args of top) {
                stack.push(_recurse(...args));
                break;
            }
        }
    }

    function augmentMatching(v: string, w: string): void {
        for (let [s, j] of [[v, w], [w, v]]) {
            while (true) {
                let bs = inblossom[s];
                if (label[bs as string] !== 1) {
                    throw new Error("Label is not 1");
                }
                if ((labeledge[bs as string] === null && !(blossombase[bs as string] in mate)) || labeledge[bs as string][0] !== mate[blossombase[bs as string]]) {
                    throw new Error("Label edge mismatch");
                }
                if (bs instanceof Blossom) {
                    augmentBlossom(bs, s);
                }
                mate[s] = j;
                if (labeledge[bs as string] === null) {
                    break;
                }
                let t = labeledge[bs as string][0];
                let bt = inblossom[t];
                if (label[bt as string] !== 2) {
                    throw new Error("Label is not 2");
                }
                [s, j] = labeledge[bt as string];
                if (blossombase[bt as string] !== t) {
                    throw new Error("Blossom base mismatch");
                }
                if (bt instanceof Blossom) {
                    augmentBlossom(bt, j);
                }
                mate[j] = s;
            }
        }
    }

    function verifyOptimum(): void {
        let vdualoffset = maxcardinality ? Math.max(0, -Math.min(...Object.values(dualvar))) : 0;
        if (Math.min(...Object.values(dualvar)) + vdualoffset < 0) {
            throw new Error("Dual variable violation");
        }
        if (Object.values(blossomdual).length > 0 && Math.min(...Object.values(blossomdual)) < 0) {
            throw new Error("Blossom dual violation");
        }
        for (let [i, j, d] of G.edges()) {
            let wt = d;
            if (i === j) {
                continue;
            }
            let s = dualvar[i] + dualvar[j] - 2 * wt;
            let iblossoms: NodeType[] = [i];
            let jblossoms: NodeType[] = [j];
            while (blossomparent[iblossoms[iblossoms.length - 1] as string] !== null) {
                iblossoms.push(blossomparent[iblossoms[iblossoms.length - 1] as string]);
            }
            while (blossomparent[jblossoms[jblossoms.length - 1] as string] !== null) {
                jblossoms.push(blossomparent[jblossoms[jblossoms.length - 1] as string]);
            }
            iblossoms.reverse();
            jblossoms.reverse();
            for (let [bi, bj] of iblossoms.map((_, i) => [iblossoms[i], jblossoms[i]])) {
                if (bi !== bj) {
                    break;
                }
                s += 2 * blossomdual[bi as string];
            }
            if (s < 0) {
                throw new Error("Slack violation");
            }
            if ((mate[i] !== j || mate[j] !== i) && (mate[i] === j && mate[j] === i)) {
                throw new Error("Matching violation");
            }
            if (mate[i] === j && mate[j] === i && s !== 0) {
                throw new Error("Slack zero violation");
            }
        }
        for (let v of gnodes) {
            if (!(v in mate) && dualvar[v] + vdualoffset !== 0) {
                throw new Error("Dual variable zero violation");
            }
        }
        for (let b of Object.keys(blossomdual)) {
            if (blossomdual[b] > 0 && b.edges.length % 2 !== 1) {
                throw new Error("Blossom edge count violation");
            }
            for (let [i, j] of b.edges.slice(1, b.edges.length, 2)) {
                if (mate[i] !== j || mate[j] !== i) {
                    throw new Error("Blossom matching violation");
                }
            }
        }
    }

    while (true) {
        console.log("outer");
        label = {};
        labeledge = {};
        bestedge = {};
        for (let b of Object.keys(blossomdual)) {
            (blossomdual[b] as Blossom).mybestedges = undefined;
        }
        allowedge = {};
        queue = [];
        for (let v of gnodes) {
            if (!(v in mate) && label[inblossom[v] as string] === undefined) {
                assignLabel(v, 1, null);
            }
        }
        let augmented = 0;
        while (true) {
            console.log(`Inner 1 ${queue.length}`);
            while (queue.length > 0 && !augmented) {
                console.log("Inner queue");
                let v = queue.pop()!;
                console.log(v);
                if (label[inblossom[v] as string] !== 1) {
                    throw new Error("Label is not 1");
                }
                for (let w of G.neighbors(v)) {
                    if (w === v) {
                        continue;
                    }
                    let bv = inblossom[v];
                    let bw = inblossom[w];
                    if (bv === bw) {
                        continue;
                    }
                    if (!allowedge[`${v}-${w}`]) {
                        let kslack = slack(v, w);
                        console.log(`kslack ${kslack}`);
                        if (kslack <= 0) {
                            allowedge[`${v}-${w}`] = allowedge[`${w}-${v}`] = true;
                        }
                    }
                    console.log(`negh ${allowedge}`);
                    if (allowedge[`${v}-${w}`]) {
                        console.log("c2");
                        if (label[bw as string] === undefined) {
                            assignLabel(w, 2, v);
                        } else if (label[bw as string] === 1) {
                            console.log("Scan");
                            let base = scanBlossom(v, w);
                            if (base !== NoNode) {
                                addBlossom(base, v, w);
                            } else {
                                augmentMatching(v, w);
                                augmented = 1;
                                break;
                            }
                        } else if (label[w] === undefined) {
                            if (label[bw as string] !== 2) {
                                throw new Error("Label is not 2");
                            }
                            label[w] = 2;
                            labeledge[w] = [v, w];
                        } else if (label[bw as string] === 1) {
                            if (bestedge[bv as string] === null || kslack < slack(...bestedge[bv as string])) {
                                bestedge[bv as string] = [v, w];
                            }
                        } else if (label[w] === undefined) {
                            if (bestedge[w] === null || kslack < slack(...bestedge[w])) {
                                bestedge[w] = [v, w];
                            }
                        }
                    }
                }
            }
            if (augmented) {
                console.log("Break!");
                break;
            }
            let deltatype = -1;
            let delta: number | null = null;
            let deltaedge: [string, string] | null = null;
            let deltablossom: Blossom | null = null;
            if (!maxcardinality) {
                deltatype = 1;
                delta = Math.min(...Object.values(dualvar));
            }
            for (let v of G.nodes()) {
                if (label[inblossom[v] as string] === undefined && bestedge[v] !== null) {
                    let d = slack(...bestedge[v]);
                    if (deltatype === -1 || d < delta) {
                        delta = d;
                        deltatype = 2;
                        deltaedge = bestedge[v];
                    }
                }
            }
            for (let b of Object.keys(blossomparent)) {
                if (blossomparent[b] === null && label[b] === 1 && bestedge[b] !== null) {
                    let kslack = slack(...bestedge[b]);
                    let d = allinteger ? kslack / 2 : kslack / 2.0;
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
                    deltablossom = blossomdual[b] as Blossom;
                }
            }
            if (deltatype === -1) {
                if (!maxcardinality) {
                    throw new Error("Deltatype violation");
                }
                deltatype = 1;
                delta = Math.max(0, Math.min(...Object.values(dualvar)));
            }
            for (let v of gnodes) {
                if (label[inblossom[v] as string] === 1) {
                    dualvar[v] -= delta;
                } else if (label[inblossom[v] as string] === 2) {
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
                if (label[inblossom[v] as string] !== 1) {
                    throw new Error("Label is not 1");
                }
                allowedge[`${v}-${w}`] = allowedge[`${w}-${v}`] = true;
                queue.push(v);
            } else if (deltatype === 3) {
                let [v, w] = deltaedge;
                allowedge[`${v}-${w}`] = allowedge[`${w}-${v}`] = true;
                if (label[inblossom[v] as string] !== 1) {
                    throw new Error("Label is not 1");
                }
                queue.push(v);
            } else if (deltatype === 4) {
                if (!(deltablossom instanceof Blossom)) {
                    throw new Error("Deltablossom is not a Blossom");
                }
                expandBlossom(deltablossom, false);
            }
        }
        for (let v of Object.keys(mate)) {
            if (mate[mate[v]] !== v) {
                throw new Error("Matching violation");
            }
        }
        if (!augmented) {
            break;
        }
        for (let b of Object.keys(blossomdual)) {
            if (blossomparent[b] === null && label[b] === 1 && blossomdual[b] > 0) {
                if (!(blossomdual[b] instanceof Blossom)) {
                    throw new Error("Blossom dual is not a Blossom");
                }
                expandBlossom(blossomdual[b], true);
            }
        }
    }
    if (allinteger) {
        verifyOptimum();
    }
    return matchingDictToSet(mate);
}

function matchingDictToSet(mate: { [key: string]: string | null }): Set<[string, string]> {
    let matching = new Set<[string, string]>();
    for (let [v, w] of Object.entries(mate)) {
        if (w !== null && v < w) {
            matching.add([v, w]);
        }
    }
    return matching;
}

if (require.main === module) {
    let g = new Graph();
    console.log("Graph:");
    let edges = [
        ["A", "B", 6],
        ["A", "C", 2],
        ["B", "C", 1],
        ["B", "D", 7],
        ["C", "E", 9],
        ["D", "E", 3],
    ];

    g.addWeightedEdgesFrom(edges);
    let res = maxWeightMatching(g);
    console.log("Pure TypeScript:");
    console.log(res);
    if (res.size !== 2 || !res.has(["C", "E"]) || !res.has(["B", "D"])) {
        throw new Error("Test case 1 failed");
    }

    g = new Graph();
    console.log("Graph:");
    edges = [
        ["A", "B", 6],
        ["A", "C", 2],
        ["B", "C", 1],
        ["B", "D", 2],
        ["C", "E", 9],
        ["D", "E", 3],
    ];

    g.addWeightedEdgesFrom(edges);
    res = maxWeightMatching(g);
    console.log("Pure TypeScript:");
    console.log(res);
    if (res.size !== 2 || !res.has(["C", "E"]) || !res.has(["A", "B"])) {
        throw new Error("Test case 2 failed");
    }
}
```

This TypeScript code closely follows the structure of the original Python code. It uses classes and functions to represent the graph, blossoms, and the matching algorithm. The code includes type annotations and error handling to ensure correctness.