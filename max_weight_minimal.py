"""
Minimal version of https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.matching.max_weight_matching.html
Removed all comments
"""


class NoNode:
    """Dummy value which is different from any node."""


class Blossom:
    """Representation of a non-trivial blossom or sub-blossom."""

    __slots__ = ["childs", "edges", "mybestedges"]

    def leaves(self):
        stack = [*self.childs]
        while stack:
            t = stack.pop()
            if isinstance(t, Blossom):
                stack.extend(t.childs)
            else:
                yield t


class Graph:
    def __init__(self):
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

    def __iter__(self):
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

    def __getitem__(self, node: str) -> tuple[dict[str, int], int]:
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
        # print(f"{dualvar_v=} {dualvar_w=} {weight=}")
        return dualvar_v + dualvar_w - weight

    def assignLabel(w: NodeType, t: int, v: NullableNodeType):
        print(f"assignLabel({w}, {t})")
        b = inblossom[w]
        assert label.get(w) is None and label.get(b) is None
        label[w] = label[b] = t
        if v is not None:
            labeledge[w] = labeledge[b] = (v, w)
        else:
            labeledge[w] = labeledge[b] = None
        bestedge[w] = bestedge[b] = None
        if t == 1:
            if isinstance(b, Blossom):
                leaves = list(b.leaves())
                print(f"Adding leaves to queue ({len(queue)}) {len(leaves)}")
                queue.extend(leaves)
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
        print("Adding blossom for node:", base, "with parent:", v, "and child:", w)
        bb = inblossom[base]
        bv = inblossom[v]
        bw = inblossom[w]
        b = Blossom()
        blossombase[b] = base
        blossomparent[b] = None
        blossomparent[bb] = b
        b.childs = path = []
        b.edges = edgs = [(v, w)]
        while bv != bb:
            blossomparent[bv] = b
            path.append(bv)
            edgs.append(labeledge[bv])
            assert label[bv] == 2 or (
                label[bv] == 1 and labeledge[bv][0] == mate[blossombase[bv]]
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
                label[bw] == 1 and labeledge[bw][0] == mate[blossombase[bw]]
            )
            w = labeledge[bw][0]
            bw = inblossom[w]
        assert label[bb] == 1
        label[b] = 1
        labeledge[b] = labeledge[bb]
        blossomdual[b] = 0
        for v in b.leaves():
            if label[inblossom[v]] == 2:
                print(f"Adding to queue ({len(queue)}) {v}")
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
            print(f"Expanding Blossom: {len(b.childs)}")
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
                    print("Assigning label in expandBlossom")
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
        print(f"Expanded Blossom: {len(b.childs)}")

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
        print(f"Augment matching {v} {w}")
        for s, j in ((v, w), (w, v)):
            print(f"{s=} {j=}")
            while 1:
                bs = inblossom[s]
                print(f"{s=} {bs=}")
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
            if mate.get(i) == j or mate.get(j) == i:
                assert mate[i] == j and mate[j] == i
                assert s == 0
        for v in gnodes:
            assert (v in mate) or dualvar[v] + vdualoffset == 0
        for b in blossomdual:
            if blossomdual[b] > 0:
                assert len(b.edges) % 2 == 1
                for i, j in b.edges[1::2]:
                    assert mate[i] == j and mate[j] == i

    while 1:
        print("Outer loop iteration")
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
            print(f"Inner loop 1 (BFS), queue length: {len(queue)}")

            while queue and not augmented:
                v = queue.pop()
                print(f"Processing node from stack: {v}")
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
                        # print(f"Slack for ({v}, {w}): {kslack:.2f}")
                        if kslack <= 0:
                            allowedge[(v, w)] = allowedge[(w, v)] = True
                    if (v, w) in allowedge:
                        print(f"Edge ({v},{w}) is allowed.")
                        if label.get(bw) is None:
                            assignLabel(w, 2, v)
                        elif label.get(bw) == 1:
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
                print("Augmented, breaking from inner dual adjustment loop.")
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
                        print(f"Deltatype 3: {d=:.3e} {kslack=:.3e}")
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

            print(f"Duals updated with {delta=:.3e}, type={deltatype}")
            if deltatype == 1:
                break
            elif deltatype == 2:
                (v, w) = deltaedge
                assert label[inblossom[v]] == 1
                allowedge[(v, w)] = allowedge[(w, v)] = True
                print(f"{deltatype=} adding to queue ({len(queue)}) {v}")
                queue.append(v)
            elif deltatype == 3:
                (v, w) = deltaedge
                allowedge[(v, w)] = True
                allowedge[(w, v)] = True
                assert label[inblossom[v]] == 1
                print(f"{deltatype=} adding to queue ({len(queue)}) {v}")
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
            if blossomparent[b] is None and label.get(b) == 1 and blossomdual[b] == 0:
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
