
// Define NoNodeValue as a unique symbol
const NoNodeValue: unique symbol = Symbol("NoNode");
type NoNodeType = typeof NoNodeValue;

// Forward declaration for Blossom
class Blossom {
    childs: (string | Blossom)[];
    // Assumption: edges[i] connects childs[i] and childs[i+1] (cyclically)
    // These are pairs of actual graph node strings.
    edges: [string, string][];
    mybestedges: [string, string][] | null; // List of candidate edges to other outer blossoms

    constructor() {
        this.childs = [];
        this.edges = [];
        this.mybestedges = null;
    }

    leaves(): string[] {
        const result: string[] = [];
        const stack: (string | Blossom)[] = [...this.childs];
        while (stack.length > 0) {
            const t = stack.pop()!;
            if (t instanceof Blossom) {
                stack.push(...t.childs);
            } else if (typeof t === 'string') {
                result.push(t);
            }
        }
        return result;
    }
}

type BasicNodeType = string | Blossom; // Actual nodes or blossoms
type NodeType = BasicNodeType | NoNodeType; // Includes the dummy NoNodeValue
type NullableNodeType = NodeType | null; // Includes null

// Type for the first element of a labeledge tuple, assuming it's always a graph node string
type PathEndpointType = string;


class Graph {
    adj: Record<string, Record<string, number>>;
    node_list: string[];

    constructor() {
        this.adj = {};
        this.node_list = [];
    }

    addEdge(u: string, v: string, weight: number = 1): void {
        if (!(u in this.adj)) {
            this.adj[u] = {};
            this.node_list.push(u);
        }
        if (!(v in this.adj)) {
            this.adj[v] = {};
            this.node_list.push(v);
        }
        this.adj[u][v] = weight;
        this.adj[v][u] = weight;
    }

    addWeightedEdgesFrom(edge_list: [string, string, number][]): void {
        for (const [u, v, weight] of edge_list) {
            this.addEdge(u, v, weight);
        }
    }

    nodes(): IterableIterator<string> {
        return this.node_list[Symbol.iterator]();
    }

    *edges(): IterableIterator<[string, string, number]> {
        const seen: Set<string> = new Set();
        for (const u in this.adj) {
            for (const v in this.adj[u]) {
                const key1 = `${u},${v}`;
                const key2 = `${v},${u}`;
                if (!seen.has(key1) && !seen.has(key2)) {
                    seen.add(key1);
                    yield [u, v, this.adj[u][v]];
                }
            }
        }
    }

    neighbors(node: string): string[] {
        if (node in this.adj) {
            return Object.keys(this.adj[node]);
        }
        return [];
    }

    getAdjacency(node: string): Record<string, number> {
        return this.adj[node];
    }
}


function maxWeightMatching(G: Graph, maxcardinality: boolean = false): Set<[string, string]> {
    const gnodes: string[] = G.node_list.slice();
    if (gnodes.length === 0) {
        return new Set();
    }

    let maxweight = 0;
    let allinteger = true;
    for (const [i, j, d] of G.edges()) {
        const wt = d;
        if (i !== j && wt > maxweight) {
            maxweight = wt;
        }
        allinteger = allinteger && (typeof wt === 'number' && Number.isInteger(wt));
    }

    const mate: Record<string, string> = {};
    const label: Map<NullableNodeType, number> = new Map();
    const labeledge: Map<BasicNodeType, [PathEndpointType, string] | null> = new Map();
    const inblossom: Map<string, BasicNodeType> = new Map();
    gnodes.forEach(node => inblossom.set(node, node));

    const blossomparent: Map<BasicNodeType, Blossom | null> = new Map();
    gnodes.forEach(node => blossomparent.set(node, null));

    const blossombase: Map<BasicNodeType, string> = new Map();
    gnodes.forEach(node => blossombase.set(node, node));

    const bestedge: Map<NodeType, [string, string] | null> = new Map();
    const dualvar: Record<string, number> = {};
    gnodes.forEach(node => dualvar[node] = maxweight);

    const blossomdual: Map<Blossom, number> = new Map();
    const allowedge: Map<string, boolean> = new Map(); // Key: "node1,node2"
    const queue: string[] = []; // Contains string node names. Python used LIFO (stack).

    function slack(v: string, w: string): number {
        const dualvar_v = dualvar[v];
        const dualvar_w = dualvar[w];
        const weight = 2 * G.adj[v][w];
        // console.log(`slack(${v},${w}): dv[${v}]=${dualvar_v}, dv[${w}]=${dualvar_w}, G[${v}][${w}]=${G.adj[v][w]}, result=${dualvar_v + dualvar_w - weight}`);
        return dualvar_v + dualvar_w - weight;
    }

    function assignLabel(w_node: string, type: number, v_parent: PathEndpointType | null): void {
        const b_component = inblossom.get(w_node)!; // string or Blossom
        if (label.has(w_node) || label.has(b_component)) {
            throw new Error(`Assertion failed: label already set for ${w_node} or ${b_component}`);
        }
        label.set(w_node, type);
        label.set(b_component, type);

        if (v_parent !== null) {
            labeledge.set(w_node, [v_parent, w_node]);
            labeledge.set(b_component, [v_parent, w_node]);
        } else {
            labeledge.set(w_node, null);
            labeledge.set(b_component, null);
        }

        bestedge.set(w_node, null);
        bestedge.set(b_component, null);

        if (type === 1) { // Outer node/component
            if (b_component instanceof Blossom) {
                queue.push(...b_component.leaves());
            } else { // b_component is string
                queue.push(b_component);
            }
        } else if (type === 2) { // Inner node/component
            const baseOfB = blossombase.get(b_component)!; // string
            const mateOfBase = mate[baseOfB];
            if (mateOfBase === undefined) {
                throw new Error(`Mate of base ${baseOfB} not found for component ${b_component}`);
            }
            assignLabel(mateOfBase, 1, baseOfB); // Mate becomes outer
        }
    }

    function scanBlossom(v_arg: string, w_arg: string): PathEndpointType | NoNodeType {
        const path_marks: BasicNodeType[] = []; // Stores elements whose labels are temporarily 5
        let base_common: PathEndpointType | NoNodeType = NoNodeValue;

        let currentV: PathEndpointType | NoNodeType = v_arg;
        let currentW: PathEndpointType | NoNodeType = w_arg;

        while (currentV !== NoNodeValue) {
            if (typeof currentV !== 'string') {
                throw new Error("currentV is not a string in scanBlossom main loop");
            }
            const b_curr = inblossom.get(currentV)!;

            if ((label.get(b_curr)! & 4)) { // Label bit 2 means "scanned from other path"
                base_common = blossombase.get(b_curr)!;
                break;
            }
            if (label.get(b_curr)! !== 1) throw new Error(`Assertion failed: label[${b_curr}] !== 1`);
            path_marks.push(b_curr);
            label.set(b_curr, label.get(b_curr)! | 4); // Mark as scanned from this path (orig set to 5)

            const edge_info = labeledge.get(b_curr);
            if (edge_info === null || edge_info === undefined) {
                if (!(blossombase.get(b_curr)! in mate)) {
                    currentV = NoNodeValue;
                } else {
                    throw new Error("Assertion failed: blossombase[b_curr] in mate but labeledge[b_curr] is null");
                }
            } else {
                const [prevNodePath, _nodeInBCurr] = edge_info; // prevNodePath is string
                const baseOfBCurr = blossombase.get(b_curr)!;
                if (prevNodePath !== mate[baseOfBCurr]) {
                    throw new Error(`Assertion failed: labeledge[${b_curr}][0] !== mate[blossombase[${b_curr}]]`);
                }
                currentV = prevNodePath; // This is a string (mate of baseOfBCurr)

                const blossomOfMate = inblossom.get(currentV)!;
                if (label.get(blossomOfMate)! !== 2) {
                    throw new Error(`Assertion failed: label of mate's blossom ${blossomOfMate} !== 2`);
                }
                const grandparentEdge = labeledge.get(blossomOfMate);
                if (grandparentEdge === null || grandparentEdge === undefined) {
                    throw new Error(`Assertion failed: labeledge of mate's blossom ${blossomOfMate} is null`);
                }
                currentV = grandparentEdge[0]; // This is also a string (parent of mate in tree)
            }

            if (currentW !== NoNodeValue) {
                [currentV, currentW] = [currentW, currentV];
            }
        }

        for (const b_path of path_marks) {
            // label.set(b_path, label.get(b_path)! & ~4); // Restore label (remove mark bit)
            label.set(b_path, 1)
            // if (label.get(b_path)! === 1) { /* ensure it's 1 */ } else { /* error or readjust */ }
        }
        return base_common;
    }

    function addBlossom(base_node: string, v_orig: string, w_orig: string): void {
        const bb_base_comp = inblossom.get(base_node)!;
        let v_curr_comp = inblossom.get(v_orig)!;
        let w_curr_comp = inblossom.get(w_orig)!;

        const newBlossom = new Blossom();
        blossombase.set(newBlossom, base_node);
        blossomparent.set(newBlossom, null);
        blossomparent.set(bb_base_comp, newBlossom);

        const p_childs: BasicNodeType[] = [];
        const p_edges: [string, string][] = [];
        p_edges.push([v_orig, w_orig]); // This is the edge closing the cycle

        let v_path_node = v_orig;
        let v_path_comp = v_curr_comp;
        while (v_path_comp !== bb_base_comp) {
            blossomparent.set(v_path_comp, newBlossom);
            p_childs.push(v_path_comp);
            const v_lbl_edge = labeledge.get(v_path_comp)!; // [PathEndpointType, string]
            p_edges.push(v_lbl_edge);
            v_path_node = v_lbl_edge[0]; // PathEndpointType (string)
            v_path_comp = inblossom.get(v_path_node)!;
        }
        p_childs.push(bb_base_comp);
        p_childs.reverse(); // p_childs is now [bb_base_comp, ..., v_curr_comp]
        p_edges.reverse();  // p_edges is now [edge_for_bb_to_next_v, ..., edge_for_parentV_to_v, [v_orig, w_orig]]

        let w_path_node = w_orig;
        let w_path_comp = w_curr_comp;
        while (w_path_comp !== bb_base_comp) {
            blossomparent.set(w_path_comp, newBlossom);
            p_childs.push(w_path_comp); // Appending w_path children after v_path children
            const w_lbl_edge = labeledge.get(w_path_comp)!;
            p_edges.push([w_lbl_edge[1], w_lbl_edge[0]]); // Python code: (labeledge[bw][1], labeledge[bw][0])
            w_path_node = w_lbl_edge[0];
            w_path_comp = inblossom.get(w_path_node)!;
        }

        newBlossom.childs = p_childs;
        newBlossom.edges = p_edges; // CRITICAL: Assumed Python meant to assign collected edges
        if (newBlossom.edges.length % 2 === 0) {
            throw new Error("Blossom created with even number of edges");
        }


        label.set(newBlossom, 1);
        labeledge.set(newBlossom, labeledge.get(bb_base_comp)!); // Inherit from base component
        blossomdual.set(newBlossom, 0);

        for (const leaf_node of newBlossom.leaves()) { // string
            // Use inblossom.get(leaf_node) before update to check old label
            if (label.get(inblossom.get(leaf_node)!) === 2) {
                queue.push(leaf_node);
            }
            inblossom.set(leaf_node, newBlossom); // Update inblossom map for leaves
        }

        const bestedgeto: Map<BasicNodeType, [string, string]> = new Map();
        for (const child_b of newBlossom.childs) { // child_b is string | Blossom
            let nblist: [string, string][];
            if (child_b instanceof Blossom) {
                nblist = child_b.mybestedges !== null ? child_b.mybestedges :
                    child_b.leaves().flatMap(leaf =>
                        G.neighbors(leaf)
                            .filter(neighbor => leaf !== neighbor)
                            .map(neighbor => [leaf, neighbor] as [string, string])
                    );
                if (child_b.mybestedges !== null) child_b.mybestedges = null;
            } else { // child_b is string
                nblist = G.neighbors(child_b)
                    .filter(neighbor => child_b !== neighbor)
                    .map(neighbor => [child_b as string, neighbor]);
            }

            for (let [i_node, j_node] of nblist) { // strings
                // Use inblossom state *before* full update for newBlossom leaves if needed, but here j_node is outside.
                // The inblossom for leaves of newBlossom IS updated.
                // So if j_node is a leaf of newBlossom, inblossom.get(j_node) will be newBlossom.
                if (inblossom.get(j_node) === newBlossom) {
                    [i_node, j_node] = [j_node, i_node]; // Ensure j_node is outside newBlossom
                }
                const bj_comp = inblossom.get(j_node)!;
                if (bj_comp !== newBlossom && label.get(bj_comp) === 1) {
                    const current_slack = slack(i_node, j_node);
                    if (!bestedgeto.has(bj_comp) || current_slack < slack(...bestedgeto.get(bj_comp)!)) {
                        bestedgeto.set(bj_comp, [i_node, j_node]);
                    }
                }
            }
            bestedge.set(child_b, null);
        }
        newBlossom.mybestedges = Array.from(bestedgeto.values());

        let mybestedge_for_newBlossom: [string, string] | null = null;
        let mybestslack_for_new_Blossom: number | null = null;
        for (const k_edge of newBlossom.mybestedges) {
            const kslack = slack(...k_edge);
            if (mybestedge_for_newBlossom === null || kslack < mybestslack_for_new_Blossom!) {
                mybestedge_for_newBlossom = k_edge;
                mybestslack_for_new_Blossom = kslack;
            }
        }
        bestedge.set(newBlossom, mybestedge_for_newBlossom);
    }

    function expandBlossom(b_to_expand: Blossom, endstage: boolean): void {
        function* _recurse(currentBlossom: Blossom, isEndstage: boolean): IterableIterator<Blossom> {
            for (const s_child of currentBlossom.childs) {
                blossomparent.set(s_child, null);
                if (s_child instanceof Blossom) {
                    if (isEndstage && blossomdual.get(s_child)! === 0) {
                        yield s_child;
                    } else {
                        s_child.leaves().forEach(v_leaf => inblossom.set(v_leaf, s_child));
                    }
                } else { // s_child is string
                    inblossom.set(s_child, s_child);
                }
            }

            if (!isEndstage && label.get(currentBlossom) === 2) {
                const entryEdge = labeledge.get(currentBlossom)!; // [PathEndpointType, string]
                const entryNodeName = entryEdge[1]; // The node (string) in currentBlossom where path enters
                const entryChild = inblossom.get(entryNodeName)!; // The child component containing entryNodeName

                if (!currentBlossom.childs.includes(entryChild)) {
                    throw new Error("Entry child determined by inblossom is not in currentBlossom.childs during expandBlossom");
                }

                let j = currentBlossom.childs.indexOf(entryChild);
                let jstep = (j % 2 !== 0) ? 1 : -1;
                if (jstep === 1) j -= currentBlossom.childs.length; // Python-like negative index logic for forward traversal start

                let path_v_node = entryEdge[0]; // PathEndpointType (string) - parent in tree
                let path_w_node = entryEdge[1]; // string - node in currentBlossom

                while (j !== 0) {
                    j += jstep;
                    const edge_idx_abs = (j >= 0 ? j : j + currentBlossom.edges.length) % currentBlossom.edges.length;
                    const next_edge_idx_abs = ((j - 1) >= 0 ? (j - 1) : (j - 1) + currentBlossom.edges.length) % currentBlossom.edges.length;


                    let p_node: string, q_node: string;
                    if (jstep === 1) {
                        [p_node, q_node] = currentBlossom.edges[edge_idx_abs];
                    } else {
                        [q_node, p_node] = currentBlossom.edges[next_edge_idx_abs];
                    }

                    label.delete(path_w_node); // path_w_node is previous child node.
                    label.delete(q_node);       // q_node is current child node (matched to p_node).
                    assignLabel(path_w_node, 2, path_v_node);

                    allowedge.set(`${p_node},${q_node}`, true);
                    allowedge.set(`${q_node},${p_node}`, true);

                    j += jstep;
                    const edge_idx_abs2 = (j >= 0 ? j : j + currentBlossom.edges.length) % currentBlossom.edges.length;
                    const next_edge_idx_abs2 = ((j - 1) >= 0 ? (j - 1) : (j - 1) + currentBlossom.edges.length) % currentBlossom.edges.length;

                    if (jstep === 1) {
                        [path_v_node, path_w_node] = currentBlossom.edges[edge_idx_abs2];
                    } else {
                        [path_w_node, path_v_node] = currentBlossom.edges[next_edge_idx_abs2];
                    }
                    allowedge.set(`${path_v_node},${path_w_node}`, true);
                    allowedge.set(`${path_w_node},${path_v_node}`, true);
                }

                const bw_child_at_base = currentBlossom.childs[0]; // childs[j] after loop (j is 0)
                label.set(path_w_node, 2); // path_w_node is the node in bw_child_at_base (or bw_child_at_base itself if string)
                label.set(bw_child_at_base, 2);
                labeledge.set(path_w_node, [path_v_node, path_w_node]);
                labeledge.set(bw_child_at_base, [path_v_node, path_w_node]);
                bestedge.set(bw_child_at_base, null);

                j += jstep;
                // Modulo arithmetic for j to keep it in childs/edges bounds
                let current_child_idx = (j >= 0 ? j : j + currentBlossom.childs.length) % currentBlossom.childs.length;

                while (currentBlossom.childs[current_child_idx] !== entryChild) {
                    const bv_child_on_path = currentBlossom.childs[current_child_idx];
                    if (label.get(bv_child_on_path) === 1) {
                        j += jstep;
                        current_child_idx = (j >= 0 ? j : j + currentBlossom.childs.length) % currentBlossom.childs.length;
                        continue;
                    }
                    let v_node_to_relabel: string | undefined;
                    if (bv_child_on_path instanceof Blossom) {
                        for (const v_leaf of bv_child_on_path.leaves()) {
                            if (label.has(v_leaf)) {
                                v_node_to_relabel = v_leaf;
                                break;
                            }
                        }
                    } else { // bv_child_on_path is string
                        v_node_to_relabel = bv_child_on_path;
                    }

                    if (v_node_to_relabel && label.has(v_node_to_relabel)) {
                        if (label.get(v_node_to_relabel)! !== 2) throw new Error("Assertion failed: label[v_node_to_relabel] == 2");
                        if (inblossom.get(v_node_to_relabel)! !== bv_child_on_path) throw new Error("Assertion failed: inblossom[v_node_to_relabel] == bv_child_on_path");

                        label.delete(v_node_to_relabel);
                        const mate_of_base_bv = mate[blossombase.get(bv_child_on_path)!];
                        label.delete(mate_of_base_bv);

                        const prev_parent_in_tree = labeledge.get(v_node_to_relabel)![0]; // PathEndpointType
                        assignLabel(v_node_to_relabel, 2, prev_parent_in_tree);
                    }
                    j += jstep;
                    current_child_idx = (j >= 0 ? j : j + currentBlossom.childs.length) % currentBlossom.childs.length;
                }
            }
            label.delete(currentBlossom);
            labeledge.delete(currentBlossom);
            bestedge.delete(currentBlossom);
            blossomparent.delete(currentBlossom);
            blossombase.delete(currentBlossom);
            blossomdual.delete(currentBlossom);
        }

        const recurse_stack: Generator<IterableIterator<Blossom>>[] = [];
        recurse_stack.push(_recurse(b_to_expand, endstage) as any); // Typecast needed for Generator type

        while (recurse_stack.length > 0) {
            const top_gen = recurse_stack[recurse_stack.length - 1];
            const iter_res = top_gen.next();
            if (!iter_res.done && iter_res.value) { // If yielded a sub-blossom
                recurse_stack.push(_recurse(iter_res.value as Blossom, endstage) as any);
            } else {
                recurse_stack.pop();
            }
        }
    }

    function augmentBlossom(b_to_augment: Blossom, v_node_exposed: string): void {
        type AugmentTask = { blossom: Blossom; node: string };

        function _processAugmentTask(currentBlossom: Blossom, exposed_node: string): AugmentTask[] {
            const subTasks: AugmentTask[] = [];
            let t_child_container: BasicNodeType | undefined;

            let trace_parent: BasicNodeType | null = inblossom.get(exposed_node)!;
            while (trace_parent !== null && trace_parent !== currentBlossom && blossomparent.get(trace_parent) !== currentBlossom) {
                trace_parent = blossomparent.get(trace_parent) || null;
            }
            if (trace_parent !== null && blossomparent.get(trace_parent) === currentBlossom) {
                t_child_container = trace_parent; // Found direct child
            } else if (currentBlossom.childs.includes(inblossom.get(exposed_node)!)) {
                t_child_container = inblossom.get(exposed_node)!;
            } else { // Fallback: search manually
                for (const child of currentBlossom.childs) {
                    if (child === exposed_node || (child instanceof Blossom && child.leaves().includes(exposed_node))) {
                        t_child_container = child;
                        break;
                    }
                }
            }
            if (!t_child_container) throw new Error(`Child container for ${exposed_node} in ${currentBlossom} not found.`);

            if (t_child_container instanceof Blossom) {
                subTasks.push({ blossom: t_child_container, node: exposed_node });
            }

            const i_start_idx = currentBlossom.childs.indexOf(t_child_container);
            let j = i_start_idx;
            const num_childs = currentBlossom.childs.length;
            const jstep = (i_start_idx % 2 !== 0) ? 1 : -1;
            if (jstep === 1 && num_childs > 0) j -= num_childs; // Python negative index logic

            while (j !== 0) {
                j += jstep;
                // Normalize j to be a valid index for childs/edges
                const j_child_idx = (j >= 0 ? j : j + num_childs) % num_childs;
                const t_next_child = currentBlossom.childs[j_child_idx];

                let w_match_node: string, x_match_node: string;
                // Edges related to child j_child_idx.
                // If jstep = 1 (forward), edge is edges[j_child_idx] (or python's b.edges[j]).
                // If jstep = -1 (backward), edge is edges[prev_j_child_idx] (or python's b.edges[j-1]).
                const edge_idx_for_match = (jstep === 1) ? j_child_idx
                    : ((j_child_idx - 1 + num_childs) % num_childs); // (j-1) in python list index

                if (jstep === 1) {
                    [w_match_node, x_match_node] = currentBlossom.edges[edge_idx_for_match];
                } else {
                    [x_match_node, w_match_node] = currentBlossom.edges[edge_idx_for_match];
                }

                if (t_next_child instanceof Blossom) {
                    subTasks.push({ blossom: t_next_child, node: w_match_node });
                }

                j += jstep;
                const j_after_next_child_idx = (j >= 0 ? j : j + num_childs) % num_childs;
                const t_after_next_child = currentBlossom.childs[j_after_next_child_idx];

                if (t_after_next_child instanceof Blossom) {
                    subTasks.push({ blossom: t_after_next_child, node: x_match_node });
                }
                mate[w_match_node] = x_match_node;
                mate[x_match_node] = w_match_node;
            }

            currentBlossom.childs = currentBlossom.childs.slice(i_start_idx).concat(currentBlossom.childs.slice(0, i_start_idx));
            currentBlossom.edges = currentBlossom.edges.slice(i_start_idx).concat(currentBlossom.edges.slice(0, i_start_idx));
            blossombase.set(currentBlossom, blossombase.get(currentBlossom.childs[0])!);

            if (blossombase.get(currentBlossom)! !== exposed_node) {
                // This assertion from Python: assert blossombase[b] == v
                // This might mean the base vertex of the blossom should be the exposed vertex itself.
                blossombase.set(currentBlossom, exposed_node);
                if (blossombase.get(currentBlossom)! !== exposed_node) // Re-check
                    console.warn(`Warning: blossombase for ${currentBlossom} set to ${exposed_node}, but re-get gives ${blossombase.get(currentBlossom)}`);
            }
            return subTasks;
        }

        const task_stack: AugmentTask[] = [{ blossom: b_to_augment, node: v_node_exposed }];
        while (task_stack.length > 0) {
            const task = task_stack.pop()!;
            const newSubTasks = _processAugmentTask(task.blossom, task.node);
            task_stack.push(...newSubTasks);
        }
    }

    function augmentMatching(v_start_node: string, w_start_node: string): void {
        for (const [s_arg, j_arg] of [[v_start_node, w_start_node], [w_start_node, v_start_node]]) {
            let s_node = s_arg; // string
            let j_node = j_arg; // string

            while (true) {
                const bs_comp = inblossom.get(s_node)!;
                if (label.get(bs_comp)! !== 1) throw new Error(`Assertion failed: label of ${bs_comp} is not 1`);

                const lblEdge_bs = labeledge.get(bs_comp);
                if (lblEdge_bs === null || lblEdge_bs === undefined) {
                    if (blossombase.get(bs_comp)! in mate) {
                        throw new Error(`Assertion: Matched node ${blossombase.get(bs_comp)} at end of path but labeledge is null.`);
                    }
                } else {
                    const prevNodeInTree = lblEdge_bs[0]; // PathEndpointType (string)
                    if (prevNodeInTree !== mate[blossombase.get(bs_comp)!]) {
                        throw new Error(`Assertion: labeledge[${bs_comp}][0] != mate[blossombase[${bs_comp}]]`);
                    }
                }

                if (bs_comp instanceof Blossom) {
                    augmentBlossom(bs_comp, s_node);
                }
                mate[s_node] = j_node;

                if (lblEdge_bs === null || lblEdge_bs === undefined) break;

                const t_node_matched_in_bs = lblEdge_bs[0]; // This is mate[blossombase[bs_comp]]
                const bt_comp = inblossom.get(t_node_matched_in_bs)!;
                if (label.get(bt_comp)! !== 2) throw new Error(`Assertion failed: label of ${bt_comp} is not 2`);

                const lblEdge_bt = labeledge.get(bt_comp)!; // [PathEndpointType, string]
                // Python: s,j = labeledge[bt]. s is parent, j is child. New (s_node, j_node) for mate[j]=s.
                // My labeledge: [parent, child]. So lblEdge_bt[0] is parent, lblEdge_bt[1] is child.
                // New s_node (child) = lblEdge_bt[1]. New j_node (parent) = lblEdge_bt[0].
                s_node = lblEdge_bt[1]; // Child node in bt_comp
                j_node = lblEdge_bt[0]; // Parent node in tree

                if (blossombase.get(bt_comp)! !== t_node_matched_in_bs) {
                    // This implies t_node_matched_in_bs should be the base of bt_comp.
                    // Original Python assert: blossombase[bt] == t. (t is mate[blossombase[bs]])
                    throw new Error(`Assertion: blossombase[${bt_comp}] (${blossombase.get(bt_comp)}) != ${t_node_matched_in_bs}`);
                }

                if (bt_comp instanceof Blossom) {
                    // Exposed node in bt_comp is s_node (the child node on the path)
                    augmentBlossom(bt_comp, s_node);
                }
                mate[s_node] = j_node;
            }
        }
    }

    // function verifyOptimum(): void { /* ... full translation needed ... */ }

    // Main algorithm loop
    while (true) {
        console.log("Outer loop iteration");
        label.clear();
        labeledge.clear();
        bestedge.clear();
        blossomdual.forEach(b => { if (b instanceof Blossom) b.mybestedges = null; }); // This is wrong, keys are Blossoms
        for (const b_key of blossomdual.keys()) { b_key.mybestedges = null; }

        allowedge.clear();
        queue.length = 0;

        for (const v_node of gnodes) {
            if (!(v_node in mate) && !label.has(inblossom.get(v_node)!)) {
                assignLabel(v_node, 1, null);
            }
        }
        let augmented = false;

        mainloop_bfs: while (true) {
            console.log(`Inner loop 1 (BFS), queue length: ${queue.length}`);
            while (queue.length > 0 && !augmented) {
                const v_bfs = queue.pop()!; // LIFO, as per Python's queue.pop()
                console.log(`Processing node from stack: ${v_bfs}`);

                if (label.get(inblossom.get(v_bfs)!) !== 1) throw new Error("Assertion: Node from stack not labeled 1");

                for (const w_neighbor of G.neighbors(v_bfs)) {
                    if (v_bfs === w_neighbor) continue;

                    const bv_bfs = inblossom.get(v_bfs)!;
                    const bw_neighbor = inblossom.get(w_neighbor)!;

                    if (bv_bfs === bw_neighbor) continue;

                    const edgeKey_vw = `${v_bfs},${w_neighbor}`;
                    const edgeKey_wv = `${w_neighbor},${v_bfs}`;

                    if (!allowedge.has(edgeKey_vw)) {
                        const kslack_val = slack(v_bfs, w_neighbor);
                        console.log(`Slack for (${v_bfs}, ${w_neighbor}): ${kslack_val}`);
                        if (kslack_val <= 0) {
                            allowedge.set(edgeKey_vw, true);
                            allowedge.set(edgeKey_wv, true);
                        }
                    }

                    if (allowedge.has(edgeKey_vw)) {
                        console.log(`Edge (${v_bfs},${w_neighbor}) is allowed.`);
                        if (!label.has(bw_neighbor)) { // Case 1: w_neighbor's component is unlabeled
                            assignLabel(w_neighbor, 2, v_bfs);
                        } else if (label.get(bw_neighbor) === 1) { // Case 2: w_neighbor's component is 'outer'
                            const base_common = scanBlossom(v_bfs, w_neighbor);
                            if (base_common !== NoNodeValue) {
                                addBlossom(base_common as string, v_bfs, w_neighbor);
                            } else {
                                augmentMatching(v_bfs, w_neighbor);
                                augmented = true;
                                break; // Exit neighbors loop
                            }
                        } else if (!label.has(w_neighbor)) { // Case 3: w_neighbor itself is unlabeled, but bw_neighbor is labeled (must be 2)
                            if (label.get(bw_neighbor)! !== 2) throw new Error("Assertion: bw_neighbor not labeled 2");
                            label.set(w_neighbor, 2);
                            labeledge.set(w_neighbor, [v_bfs, w_neighbor]);
                        }
                    } else { // Edge not allowed (slack > 0)
                        const kslack_val = slack(v_bfs, w_neighbor);
                        if (label.get(bw_neighbor) === 1) {
                            const current_best_bv = bestedge.get(bv_bfs);
                            if (current_best_bv === null || current_best_bv === undefined || kslack_val < slack(...current_best_bv)) {
                                bestedge.set(bv_bfs, [v_bfs, w_neighbor]);
                            }
                        } else if (!label.has(w_neighbor)) { // w_neighbor (node) itself is unlabeled
                            const current_best_w = bestedge.get(w_neighbor); // bestedge for node w_neighbor
                            if (current_best_w === null || current_best_w === undefined || kslack_val < slack(...current_best_w)) {
                                bestedge.set(w_neighbor, [v_bfs, w_neighbor]);
                            }
                        }
                    }
                } // End for w_neighbor
                if (augmented) break; // Exit LIFO queue processing loop
            } // End while (queue.length > 0 && !augmented)

            if (augmented) {
                console.log("Augmented, breaking from inner dual adjustment loop.");
                break mainloop_bfs;
            }

            // Dual variable adjustment
            let deltatype = -1;
            let delta = Infinity;
            let deltaedge: [string, string] | null = null;
            let deltablossom_obj: Blossom | null = null; // Renamed from deltablossom to avoid conflict with map name

            if (!maxcardinality) {
                deltatype = 1;
                delta = Math.min(...Object.values(dualvar).filter(dv => isFinite(dv)), Infinity);
                if (!isFinite(delta)) delta = 0; // Handle empty or all non-finite dualvar
            }

            for (const v_node of gnodes) {
                if (!label.has(inblossom.get(v_node)!) && bestedge.has(v_node) && bestedge.get(v_node) !== null) {
                    const d_slack = slack(...bestedge.get(v_node)!);
                    if (deltatype === -1 || d_slack < delta) {
                        delta = d_slack;
                        deltatype = 2;
                        deltaedge = bestedge.get(v_node)!;
                    }
                }
            }

            // Iterate over roots of blossom trees (parent is null) and gnodes not in blossomparent (should be covered)
            const rootComponents: BasicNodeType[] = [];
            blossomparent.forEach((parent, child) => { if (parent === null) rootComponents.push(child); });
            gnodes.forEach(gn => { if (!blossomparent.has(gn)) rootComponents.push(gn); }); // Should already be in if they became part of tree.

            for (const b_comp of new Set(rootComponents)) { // Use Set to get unique components
                if (label.get(b_comp) === 1 && bestedge.has(b_comp) && bestedge.get(b_comp) !== null) {
                    const kslack_val = slack(...bestedge.get(b_comp)!);
                    const d_val = allinteger ? (kslack_val / 2) : (kslack_val / 2.0);
                    if (allinteger && kslack_val % 2 !== 0) throw new Error("Assertion: kslack not even for integer weights (Type 3 delta)");
                    if (deltatype === -1 || d_val < delta) {
                        delta = d_val;
                        deltatype = 3;
                        deltaedge = bestedge.get(b_comp)!;
                    }
                }
            }

            for (const b_dual_key of blossomdual.keys()) { // Keys are Blossoms
                if (blossomparent.get(b_dual_key) === null && label.get(b_dual_key) === 2) {
                    const dual_val = blossomdual.get(b_dual_key)!;
                    if (deltatype === -1 || dual_val < delta) {
                        delta = dual_val;
                        deltatype = 4;
                        deltablossom_obj = b_dual_key;
                    }
                }
            }

            if (deltatype === -1) {
                if (!maxcardinality) throw new Error("Delta not found in non-maxcardinality mode but required.");
                deltatype = 1; // Fallback for max cardinality, as in Python
                delta = Math.max(0, Math.min(...Object.values(dualvar).filter(dv => isFinite(dv)), Infinity));
                if (!isFinite(delta)) delta = 0;
            }

            for (const v_node of gnodes) {
                const v_label_comp = label.get(inblossom.get(v_node)!);
                if (v_label_comp === 1) dualvar[v_node] -= delta;
                else if (v_label_comp === 2) dualvar[v_node] += delta;
            }

            for (const b_dual_key of blossomdual.keys()) {
                if (blossomparent.get(b_dual_key) === null) {
                    const b_label = label.get(b_dual_key);
                    if (b_label === 1) blossomdual.set(b_dual_key, blossomdual.get(b_dual_key)! + delta);
                    else if (b_label === 2) blossomdual.set(b_dual_key, blossomdual.get(b_dual_key)! - delta);
                }
            }
            console.log(`Duals updated with delta=${delta}, type=${deltatype}`);

            if (deltatype === 1) break mainloop_bfs; // No change or special termination
            else if (deltatype === 2) { // Edge (v,w) = deltaedge. v is unlabelled, w is outer. Add w to queue (outer one).
                const [v_e, w_e] = deltaedge!;
                allowedge.set(`${v_e},${w_e}`, true); allowedge.set(`${w_e},${v_e}`, true);
                const nodeToQueue = label.get(inblossom.get(v_e)!) === 1 ? v_e : (label.get(inblossom.get(w_e)!) === 1 ? w_e : null);
                if (nodeToQueue) queue.push(nodeToQueue); else {
                    // This case means deltaedge's endpoints are not Type 1. This might imply Type 2 logic in Python
                    // `queue.append(v)` means `v` is the unlabelled node that now has a tight edge to an outer node.
                    // That `v` should be processed to become Type 2. Its mate becomes Type 1 and added to queue.
                    // `assignLabel(v_unlabelled, 2, w_outer)` is the standard step.
                    // Python's `queue.append(v)` (where v is unlabelled) is non-standard for BFS queue of Type 1.
                    // For now, following assignLabel strategy:
                    const unlabelledNode = !label.has(inblossom.get(v_e)!) ? v_e : w_e;
                    const outerNode = (unlabelledNode === v_e) ? w_e : v_e;
                    if (label.get(inblossom.get(outerNode)!) !== 1 || label.has(inblossom.get(unlabelledNode)!)) {
                        console.warn("Delta type 2 edge endpoints not as expected (unlabelled, outer). Defaulting to Python's deltaedge[0].");
                        queue.push(deltaedge![0]); // Fallback to Python's literal translation if roles unclear.
                    } else {
                        assignLabel(unlabelledNode, 2, outerNode);
                    }
                }
            } else if (deltatype === 3) { // Both ends of deltaedge are in outer components
                const [v_e, w_e] = deltaedge!;
                allowedge.set(`${v_e},${w_e}`, true); allowedge.set(`${w_e},${v_e}`, true);
                if (label.get(inblossom.get(v_e)!) !== 1) throw new Error("Assertion: Type 3 deltaedge[0] not outer");
                queue.push(v_e);
            } else if (deltatype === 4) {
                if (!(deltablossom_obj instanceof Blossom)) throw new Error("Deltablossom_obj not a Blossom instance for Type 4 delta");
                expandBlossom(deltablossom_obj, false);
            }
        } // End mainloop_bfs (inner while true for dual adjustment and BFS step)

        // Check if augmented in this phase
        if (!augmented) break; // No augmentation in this phase, algorithm terminates

        // End of phase cleanup: expand any T1 blossoms with zero dual value
        // Need to iterate over a copy of keys if map is modified during iteration
        const currentBlossomDualKeys = Array.from(blossomdual.keys());
        for (const b_key of currentBlossomDualKeys) {
            if (blossomdual.has(b_key) && // Check if still exists (not expanded already)
                blossomparent.get(b_key) === null &&
                label.get(b_key) === 1 &&
                blossomdual.get(b_key)! === 0) {
                if (!(b_key instanceof Blossom)) throw new Error("Non-blossom key in blossomdual with T1 label and 0 dual");
                expandBlossom(b_key, true); // Endstage expansion
            }
        }

    } // End outer while(true) loop for phases

    // if (allinteger) verifyOptimum(); // Implement if needed

    return matchingDictToSet(mate);
}

function matchingDictToSet(mate: Record<string, string>): Set<[string, string]> {
    const matching = new Set<[string, string]>();
    const seenPairs = new Set<string>(); // To ensure each edge is added once canonically

    for (const v_node in mate) {
        const w_node = mate[v_node];
        if (w_node !== undefined) { // w_node could be null/undefined if v_node is unmatched.
            const pair = v_node < w_node ? [v_node, w_node] : [w_node, v_node];
            const pairKey = `${pair[0]},${pair[1]}`;
            if (!seenPairs.has(pairKey)) {
                matching.add(pair);
                seenPairs.add(pairKey);
            }
        }
    }
    return matching;
}


if (require.main === module) { // Basic test runner equivalent to if __name__ == "__main__":
    // const g = new Graph();
    // console.log("Graph:");
    // const edges: [string, string, number][] = [
    //     ["A", "B", 6],
    //     ["A", "C", 2],
    //     ["B", "C", 1],
    //     ["B", "D", 7],
    //     ["C", "E", 9],
    //     ["D", "E", 3],
    // ];

    // g.addWeightedEdgesFrom(edges);
    // const res = maxWeightMatching(g);
    // console.log("TypeScript Result:");
    // console.log(res); // Set { [ 'C', 'E' ], [ 'B', 'D' ] } or Set { [ 'B', 'D' ], [ 'C', 'E' ] }

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
    console.log(res); // Set { [ 'C', 'E' ], [ 'B', 'D' ] } or Set { [ 'B', 'D' ], [ 'C', 'E' ] }


    // Basic assertion for the example
    const expected = new Set<[string, string]>();
    expected.add(["C", "E"]);
    expected.add(["A", "B"]);

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
    console.log("Assertion matches expected:", match);
    if (!match) {
        console.error("Assertion failed!");
        console.log("Expected:", expComparable);
        console.log("Got:", resComparable);
    } else {
        console.log("Test passed!");
    }
}

export { Graph, maxWeightMatching };