// Transcrypt'ed from Python, 2025-05-22 21:18:20
import { AssertionError, AttributeError, BaseException, DeprecationWarning, Exception, IndexError, IterableError, KeyError, NotImplementedError, RuntimeWarning, StopIteration, UserWarning, ValueError, Warning, __JsIterator__, __PyIterator__, __Terminal__, __add__, __and__, __call__, __class__, __envir__, __eq__, __floordiv__, __ge__, __get__, __getcm__, __getitem__, __getslice__, __getsm__, __gt__, __i__, __iadd__, __iand__, __idiv__, __ijsmod__, __ilshift__, __imatmul__, __imod__, __imul__, __in__, __init__, __ior__, __ipow__, __irshift__, __isub__, __ixor__, __jsUsePyNext__, __jsmod__, __k__, __kwargtrans__, __le__, __lshift__, __lt__, __matmul__, __mergefields__, __mergekwargtrans__, __mod__, __mul__, __ne__, __neg__, __nest__, __or__, __pow__, __pragma__, __pyUseJsNext__, __rshift__, __setitem__, __setproperty__, __setslice__, __sort__, __specialattrib__, __sub__, __super__, __t__, __terminal__, __truediv__, __withblock__, __xor__, _sort, abs, all, any, assert, bin, bool, bytearray, bytes, callable, chr, delattr, dict, dir, divmod, enumerate, filter, float, getattr, hasattr, hex, input, int, isinstance, issubclass, len, list, map, max, min, object, oct, ord, pow, print, property, py_TypeError, py_iter, py_metatype, py_next, py_reversed, py_typeof, range, repr, round, set, setattr, sorted, str, sum, tuple, zip } from './org.transcrypt.__runtime__.js';
import { repeat } from './itertools.js';
var __name__ = '__main__';
export var matching_dict_to_set = function (matching) {
	var edges = set();
	for (var edge of matching.py_items()) {
		var __left0__ = edge;
		var u = __left0__[0];
		var v = __left0__[1];
		if (__in__(tuple([v, u]), edges) || __in__(edge, edges)) {
			continue;
		}
		edges.add(edge);
	}
	return edges;
};
export var max_weight_matching = function (G, maxcardinality, weight) {
	if (typeof maxcardinality == 'undefined' || (maxcardinality != null && maxcardinality.hasOwnProperty("__kwargtrans__"))) {
		;
		var maxcardinality = false;
	};
	if (typeof weight == 'undefined' || (weight != null && weight.hasOwnProperty("__kwargtrans__"))) {
		;
		var weight = 'weight';
	};
	var NoNode = __class__('NoNode', [object], {
		__module__: __name__,
	});
	var Blossom = __class__('Blossom', [object], {
		__module__: __name__,
		__slots__: ['childs', 'edges', 'mybestedges'],
		get leaves() {
			return __get__(this, function* (self) {
				var stack = [self.childs];
				while (stack) {
					var t = stack.py_pop();
					if (isinstance(t, Blossom)) {
						stack.extend(t.childs);
					}
					else {
						yield t;
					}
				}
			});
		}
	});
	var gnodes = list(G);
	if (!(gnodes)) {
		return set();
	}
	var maxweight = 0;
	var allinteger = true;
	for (var [i, j, d] of G.edges(__kwargtrans__({ data: true }))) {
		var wt = d['weight'];
		if (i != j && wt > maxweight) {
			var maxweight = wt;
		}
		var allinteger = allinteger && __in__(str(py_typeof(wt)).py_split("'")[1], tuple(['int', 'long']));
	}

	console.log("Setting data structures");
	var mate = dict({});
	var label = dict({});
	var labeledge = dict({});
	var inblossom = dict(zip(gnodes, gnodes));
	// no more repeat since it is eagerly evaled
	var blossomparent = {};
	for (let node of gnodes) {
		blossomparent[node] = null;
	}
	var blossombase = dict(zip(gnodes, gnodes));
	var bestedge = dict({});
	// no more repeat since it is eagerly evaled
	var dualvar = {}
	for (let node of gnodes) {
		dualvar[node] = maxweight;
	}
	var blossomdual = dict({});
	var allowedge = dict({});
	var queue = [];
	console.log("Data structures setup")
	// var slack = function (v, w) {
	// 	return (dualvar[v] + dualvar[w]) - 2 * G[v][w];
	// };
	var slack = function (v, w) {

		var dualvar_v = dualvar[v];
		var dualvar_w = dualvar[w];
		var weight = 0;
		// if (G[v] && G[v][w] !== undefined) {
		// 	weight = 2 * G[v][w]
			
		// }
		weight = 2 * G.indexer(v, w);
		

		console.log("slack", dualvar_v, dualvar_w, weight)

		return dualvar_v + dualvar_w - weight;
	};


	var assignLabel = function (w, t, v) {
		var b = inblossom[w];
		var __left0__ = t;
		label[w] = __left0__;
		label[b] = __left0__;
		if (v !== null) {
			var __left0__ = tuple([v, w]);
			labeledge[w] = __left0__;
			labeledge[b] = __left0__;
		}
		else {
			var __left0__ = null;
			labeledge[w] = __left0__;
			labeledge[b] = __left0__;
		}
		var __left0__ = null;
		bestedge[w] = __left0__;
		bestedge[b] = __left0__;
		if (t == 1) {
			if (isinstance(b, Blossom)) {
				queue.extend(b.leaves());
			}
			else {
				queue.append(b);
			}
		}
		else if (t == 2) {
			var base = blossombase[b];
			assignLabel(mate[base], 1, base);
		}
	};
	var scanBlossom = function (v, w) {
		var path = [];
		var base = NoNode;
		while (v !== NoNode) {
			var b = inblossom[v];
			if (label[b] & 4) {
				var base = blossombase[b];
				break;
			}
			path.append(b);
			label[b] = 5;
			if (labeledge[b] === null) {
				var v = NoNode;
			}
			else {
				var v = labeledge[b][0];
				var b = inblossom[v];
				var v = labeledge[b][0];
			}
			if (w !== NoNode) {
				var __left0__ = tuple([w, v]);
				var v = __left0__[0];
				var w = __left0__[1];
			}
		}
		for (var b of path) {
			label[b] = 1;
		}
		return base;
	};
	var addBlossom = function (base, v, w) {
		var bb = inblossom[base];
		var bv = inblossom[v];
		var bw = inblossom[w];
		var b = Blossom();
		blossombase[b] = base;
		blossomparent[b] = null;
		blossomparent[bb] = b;
		var __left0__ = [];
		b.childs = __left0__;
		var path = __left0__;
		var __left0__ = [tuple([v, w])];
		b.edges = __left0__;
		var edgs = __left0__;
		while (bv != bb) {
			blossomparent[bv] = b;
			path.append(bv);
			edgs.append(labeledge[bv]);
			var v = labeledge[bv][0];
			var bv = inblossom[v];
		}
		path.append(bb);
		path.reverse();
		edgs.reverse();
		while (bw != bb) {
			blossomparent[bw] = b;
			path.append(bw);
			edgs.append(tuple([labeledge[bw][1], labeledge[bw][0]]));
			var w = labeledge[bw][0];
			var bw = inblossom[w];
		}
		label[b] = 1;
		labeledge[b] = labeledge[bb];
		blossomdual[b] = 0;
		for (var v of b.leaves()) {
			if (label[inblossom[v]] == 2) {
				queue.append(v);
			}
			inblossom[v] = b;
		}
		var bestedgeto = dict({});
		for (var bv of path) {
			if (isinstance(bv, Blossom)) {
				if (bv.mybestedges !== null) {
					var nblist = bv.mybestedges;
					bv.mybestedges = null;
				}
				else {
					var nblist = (function () {
						var __accu0__ = [];
						for (var v of bv.leaves()) {
							for (var w of G.neighbors(v)) {
								if (v != w) {
									__accu0__.append(tuple([v, w]));
								}
							}
						}
						return __accu0__;
					})();
				}
			}
			else {
				var nblist = (function () {
					var __accu0__ = [];
					for (var w of G.neighbors(bv)) {
						if (bv != w) {
							__accu0__.append(tuple([bv, w]));
						}
					}
					return __accu0__;
				})();
			}
			for (var k of nblist) {
				var __left0__ = k;
				var i = __left0__[0];
				var j = __left0__[1];
				if (inblossom[j] == b) {
					var __left0__ = tuple([j, i]);
					var i = __left0__[0];
					var j = __left0__[1];
				}
				var bj = inblossom[j];
				if (bj != b && label.py_get(bj) == 1 && (!__in__(bj, bestedgeto) || slack(i, j) < slack(...bestedgeto[bj]))) {
					bestedgeto[bj] = k;
				}
			}
			bestedge[bv] = null;
		}
		b.mybestedges = list(bestedgeto.py_values());
		var mybestedge = null;
		bestedge[b] = null;
		for (var k of b.mybestedges) {
			var kslack = slack(...k);
			if (mybestedge === null || kslack < mybestslack) {
				var mybestedge = k;
				var mybestslack = kslack;
			}
		}
		bestedge[b] = mybestedge;
	};
	var expandBlossom = function (b, endstage) {
		var _recurse = function* (b, endstage) {
			for (var s of b.childs) {
				blossomparent[s] = null;
				if (isinstance(s, Blossom)) {
					if (endstage && blossomdual[s] == 0) {
						yield s;
					}
					else {
						for (var v of s.leaves()) {
							inblossom[v] = s;
						}
					}
				}
				else {
					inblossom[s] = s;
				}
			}
			if (!(endstage) && label.py_get(b) == 2) {
				var entrychild = inblossom[labeledge[b][1]];
				var j = b.childs.index(entrychild);
				if (j & 1) {
					j -= len(b.childs);
					var jstep = 1;
				}
				else {
					var jstep = -(1);
				}
				var __left0__ = labeledge[b];
				var v = __left0__[0];
				var w = __left0__[1];
				while (j != 0) {
					if (jstep == 1) {
						var __left0__ = b.edges[j];
						var p = __left0__[0];
						var q = __left0__[1];
					}
					else {
						var __left0__ = b.edges[j - 1];
						var q = __left0__[0];
						var p = __left0__[1];
					}
					label[w] = null;
					label[q] = null;
					assignLabel(w, 2, v);
					var __left0__ = true;
					allowedge.__setitem__([p, q], __left0__);
					allowedge.__setitem__([q, p], __left0__);
					j += jstep;
					if (jstep == 1) {
						var __left0__ = b.edges[j];
						var v = __left0__[0];
						var w = __left0__[1];
					}
					else {
						var __left0__ = b.edges[j - 1];
						var w = __left0__[0];
						var v = __left0__[1];
					}
					var __left0__ = true;
					allowedge.__setitem__([v, w], __left0__);
					allowedge.__setitem__([w, v], __left0__);
					j += jstep;
				}
				var bw = b.childs[j];
				var __left0__ = 2;
				label[w] = __left0__;
				label[bw] = __left0__;
				var __left0__ = tuple([v, w]);
				labeledge[w] = __left0__;
				labeledge[bw] = __left0__;
				bestedge[bw] = null;
				j += jstep;
				while (b.childs[j] != entrychild) {
					var bv = b.childs[j];
					if (label.py_get(bv) == 1) {
						j += jstep;
						continue;
					}
					if (isinstance(bv, Blossom)) {
						for (var v of bv.leaves()) {
							if (label.py_get(v)) {
								break;
							}
						}
					}
					else {
						var v = bv;
					}
					if (label.py_get(v)) {
						label[v] = null;
						label[mate[blossombase[bv]]] = null;
						assignLabel(v, 2, labeledge[v][0]);
					}
					j += jstep;
				}
			}
			label.py_pop(b, null);
			labeledge.py_pop(b, null);
			bestedge.py_pop(b, null);
			delete blossomparent[b];
			delete blossombase[b];
			delete blossomdual[b];
		};
		var stack = [_recurse(b, endstage)];
		while (stack) {
			var top = stack[-(1)];
			var __break1__ = false;
			for (var s of top) {
				stack.append(_recurse(s, endstage));
				__break1__ = true;
				break;
			}
			if (!__break1__) {
				stack.py_pop();
			}
		}
	};
	var augmentBlossom = function (b, v) {
		var _recurse = function* (b, v) {
			var t = v;
			while (blossomparent[t] != b) {
				var t = blossomparent[t];
			}
			if (isinstance(t, Blossom)) {
				yield tuple([t, v]);
			}
			var __left0__ = b.childs.index(t);
			var i = __left0__;
			var j = __left0__;
			if (i & 1) {
				j -= len(b.childs);
				var jstep = 1;
			}
			else {
				var jstep = -(1);
			}
			while (j != 0) {
				j += jstep;
				var t = b.childs[j];
				if (jstep == 1) {
					var __left0__ = b.edges[j];
					var w = __left0__[0];
					var x = __left0__[1];
				}
				else {
					var __left0__ = b.edges[j - 1];
					var x = __left0__[0];
					var w = __left0__[1];
				}
				if (isinstance(t, Blossom)) {
					yield tuple([t, w]);
				}
				j += jstep;
				var t = b.childs[j];
				if (isinstance(t, Blossom)) {
					yield tuple([t, x]);
				}
				mate[w] = x;
				mate[x] = w;
			}
			b.childs = b.childs.__getslice__(i, null, 1) + b.childs.__getslice__(0, i, 1);
			b.edges = b.edges.__getslice__(i, null, 1) + b.edges.__getslice__(0, i, 1);
			blossombase[b] = blossombase[b.childs[0]];
		};
		var stack = [_recurse(b, v)];
		while (stack) {
			var top = stack[-(1)];
			var __break1__ = false;
			for (var args of top) {
				stack.append(_recurse(...args));
				__break1__ = true;
				break;
			}
			if (!__break1__) {
				stack.py_pop();
			}
		}
	};
	var augmentMatching = function (v, w) {
		for (var [s, j] of tuple([tuple([v, w]), tuple([w, v])])) {
			while (1) {
				var bs = inblossom[s];
				if (isinstance(bs, Blossom)) {
					augmentBlossom(bs, s);
				}
				mate[s] = j;
				if (labeledge[bs] === null) {
					break;
				}
				var t = labeledge[bs][0];
				var bt = inblossom[t];
				var __left0__ = labeledge[bt];
				var s = __left0__[0];
				var j = __left0__[1];
				if (isinstance(bt, Blossom)) {
					augmentBlossom(bt, j);
				}
				mate[j] = s;
			}
		}
	};
	var verifyOptimum = function () {
		if (maxcardinality) {
			var vdualoffset = max(0, -(min(dualvar.py_values())));
		}
		else {
			var vdualoffset = 0;
		}
		for (var [i, j, d] of G.edges({ data: true })) {
			var wt = d['weight'];
			if (i == j) {
				continue;
			}
			var s = (dualvar[i] + dualvar[j]) - 2 * wt;
			var iblossoms = [i];
			var jblossoms = [j];
			while (blossomparent[iblossoms[-(1)]] !== null) {
				iblossoms.append(blossomparent[iblossoms[-(1)]]);
			}
			while (blossomparent[jblossoms[-(1)]] !== null) {
				jblossoms.append(blossomparent[jblossoms[-(1)]]);
			}
			iblossoms.reverse();
			jblossoms.reverse();
			for (var [bi, bj] of zip(iblossoms, jblossoms)) {
				if (bi != bj) {
					break;
				}
				s += 2 * blossomdual[bi];
			}
			if (mate.py_get(i) == j || mate.py_get(j) == i) {
			}
		}
		for (var v of gnodes) {
		}
		for (var b in blossomdual) {
			if (blossomdual[b] > 0) {
				for (var [i, j] of b.edges.__getslice__(1, null, 2)) {
				}
			}
		}
	};
	// if (__in__(tuple([v, w]), allowedge)) { is bad
	var edge_in = function (u, v, allowedge) {
		return (`${u},${v}` in allowedge);
	};
	console.log("Start!")
	// while (1) {
	for (var i = 1; i < 10000 + 1; i++) {
		console.log("Outer")
		label.py_clear();
		labeledge.py_clear();
		bestedge.py_clear();
		for (var b of Object.keys(blossomdual)) {
			b.mybestedges = null;
		}
		// allowedge.py_clear();
		allowedge = {};
		// queue.__setslice__(0, null, null, []);
		var queue = [];

		for (var v of gnodes) {
			if (!__in__(v, mate) && label.py_get(inblossom[v]) === null) {
				assignLabel(v, 1, null);
			}
		}
		var augmented = 0;
		while (1) {
			console.log("Inner 1", queue.length)
			while (queue.length > 0 && !(augmented)) {
				console.log("inner queue")
				var v = queue.py_pop();
				console.log(v)
				for (var w of G.neighbors(v)) {
					if (w == v) {
						continue;
					}
					var bv = inblossom[v];
					var bw = inblossom[w];
					if (bv == bw) {
						continue;
					}
					// if (!__in__(tuple([v, w]), allowedge)) {
					if (!edge_in(v, w, allowedge)) {
						console.log("Gonna populate")
						var kslack = slack(v, w);
						console.log(kslack);

						if (kslack <= 0) {
							// var __left0__ = true;
							// allowedge.__setitem__([v, w], __left0__);
							// allowedge.__setitem__([w, v], __left0__);
							allowedge[`${v},${w}`] = true;
							allowedge[`${w},${v}`] = true;
						}
					}
					console.log("negh ",);
					console.log(allowedge)
					// if (__in__(tuple([v, w]), allowedge)) {
					if (edge_in(v, w, allowedge) || edge_in(w, v, allowedge)) {
						console.log("c2")
						if (label.py_get(bw) === null) {
							assignLabel(w, 2, v);
						}

						else if (label.py_get(bw) == 1) {
							print("Scan")
							var base = scanBlossom(v, w);
							if (base !== NoNode) {
								addBlossom(base, v, w);
							}
							else {
								augmentMatching(v, w);
								var augmented = 1;
								break;
							}
						}
						else if (label.py_get(w) === null) {
							label[w] = 2;
							labeledge[w] = tuple([v, w]);
						}
					}
					else if (label.py_get(bw) == 1) {
						if (bestedge.py_get(bv) === null || kslack < slack(...bestedge[bv])) {
							bestedge[bv] = tuple([v, w]);
						}
					}
					else if (label.py_get(w) === null) {
						if (bestedge.py_get(w) === null || kslack < slack(...bestedge[w])) {
							bestedge[w] = tuple([v, w]);
						}
					}
				}
			}
			if (augmented) {
				console.log("Break!")
				break;
			}
			var deltatype = -(1);
			var __left0__ = null;
			var delta = __left0__;
			var deltaedge = __left0__;
			var deltablossom = __left0__;
			// if (!(maxcardinality)) {
			// 	var deltatype = 1;
			// 	var delta = min(dualvar.py_values());
			// }
			if (!(maxcardinality)) {
				var deltatype = 1;
				var delta = Math.min(...Object.values(dualvar)); // Use Object.values and Math.min
			}

			for (var v of G.nodes()) {
				if (label.py_get(inblossom[v]) === null && bestedge.py_get(v) !== null) {
					var d = slack(...bestedge[v]);
					if (deltatype == -(1) || d < delta) {
						var delta = d;
						var deltatype = 2;
						var deltaedge = bestedge[v];
					}
				}
			}
			for (var b in blossomparent) {
				if (blossomparent[b] === null && label.py_get(b) == 1 && bestedge.py_get(b) !== null) {
					var kslack = slack(...bestedge[b]);
					if (allinteger) {
						var d = Math.floor(kslack / 2);
					}
					else {
						var d = kslack / 2.0;
					}
					if (deltatype == -(1) || d < delta) {
						var delta = d;
						var deltatype = 3;
						var deltaedge = bestedge[b];
					}
				}
			}
			for (var b in blossomdual) {
				if (blossomparent[b] === null && label.py_get(b) == 2 && (deltatype == -(1) || blossomdual[b] < delta)) {
					var delta = blossomdual[b];
					var deltatype = 4;
					var deltablossom = b;
				}
			}
			if (deltatype == -(1)) {
				var deltatype = 1;
				var delta = max(0, min(dualvar.py_values()));
			}
			for (var v of gnodes) {
				if (label.py_get(inblossom[v]) == 1) {
					dualvar[v] -= delta;
				}
				else if (label.py_get(inblossom[v]) == 2) {
					dualvar[v] += delta;
				}
			}
			for (var b in blossomdual) {
				if (blossomparent[b] === null) {
					if (label.py_get(b) == 1) {
						blossomdual[b] += delta;
					}
					else if (label.py_get(b) == 2) {
						blossomdual[b] -= delta;
					}
				}
			}
			if (deltatype == 1) {
				break;
			}
			else if (deltatype == 2) {
				var __left0__ = deltaedge;
				var v = __left0__[0];
				var w = __left0__[1];
				var __left0__ = true;
				allowedge.__setitem__([v, w], __left0__);
				allowedge.__setitem__([w, v], __left0__);
				queue.append(v);
			}
			else if (deltatype == 3) {
				var __left0__ = deltaedge;
				var v = __left0__[0];
				var w = __left0__[1];
				var __left0__ = true;
				allowedge.__setitem__([v, w], __left0__);
				allowedge.__setitem__([w, v], __left0__);
				queue.append(v);
			}
			else if (deltatype == 4) {
				expandBlossom(deltablossom, false);
			}
		}
		for (var v in mate) {
		}
		if (!(augmented)) {
			break;
		}
		for (var b of list(blossomdual.py_keys())) {
			if (!__in__(b, blossomdual)) {
				continue;
			}
			if (blossomparent[b] === null && label.py_get(b) == 1 && blossomdual[b] == 0) {
				expandBlossom(b, true);
			}
		}
	}
	if (allinteger) {
		verifyOptimum();
	}
	return matching_dict_to_set(mate);
};
export var Graph = __class__('Graph', [object], {
	__module__: __name__,
	get __init__() {
		return __get__(this, function (self) {
			self.adj = dict({});
			self.node_list = [];
		});
	},
	get add_edge() {
		return __get__(this, function (self, u, v, weight) {
			if (typeof weight == 'undefined' || (weight != null && weight.hasOwnProperty("__kwargtrans__"))) {
				var weight = 1;
			};
			if (!__in__(u, self.adj)) {
				self.adj[u] = dict({});
				self.node_list.append(str(u));
			}
			if (!__in__(v, self.adj)) {
				self.adj[v] = dict({});
				self.node_list.append(str(v));
			}
			self.adj[u][v] = weight;
			self.adj[v][u] = weight;
		});
	},
	get add_weighted_edges_from() {
		return __get__(this, function (self, edge_list) {
			for (var [u, v, weight] of edge_list) {
				self.add_edge(u, v, weight);
			}
		});
	},
	get __iter__() {
		return __get__(this, function (self) {
			return py_iter(self.node_list);
		});
	},
	[Symbol.iterator]() { return this.__iter__() },
	get edges() {
		return __get__(this, function* (self, data) {
			if (typeof data == 'undefined' || (data != null && data.hasOwnProperty("__kwargtrans__"))) {
				var data = false;
			};
			var seen = set();
			for (var u of Object.keys(self.adj)) { // Iterate over keys
				for (var v of Object.keys(self.adj[u])) { //Also iterate over the keys here
					if (!__in__(tuple([u, v]), seen) && !__in__(tuple([v, u]), seen)) {
						seen.add(tuple([u, v]));
						if (data) {
							yield tuple([u, v, dict({ 'weight': self.adj[u][v] })]);
						}
						else {
							yield tuple([u, v, dict({ 'weight': 1 })]);
						}
					}
				}
			}
		});
	},
	get neighbors() {
		return __get__(this, function (self, node) {
			if (self.adj[node]) { // Check if the node exists in self.adj
				return list(Object.keys(self.adj[node])); //Use object keys instead of py_keys
			} else {
				return []; // Return an empty list if the node doesn't exist
			}
		});
	},
	get __getitem__() {
		return __get__(this, function (self, node) {
			return self.adj[node];
		});
	},
	get nodes() {
		return __get__(this, function (self) {
			return py_iter(self.node_list);
		});
	},
	get indexer() {
		return __get__(this, function(self, u, v) {
			if (v == null){
				return self.adj[u]
			} else {
				return self.adj[u][v];
			}

		});

	}
});
var matching_dict_to_set = function (mate) {
	var matching = set();
	for (var [v, w] of mate.py_items()) {
		if (w !== null && v < w) {
			matching.add(tuple([v, w]));
		}
	}
	return matching;
};
if (__name__ == '__main__') {
	console.log("Starting")
	var g = Graph();
	var edges = [tuple([1, 2, 6]), tuple([1, 3, 2]), tuple([2, 3, 1]), tuple([2, 4, 7]), tuple([3, 5, 9]), tuple([4, 5, 3])];


	g.add_weighted_edges_from(edges);
	// console.log(g)
	// console.log(g.nodes())
	// console.log(g.__getitem__["1"])
	// console.log(g.node_list)
	// console.log(g.adj.py_keys())
	// console.log(g.adj["1"])
	// console.log(g.indexer("1"))
	// console.log(g.indexer("2", "3"))

	var res = max_weight_matching(g);
	print (res);
}

//# sourceMappingURL=max-weight.map