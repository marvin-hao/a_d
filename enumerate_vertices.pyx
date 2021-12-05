#cython: language_level=3
# ----------------------- Enumerate vertices of a zonotope in dimension d -----------------------
import os
import time
import cplex
import tqdm
import numpy as np

from collections import OrderedDict
from itertools import combinations
from math import floor
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Dict
from dataclasses import dataclass

cimport numpy as cnp
from libcpp.list cimport list as cpplist


LOOKUP_TABLE = np.array([
	1, 1, 2, 6, 24, 120, 720, 5040, 40320,
	362880, 3628800, 39916800, 479001600,
	6227020800, 87178291200, 1307674368000,
	20922789888000, 355687428096000, 6402373705728000,
	121645100408832000, 2432902008176640000], dtype='int64')


def fast_factorial(n):
	if n > 20:
		raise ValueError
	return LOOKUP_TABLE[n]


def int_to_vector(g: int, d: int) -> np.ndarray:
	"""
	Convert a generator from its integer form to a Numpy vector.

	"""
	return np.fromstring(('{0:0' + str(d) + 'b}').format(g), 'i1') - 48


def path_list_to_point(path: list, d: int) -> np.ndarray:
	"""
	Convert a list of generators to a point vector.

	"""
	p = np.array([0] * d)
	for g in path:
		p = p + (np.fromstring(('{0:0' + str(d) + 'b}').format(g), 'i1') - 48)
	return p


def n_vertices(orbit: np.ndarray, d: int) -> int:
	"""
	Calculate the size of the vertex group that a canonical vertex represents.

	"""
	total = fast_factorial(d)
	_, unique_counts = np.unique(orbit, return_counts=True)
	for c in unique_counts:
		total //= fast_factorial(c)
	return total * 2


def clear_vertices(d):
	if d >= 2:
		open(f'{d}.archive', 'w').close()


class DecompResult(object):
	def __init__(self):
		self.yes = False


class LPResult(object):
	def __init__(self):
		pass


class LRUCache:
	def __init__(self, capacity: int):
		self.cache = OrderedDict()
		self.capacity = capacity

	def get(self, key):
		if key not in self.cache:
			return -1
		else:
			self.cache.move_to_end(key)
			return self.cache[key]

	def put(self, key, value) -> None:
		self.cache[key] = value
		self.cache.move_to_end(key)
		if len(self.cache) > self.capacity:
			self.cache.popitem(last=False)

	def add(self, key):
		return self.put(key, None)

	def __len__(self):
		return len(self.cache)

	def __iter__(self):
		return self.cache.__iter__()


@dataclass
class Point(object):
	"""
	Stores metadata of a point that will be used in various calculating steps.

	"""
	p = None
	orbit = None
	key = None
	path: list = None
	g = None
	last_p = None
	last_path = None
	d = None

	def __hash__(self):
		return hash(self.key)

	def __repr__(self):
		return f'{self.p}'


def decomposable(point: Point) -> DecompResult:
	"""
	To determine if a point is doubly-decomposable.

	"""
	cdef short n_ones = 0
	cdef short g = point.g
	cdef short g_copy = g
	cdef short d = point.d
	cdef short i
	cdef cpplist[short] path = sorted(point.path)
	cdef short n_type_zero_candidate = 0


	for i in range(d):
		if g_copy & 1 == 1:
			n_ones += 1
		g_copy >>= 1

	# Lemma 3.8
	for i in path:
		if i >= g:
			break
		if (~i) & (~g) == ~g:
			n_type_zero_candidate += 1

	if n_type_zero_candidate == (1 << (n_ones - 1)) - 1:
		new_decomp = local_decomposition(path, point.g, point.d)
		res = DecompResult()
		if new_decomp is not None:
			res.yes = True
		else:
			res.yes = False
		return res
	else:
		res = DecompResult()
		res.yes = True
		return res



def get_orbit(cnp.ndarray v, short d) -> np.ndarray:
	"""
	Find a vertex's canonical form.

	"""

	if np.sum(v) == d * (1 << (d - 2)):
		furthest = np.array([1 << (d - 1)] * d)
		symmetry = furthest - v
		symmetry_rank = (symmetry > (1 << (d - 2))).sum()
		v_rank = (v > (1 << (d - 2))).sum()
		if symmetry_rank < v_rank:
			v = symmetry
		elif symmetry_rank == v_rank:
			cmp = np.sort(v) - np.sort(symmetry)
			for e in cmp:
				if e == 0:
					continue
				elif e > 0:
					v = symmetry
					break
				else:
					break

	return np.sort(v)


def new_pairs(short g1, short g2, short d):
	"""
	Used for generating pair of generators to test if a path is doubly-decomposable.

	"""
	cdef short common = g1 & g2
	cdef short merged = g1 | g2
	cdef list changeable = []
	cdef short new_g1, new_g2

	cdef short i
	cdef short index
	cdef int new_bit

	for i in range(d):
		if common & 1 == 0 and merged & 1 == 1:
			changeable.append(i)
		common >>= 1
		merged >>= 1

	for i in range(1, len(changeable)):
		for indices in combinations(changeable, i):

			new_g1, new_g2 = g1, g2

			for index in indices:
				new_bit = 2 ** index
				if new_g1 & new_bit == 0:
					new_g1 += new_bit
					new_g2 -= new_bit
				else:
					new_g1 -= new_bit
					new_g2 += new_bit
			if new_g1 != new_g2:
				yield (new_g1, new_g2)
			else:
				continue

def local_decomposition(cpplist[short] path, short g, short d) -> list or None:
	"""
	Determine if a set of generator is decomposable or not through a local permutation.

	"""
	path.reverse()
	cdef cpplist[short] new_path
	cdef length = path.size()
	cdef short gg, g1, g2, ggg
	cdef size_t i

	for gg in path:
		for g1, g2 in new_pairs(gg, g, d):
			for ggg in path:
				if ggg == g1 or ggg == g2:
					break
			else:
				new_path.remove(gg)
				new_path.remove(g)
				new_path.push_back(g1)
				new_path.push_back(g2)
				return new_path

	return None


def load_vertices_from_map(d, vertices: dict) -> int:
	nv = 0
	if d >= 2:
		try:
			with open(f'{d - 1}.archive', 'r') as f:
				for l in f:
					l = l.replace('[', '')
					l = l.replace(']', '')
					l = l.strip()

					l, path = l.split(';')
					path = list(map(lambda i: int(i), path.split(",")))
					p = Point()
					p.p = np.fromstring('0 ' + l, dtype=int, sep=' ')
					p.orbit = get_orbit(p.p, d)
					p.key = p.orbit.tostring()
					p.d = d
					p.path = path
					nv += n_vertices(p.orbit, d)
					vertices[p.key] = path
		except FileNotFoundError:
			return 0
	return nv



class CplexSolver(object):
	"""
	Performs LP checks to certify vertices.

	"""
	def __init__(self, d):
		self.d = d
		self.obj = [0] * d
		self.rhs = [-1] * (2 ** d - 1)
		self.senses = ['L'] * (2 ** d - 1)
		self.lb = [-cplex.infinity] * d
		self.ub = [cplex.infinity] * d

		self.prob = cplex.Cplex()

		# disable logging
		self.prob.set_log_stream(None)
		self.prob.set_error_stream(None)
		self.prob.set_warning_stream(None)
		self.prob.set_results_stream(None)

		self.prob.objective.set_sense(self.prob.objective.sense.maximize)
		self.prob.variables.add(obj=self.obj, ub=self.ub, lb=self.lb)

		self.variable_index = list(range(d))
		self.generators = [[1 if digit=='1' else 0 for digit in ('{0:0' + str(d) + 'b}').format(g)] for g in range(1, 2 ** self.d)]
		self.negative_generators = [[-1 if digit=='1' else 0 for digit in ('{0:0' + str(d) + 'b}').format(g)] for g in range(1, 2 ** self.d)]

	def solve(self, path: list):
		rows = [[self.variable_index, self.negative_generators[g - 1]] if g in path else [self.variable_index, self.generators[g - 1]] for g in range(1, 2 ** self.d)]
		self.prob.linear_constraints.delete()
		self.prob.linear_constraints.add(lin_expr=rows, senses=self.senses, rhs=self.rhs)
		self.prob.solve()
		return self.prob.solution.is_primal_feasible()


def grow(v: np.ndarray, path: list, g: int, d: int) -> (np.ndarray, int):
	"""
	Produce new points from vertices to proceed the algorithm.

	"""
	v_ = v + (np.fromstring(('{0:0' + str(d) + 'b}').format(g), 'i1') - 48)
	return v_, path + [g]



def save_meta(d: int, total_vertices: int, total_orbit: int, time_spent: float, path_length: int):
	with open(os.path.join(f'a_{d}', f'{path_length}.meta'), 'w') as f:
		f.writelines([
			f'length: {path_length}\n',
			f'total vertices: {total_vertices}\n',
			f'total canonical vertices: {total_orbit}\n',
			f'time spent: {time_spent}\n'
		])

	print(''.join([
			f'\nlength: {path_length}\n',
			f'total vertices: {total_vertices}\n',
			f'total canonical vertices: {total_orbit}\n',
			f'time spent: {time_spent}\n'
		]))


def save_vertices_by_path_length(vertices: List[Tuple[str, Tuple[np.ndarray, str]]], d, path_length: int):
	if d >= 2:
		with open(os.path.join(f'a_{d}', f'{path_length}.archive'), 'w') as f:
			for v, point in vertices:
				# print(f'{np.fromstring(v, dtype=int)};{int_to_path(path)}', file=f)
				print(f'{np.fromstring(v, dtype=int)};{point[1]}', file=f)


def process_per_bundle(bundle: List[Tuple[np.ndarray, str]], d) -> Dict[bytes, Tuple[np.ndarray, str]]:
	"""
	Given a set of vertices, test if their children are vertices or not.

	"""
	lp_solver = CplexSolver(d)
	max_length = 2 ** (d - 1)
	max_g = 2 ** d - 1
	non_vertices = LRUCache(2 ** (d))
	vertices = {}
	for task_p, task_path in bundle:
		if task_path == '':
			task_path = []
		else:
			task_path = [int(g) for g in task_path.split(',')]
		for g in range(1, (1 << d) - 1):
			if g not in task_path and (max_g - g) not in task_path:
				point = Point()
				point.last_p, point.g, point.d, point.last_path = task_p, g, d, task_path
				point.p, point.path = grow(point.last_p, point.last_path, g, d)
				n_generator = len(point.last_path)
				if n_generator + 1 <= max_length:
					key = get_orbit(point.p, d).tostring()

					if key in non_vertices:
						continue
					elif key in vertices:
						continue

					res = decomposable(point)
					if res.yes:
						non_vertices.add(key)
						continue

					if lp_solver.solve(point.path):
						vertices[key] = (point.p, ','.join(map(str, point.path)))
					else:
						non_vertices.add(key)
	return vertices


def process_per_layer(prev_layer: List[Tuple[np.ndarray, str]], d: int, concurrency: int) -> List[Tuple[np.ndarray, str]]:
	"""
	Given a set of vertices with the same length (k) of path, find all the vertices in the next layer in parallel.

	"""
	start = time.time()

	if prev_layer[0][1] == '':
		path_length = 1
	else:
		path_length = len(prev_layer[0][1].split(',')) + 1

	# n_bundle = concurrency * path_length
	n_bundle = path_length * 2
	bundle_size = floor(len(prev_layer) / n_bundle)
	bundle_group = []
	for i in range(n_bundle):
		if i != n_bundle - 1:
			bundle_group.append(prev_layer[i * bundle_size: i*bundle_size + bundle_size])
		else:
			bundle_group.append(prev_layer[i * bundle_size:])
	del prev_layer

	process_pool = Pool(processes=concurrency, maxtasksperchild=concurrency//2)
	pbar = tqdm.tqdm(total=n_bundle)

	vertices = {}
	total_vertices = 0

	def handle_result(local_vertices):
		# L_{k+1}(G) ← L_{k+1}(G) ∪ {S ∪ {g^j}}
		vertices.update(local_vertices)
		pbar.update(1)
		del local_vertices

	for bundle in bundle_group:
		process_pool.apply_async(process_per_bundle, (bundle, d), callback=handle_result)

	process_pool.close()
	process_pool.join()
	pbar.close()

	sorted_vertices_tuple_list = sorted(vertices.items(), key=lambda x: x[0])

	for k, vertex in sorted_vertices_tuple_list:
		total_vertices += n_vertices(vertex[0], d)

	time_spent = time.time() - start
	save_vertices_by_path_length(sorted_vertices_tuple_list, d, path_length)
	save_meta(d, total_vertices, len(sorted_vertices_tuple_list), time_spent, path_length)
	return list(map(lambda x: x[1], sorted_vertices_tuple_list))


def load_vertices_bundle(d, k, start, end) -> List[Point]:
	canonical_vertices = []
	with open(os.path.join(f'a_{d}/{k}.archive'), 'r') as f:
		for _ in range(start):
			next(f)
		for _ in range(start, end):
			l = f.readline()
			l = l.replace('[', '')
			l = l.replace(']', '')
			l = l.strip()

			l = l.split(';')
			p = list(map(lambda i: int(i), l[1].split(",")))

			v = l[0]
			v = np.fromstring(v, dtype=int, sep=' ')
			point = Point()
			point.d = d
			point.path = p
			point.orbit = v
			point.p = path_list_to_point(p, d)
			canonical_vertices.append(point)
		return canonical_vertices

def restart(d, concurrency) -> List[Point]:
	"""
	A utility to restart the algorithm from checkpoints.

	"""
	if not os.path.exists(f'a_{d}'):
		return []


	meta_files = [int(f.split('.')[0]) for f in os.listdir(f'a_{d}') if f.endswith('.meta')]
	meta_files.sort(reverse=True)
	for k in meta_files:
		archive = open(os.path.join(f'a_{d}/{k}.archive'), 'r')
		n_lines = len(archive.readlines())
		archive.close()
		if n_lines == 0:
			continue

		meta = open(os.path.join(f'a_{d}/{k}.meta'), 'r')
		for line in meta.readlines():
			if line.startswith('total canonical vertices: '):
				expected_n_lines = int(line.split('total canonical vertices: ')[1])
				if expected_n_lines == n_lines:
					total_lines = len(open(os.path.join(f'a_{d}/{k}.archive'), 'r').readlines())
					print(f'loading ... {total_lines} ')
					n_bundle = concurrency * 3
					bundle_size = floor(total_lines / n_bundle)
					bundle_group = []
					for i in range(n_bundle):
						if i != n_bundle - 1:
							bundle_group.append([i * bundle_size, i * bundle_size + bundle_size])
						else:
							bundle_group.append([i * bundle_size, total_lines])

					process_pool = Pool(processes=concurrency)
					pbar = tqdm.tqdm(total=n_bundle)

					vertices = []

					def handle_result(local_vertices):
						vertices.extend(local_vertices)
						pbar.update(1)
						del local_vertices
					for start, end in bundle_group:
						process_pool.apply_async(load_vertices_bundle, (d, k, start, end), callback=handle_result)
					process_pool.close()
					process_pool.join()
					pbar.close()

					print(f'\nloading done ... {len(vertices)}')

					return vertices

				else:
					meta.close()
	return []


def run(d: int, n_worker: int, checkpoint: List[Point]):
	origin = Point()
	origin.p = np.array([0] * d)
	origin.path = []
	clear_vertices(d)
	if not os.path.exists(f'a_{d}'):
		os.makedirs(f'a_{d}')

	if n_worker is None:
		n_processor = cpu_count()
	else:
		n_processor = n_worker

	# Algorithm 3
	try:
		if len(checkpoint) > 0:
			layer = [(e.p, ','.join(map(str, e.path))) for e in checkpoint]
		else:
			layer = [(origin.p, ','.join(origin.path))]
		while True:
			layer = process_per_layer(layer, d, n_processor)
			if len(layer) == 0 or layer[0][0][0] == (1 << (d - 1)):
				break

	except KeyboardInterrupt:
		pass
