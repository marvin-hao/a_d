#cython: language_level=3
# distutils : language = c++
# ----------------------- Utility script for a distributed calculation on the Spark cluster -----------------------
from typing import Iterable

cimport numpy as cnp
import cplex
import numpy as np
import re

from itertools import combinations
from collections import OrderedDict
from libcpp.list cimport list as cpplist

LOOKUP_TABLE = np.array([
	1, 1, 2, 6, 24, 120, 720, 5040, 40320,
	362880, 3628800, 39916800, 479001600,
	6227020800, 87178291200, 1307674368000,
	20922789888000, 355687428096000, 6402373705728000,
	121645100408832000, 2432902008176640000], dtype='int64')

generator_lookup = [
	{g: np.fromstring(('{0:0' + str(d) + 'b}').format(g), 'i1') - 48 for g in range(1, 2**d)} for d in range(0, 10)
]


def fast_factorial(n):
	if n > 20:
		raise ValueError
	return LOOKUP_TABLE[n]


class LRUCache:

	# initialising capacity
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

def decomposable(path_in, short g, short d) -> bool:
	# if not will_shrink(v_, v, d):
	# 	print('not shrink')

	cdef short n_ones = 0
	cdef short g_copy = g
	cdef short i
	cdef cpplist[short] path = sorted(path_in)
	cdef short n_type_zero_candidate = 0


	for i in range(d):
		if g_copy & 1 == 1:
			n_ones += 1
		g_copy >>= 1

	# print('******')
	for i in path:
		if i >= g:
			break
		if (~i) & (~g) == ~g:
			n_type_zero_candidate += 1

	if n_type_zero_candidate == (1 << (n_ones - 1)) - 1:
		new_decomp = local_decomposition(path, g, d)
		if new_decomp is not None:
			return True
		else:
			return False
	else:
		return True


def get_orbit(cnp.ndarray v, short d) -> np.ndarray:

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
				yield new_g1, new_g2
			else:
				continue


def local_decomposition(cpplist[short] path, short g, short d) -> list or None:
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


class CplexSolver(object):
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
	v_ = v + (np.fromstring(('{0:0' + str(d) + 'b}').format(g), 'i1') - 48)
	return v_, path + [g]


def path_list_to_point(path: list, d: int) -> np.ndarray:
	point = np.array([0] * d)
	g_list = [generator_lookup[d][g] for g in path]

	for g in path:
		point = point + generator_lookup[d][g]
	return point


def find_vertices(bundle: Iterable[str], d):
	lp_solver = CplexSolver(d)
	max_length = 2 ** (d - 1)
	max_g = 2 ** d - 1
	non_vertices = LRUCache(2 ** (d))
	vertices = {}
	for task_path in bundle:
		if task_path == '':
			task_path = []
		else:
			task_path = list(map(int, task_path.split(',')))
		task_p = path_list_to_point(task_path, d)
		for g in range(1, (1 << d) - 1):
			if g not in task_path and (max_g - g) not in task_path:
				last_p, last_path = task_p, task_path
				p, path = grow(last_p, last_path, g, d)
				n_generator = len(last_path)
				if n_generator + 1 <= max_length:
					orbit = get_orbit(p, d)
					key = np.array2string(orbit)
					if key in non_vertices:
						continue
					elif key in vertices:
						continue

					yes = decomposable(path, g, d)
					if yes:
						non_vertices.add(key)
						continue

					if lp_solver.solve(path):
						vertices[key] = path
					else:
						non_vertices.add(key)
	return list(vertices.items())


def n_vertices(orbit: str, d: int) -> int:
	total = fast_factorial(d)
	orbit_arr = np.fromstring(
		re.sub(' +', ' ', orbit[1: -1].strip()),
		sep=' ', dtype=int)
	_, unique_counts = np.unique(orbit_arr, return_counts=True)
	for c in unique_counts:
		total //= fast_factorial(c)
	return np.asscalar(total * 2)