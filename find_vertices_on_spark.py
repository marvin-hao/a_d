import math
import os
import time
from find_vertices import find_vertices, n_vertices
from pyspark import SparkFiles

import tqdm
from typing import List, Tuple

from pyspark.sql import SparkSession


def restart(root) -> Tuple[List[str], int]:
	length = 0
	meta_files = [int(f.split('.')[0]) for f in os.listdir(root) if f.endswith('.meta')]
	meta_files.sort(reverse=True)
	for k in meta_files:
		archive = open(os.path.join(root, f'{k}.archive'), 'r')
		n_lines = len(archive.readlines())
		archive.close()
		if n_lines == 0:
			continue

		meta = open(os.path.join(root, f'{k}.meta'), 'r')
		length = k
		for line in meta.readlines():
			if line.startswith('total canonical vertices: '):
				expected_n_lines = int(line.split('total canonical vertices: ')[1])
				if expected_n_lines == n_lines:
					total_lines = len(open(os.path.join(root, f'{k}.archive'), 'r').readlines())
					print(f'loading ... {total_lines} ')

					pbar = tqdm.tqdm(total=n_lines)
					paths = []

					with open(os.path.join(SparkFiles.getRootDirectory(), f'{k}.archive'), 'r') as f:
						for l in f.readlines():
							paths.append(l.split(';')[1].replace(' ', '').replace('[', '').replace(']', ''))
							pbar.update(1)

					print(f'\nloading done ... {len(paths)}')

					return paths, length

				else:
					meta.close()
	return [''], length


def choose_vertex(p1: list, p2: list):
	for g1, g2 in zip(p1, p2):
		if g1 == g2:
			continue
		elif g1 < g2:
			return p1
		else:
			return p2
	else:
		return p1


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


def save_vertices_by_path_length(vertices: List[Tuple[str, str, int]], d, path_length: int):
	if d >= 2:
		with open(os.path.join(f'a_{d}', f'{path_length}.archive'), 'w') as f:
			for v, path, _ in vertices:
				print(f'{v};{path}', file=f)


def main():
	dim = 7
	start = time.time()
	spark = SparkSession.builder.config(
		"spark.driver.memory", "4g"
	).config(
		'spark.ui.showConsoleProgress', 'true'
	).appName(
		"counting_vertices"
	).getOrCreate()

	loaded, length = restart(root=SparkFiles.getRootDirectory())
	print(loaded)
	bundle_size = (2 ** dim) * dim
	# bundle_size = 128
	concurrency = math.ceil(len(loaded) / bundle_size)
	vertices = spark.sparkContext.parallelize(loaded, concurrency)
	task = vertices.mapPartitions(
		lambda p_list: find_vertices(p_list, dim)
	).reduceByKey(
		lambda x, y: choose_vertex(x, y)
	).sortByKey(ascending=True).map(
		lambda x: (x[0], str(x[1]), n_vertices(x[0], dim))
	)

	result = task.collect()

	time_spent = time.time() - start
	total_vertices = 0
	for r in result:
		total_vertices += r[2]
	total_canonical_vertices = len(result)
	path_length = length + 1
	save_meta(dim, total_vertices, total_canonical_vertices, time_spent, path_length)
	save_vertices_by_path_length(result, dim, path_length)


if __name__ == '__main__':
	main()
