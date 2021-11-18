import multiprocessing
import sys
import os
import time
import tqdm
import numpy as np

from typing import List, Tuple
from functools import partial
from pyspark import SparkFiles
from pyspark.sql import SparkSession

from find_vertices import n_vertices, get_orbit, path_list_to_point



def load_layer(root, k):
	print(f'loading layer {k}...')
	paths = []
	with open(os.path.join(root, f'{k}.archive'), 'r') as layer:
		with open(os.path.join(root, f'{k}.meta'), 'r') as meta:
			for l in meta.readlines():
				if 'total canonical vertices' in l:
					n_lines = int(l.split(":")[1])
					break
			else:
				raise
			p_bar = tqdm.tqdm(total=n_lines)

		for l in layer.readlines():
			paths.append(l.replace(' ', '').replace('[', '').replace(']', ''))
			p_bar.update(1)
		return paths


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


def line_to_pair(line: str):
	line_split = line.split(';')
	key = line_split[0]
	value = list(map(int, line_split[1].replace('[', '').replace(']', '').split(',')))
	return key, value


def enumerate_prev(d, kv):
	current, generators = kv
	prev_path: list = generators.copy()
	for g in generators:
		prev_path.remove(g)
		orbit = get_orbit(path_list_to_point(prev_path, d), d)
		prev_path.append(g)
		yield np.array2string(orbit), current


def main():
	d = int(sys.argv[1])
	k = int(sys.argv[2])
	save_dir = sys.argv[3]
	os.makedirs(save_dir, exist_ok=True)
	print(f'processing the layer {k} in dimension {d} ... ')
	cpu_count = multiprocessing.cpu_count()

	start = time.time()
	spark = SparkSession.builder.config(
		"spark.driver.memory", "4g"
	).config(
		'spark.ui.showConsoleProgress', 'true'
	).appName(
		"counting_vertices"
	).getOrCreate()

	upper_layer = spark.sparkContext.textFile(os.path.join(SparkFiles.getRootDirectory(), f'{k}.archive'), minPartitions=cpu_count//2)
	if k > 1:
		lower_layer = spark.sparkContext.textFile(os.path.join(SparkFiles.getRootDirectory(), f'{k-1}.archive'), minPartitions=cpu_count//2)
	else:
		lower_layer = spark.sparkContext.parallelize([f'{np.array([0]*d)};'])

	upper_layer = upper_layer.map(line_to_pair).flatMap(partial(enumerate_prev, d))
	joined_layer = lower_layer.map(
		lambda l: (l.split(';')[0], None)
	).join(upper_layer).map(lambda kv: (kv[0], kv[1][1])).toDF(['lower', 'upper'])


	in_degree = joined_layer.groupBy('upper').count().rdd.map(
		lambda kv: (kv[0], kv[1], kv[1] * n_vertices(kv[0], d)))

	out_degree = joined_layer.rdd.groupByKey().map(
		lambda x: (x[0], sum(map(lambda o: n_vertices(o, d) * list(x[1]).count(o), set(x[1])))))

	in_degree.toDF(
		['vertex', 'in', 'total_in']
	).toPandas().to_csv(os.path.join(save_dir, f'{k}_in.csv'), index=False)

	out_degree.toDF(
		['vertex', 'total_out']
	).toPandas().to_csv(os.path.join(save_dir, f'{k-1}_out.csv'), index=False)

	if k == 2 ** (d-1) - 1:
		spark.sparkContext.textFile(
			os.path.join(SparkFiles.getRootDirectory(), f'{k}.archive'), minPartitions=cpu_count//2
		).map(
			lambda l: l.split(';')[0]
		).map(
			lambda o: (o, 1, n_vertices(o, d))
		).toDF(['vertex', 'out', 'total_out']).toPandas().to_csv(
			os.path.join(save_dir, f'{k}_out.csv'), index=False
		)


if __name__ == '__main__':
	main()
