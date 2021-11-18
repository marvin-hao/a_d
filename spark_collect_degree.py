import multiprocessing
import sys
import os
import time
import tqdm
import pandas as pd
import numpy as np

from typing import List, Tuple
from pyspark import SparkFiles
from pyspark.sql.types import IntegerType
from pyspark.sql import SparkSession, functions

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

	in_degree = spark.read.format('csv').option('delimiter', ',').option('header', 'true').load(
		os.path.join(SparkFiles.getRootDirectory(), f'{k}_in.csv'))
	out_degree = spark.read.format('csv').option('delimiter', ',').option('header', 'true').load(
		os.path.join(SparkFiles.getRootDirectory(), f'{k}_out.csv'))

	joined = in_degree.join(out_degree, 'vertex').withColumn(
		'orbit_size', functions.udf(n_vertices, IntegerType())('vertex', functions.lit(d))
	).withColumn(
		'total_degree', (functions.col('total_in') + functions.col('total_out')).cast(IntegerType())
	).withColumn(
		'degree', (functions.col('total_degree') / functions.col('orbit_size')).cast(IntegerType())
	).withColumn(
		'out', (functions.col('degree') - functions.col('in')).cast(IntegerType())
	)

	df: pd.DataFrame = joined.toPandas()[
		['vertex', 'in', 'out', 'degree', 'total_in', 'total_out', 'total_degree', 'orbit_size']
	]

	df.to_csv(os.path.join(save_dir, f'{k}.csv'), index=False)

	in_degree_dist = df.groupby('in')['in'].count().reset_index(name='count')
	out_degree_dist = df.groupby('out')['out'].count().reset_index(name='count')
	degree_dist = df.groupby('degree')['degree'].count().reset_index(name='count')

	in_degree_dist.to_csv(os.path.join(save_dir, f'{k}_in_dist.csv'), index=False)
	out_degree_dist.to_csv(os.path.join(save_dir, f'{k}_out_dist.csv'), index=False)
	degree_dist.to_csv(os.path.join(save_dir, f'{k}_degree_dist.csv'), index=False)

	with open(os.path.join(save_dir, f'{k}_summary.txt'), 'w') as f:
		f.writelines([
			f'min in: {in_degree_dist["in"].min()}\n',
			f'max in: {in_degree_dist["in"].max()}\n',
			f'min out: {out_degree_dist["out"].min()}\n',
			f'max out: {out_degree_dist["out"].max()}\n',
			f'min degree: {degree_dist["degree"].min()}\n',
			f'max degree: {degree_dist["degree"].max()}\n',
		])


if __name__ == '__main__':
	main()
