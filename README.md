### Prerequisites:

1. CPython interpreter v3.6.
2. IBM ILOG CPLEX Optimizer v12.10.0 and its bundled cplex Python package.
3. A Spark (v3.0.3) cluster on Hadoop (v2.7).

### Installation:
1. Install requirements: `sudo pip3.6 install -r requirements.txt`
2. Install the algorithm on all Spark node: `sudo python3.6 setup.py install`

### Operations:

#### Calculate a(d) within a single Python interpreter
1. Update the dimension in `main.py`
2. Run `python3.6 main.py`. The result will be saved in a directory `a_dim`.
3. To get the total number of vertices, update the dimension in `collect_total.py` and run `python3.6 collect_total.py`.

#### Calculate a(d) on a Spark cluster.
1. Update the dimension in `find_vertices_on_spark.py`.
2. Provide parameters to `find_vertices_on_spark.sh`
3. Make `find_vertices_on_spark.py` an executable and run the script.

#### Calculate e(d) on a Spark cluster.
1. Provide parameters to `calculate_degree_on_spark.sh`, and `collect_degree_on_spark.sh`.
2. Run `calculate_degree_on_spark.sh` to generate intermediate results.
3. Run `collect_degree_on_spark.sh` to gather degree information for each layer.
4. Run Update the dimension in `collect_total_edges.py` to get the total number of edges of the zonotope.

