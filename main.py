import time
import warnings
from enumerate_vertices import run, restart

warnings.filterwarnings('ignore')


def main():
	d = 7
	concurrency = 16
	start = time.time()
	run(d, concurrency, checkpoint=restart(d, concurrency))
	print(f'Time spent: {time.time() - start:0.3f}')


# run_lp(d)


if __name__ == '__main__':
	main()

