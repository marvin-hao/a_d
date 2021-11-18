import os


def main(d):
	degree_files = [int(f.split('.')[0]) for f in os.listdir(f'a_{d}_degree') if f.split('.')[0].isnumeric()]
	degree_files.sort()
	total_degree = 0

	for k in degree_files:
		with open(os.path.join(f'a_{d}_degree/{k}.csv'), 'r') as f:
			sub_total = 0
			lines = f.readlines()[1:]
			for l in lines:
				degree = int(l.split(',')[-2])
				sub_total += degree
				total_degree += degree
			print(k, sub_total)

	return total_degree // 2 + d


if __name__ == '__main__':
	d = 7
	print(main(d))
