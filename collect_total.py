import os


def main(d):
	"""
	To sum up the number of vertices in all the layers.
	:returns: The number of vertices in dimension d.

	"""
	meta_files = [int(f.split('.')[0]) for f in os.listdir(f'a_{d}') if f.endswith('.meta')]
	meta_files.sort(reverse=True)
	total = 2

	for k in meta_files:
		meta = open(os.path.join(f'a_{d}/{k}.meta'), 'r')
		subtotal = int(meta.readlines()[1].strip().split(':')[1])
		total += subtotal

	return total


if __name__ == '__main__':
	d = 7
	print(main(d))
