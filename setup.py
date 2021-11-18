import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
	Extension('enumerate_vertices', sources=['enumerate_vertices.pyx'], language='c++', extra_compile_args=['-O3']),
	Extension('find_vertices', sources=['find_vertices.pyx'], language='c++', extra_compile_args=['-O3'])
]

setup(
	name='counting_vertices',
	url='',
	license='',
	author='mingfei',
	author_email='',
	description='',
	ext_modules=cythonize(extensions, compiler_directives={
		'language_level': "3",
		# 'linetrace': True,
		# 'binding': True
	},
	annotate=True),
	include_dirs=[numpy.get_include()]
)
