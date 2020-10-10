from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import os

os.environ['CC'] = 'zig cc'
os.environ['CXX'] = 'zig cc'

rrrvec = Extension(
    'rrrvec',
    ['rrrvec.pyx'],
    extra_compile_args=['-O2'],
    extra_link_args=[],
    libraries=None,
)

setup(
    name = 'rrrvec',
    ext_modules = cythonize([rrrvec], language_level='3'),
    zip_safe = False,
)
