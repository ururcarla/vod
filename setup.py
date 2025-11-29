import os
from setuptools import setup, Extension
import numpy as np

this_dir = os.path.abspath(os.path.dirname(__file__))
common_dir = os.path.join(this_dir, 'common')
pycoco_dir = os.path.join(this_dir, 'pycocotools')

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        'pycocotools._mask',
        sources=[
            os.path.join(common_dir, 'maskApi.c'),
            os.path.join(pycoco_dir, '_mask.pyx'),
        ],
        include_dirs = [np.get_include(), common_dir],
        extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(
    name='pycocotools',
    packages=['pycocotools'],
    package_dir = {'pycocotools': 'pycocotools'},
    install_requires=[
        'setuptools>=18.0',
        'cython>=0.27.3',
        'matplotlib>=2.1.0'
    ],
    version='2.0',
    ext_modules= ext_modules
)
