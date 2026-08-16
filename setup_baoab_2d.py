from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        "baoab_2d_cython",
        ["baoab_2d_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]


setup(
    name="baoab_2d_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
            "language_level": 3,
        },
    ),
)
