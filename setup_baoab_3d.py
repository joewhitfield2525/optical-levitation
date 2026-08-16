from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "baoab_3d_cython",
        ["baoab_3d_cython.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="baoab_3d_cython",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
