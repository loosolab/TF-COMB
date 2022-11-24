import os
import sys
import re
from setuptools import setup, Extension, dist

# Test if numpy is installed
try:
    import numpy as np
except:
    # Else, fetch numpy if needed
    dist.Distribution().fetch_build_eggs(['numpy'])
    import numpy as np

# Add cython modules depending on the availability of cython
cmdclass = {}
try:
    from Cython.Distutils import build_ext
    cmdclass = {'build_ext': build_ext}
except ImportError:
    use_cython = False
else:
    use_cython = True

# To compile or not to compile
if use_cython:
    ext_modules = [Extension("tfcomb.counting", ["tfcomb/counting.pyx"], include_dirs=[".", np.get_include()])]
else:
    if os.path.exists("tfcomb/counting.c"):
        ext_modules = [Extension("tfcomb.counting", ["tfcomb/counting.pyx"], include_dirs=[".", np.get_include()])]
    else:  # Only happens when installing directly from source
        sys.exit("Cython is needed to compile TF-comb.")

# Path of setup file to establish version
setupdir = os.path.abspath(os.path.dirname(__file__))


def find_version(init_file):
    version_file = open(init_file).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")


# Readme from git
def readme():
    with open('README.md') as f:
        return f.read()

setup(name='TF-COMB',
        version=find_version(os.path.join(setupdir, "tfcomb", "__init__.py")),   # get version from __init__.py
        description='Transcription Factor Co-Occurrence using Market Basket analysis',
        long_description=readme(),
        long_description_content_type='text/markdown',
        url='',
        author='Mette Bentsen',
        author_email='mette.bentsen@mpi-bn.mpg.de',
        license='MIT',
        packages=["tfcomb"],
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        python_requires='>=3',
        setup_requires=["numpy"],
        install_requires=[
            'numpy',
            'scipy',
            'pysam',
            'matplotlib>=2',
            'pandas',
            'tobias',
            'networkx>=2.4',  # G.node -> G.nodes
            'python-louvain',
            'graphviz',
            'statsmodels',
            'goatools',
            'uropa',
            'qnorm',
            'dill',
            'seaborn',
            'tqdm',
            'IPython'
        ],
        classifiers=[
            'License :: OSI Approved :: MIT License',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Programming Language :: Python :: 3'
        ],
        zip_safe=False,  # gives cython import error if True
        include_package_data=True
        )
