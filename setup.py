from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='cbmanifold',
    version='0.1',
    long_description=open('README.md').read(),
    python_requires='>=3.11',
    install_requires=['numpy', 'scipy','scikit-learn', 'matplotlib'],
    #package_data={'cbmanifold': ['*/requirements.txt']},
    #dependency_links = [],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
)
