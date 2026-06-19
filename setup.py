from setuptools import setup, find_packages
import tirx_tutorial

requirements = [
    'jupyter',
    'numpy',
    'matplotlib',
    'requests',
    'pandas',
    'sphinx==5.3.0',
    # Pin sphinxcontrib-bibtex and pybtex together: newer sphinxcontrib-bibtex
    # (>=2.6) relies on a private symbol `_FakeEntryPoint` that only exists in
    # pybtex 0.24.x. Mismatched versions break `import sphinxcontrib.bibtex`,
    # which d2lbook always triggers via _build/rst/conf.py.
    'sphinxcontrib-bibtex<2.6',
    'pybtex<0.25',
]

setup(
    name='tirx-tutorial',
    version=tirx_tutorial.__version__,
    python_requires='>=3.10',
    author='MLC Community',
    description='Modern GPU Programming',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
