from setuptools import setup, find_packages
import tirx_tutorial

requirements = [
    'jupyter',
    'numpy',
    'matplotlib',
    'requests',
    'pandas',
    'sphinx',
]

setup(
    name='tirx-tutorial',
    version=tirx_tutorial.__version__,
    python_requires='>=3.10',
    author='TIRX Developers',
    url='https://github.com/mlc-ai/tirx-tutorial',
    description='TIRX Tutorial',
    packages=find_packages(),
    zip_safe=True,
    install_requires=requirements,
)
