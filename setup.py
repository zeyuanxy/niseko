import sys

from setuptools import setup, find_packages


# check Python version
if sys.version_info < (3, 6):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. Davos requires Python '
        '3.6 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

packages = find_packages(where='.')
with open('requirements.txt') as f:
    install_requires = [r.rstrip() for r in f.readlines()
                        if not r.startswith('#') and not r.startswith('git+')]

setup(
    name='niseko',
    author='Zeyuan Shang',
    author_email='zeyuanxy@gmail.com',
    description='Niseko: A Large-Scale Meta-Learning Platform',
    version='0.2',
    packages=packages,
    install_requires=install_requires,
    include_package_data=True,
    python_requires='>=3.6.*',
    url='https://www.shangzeyuan.com/'
)
