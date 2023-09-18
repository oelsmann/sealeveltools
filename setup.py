from setuptools import setup, find_packages


packages = find_packages(exclude=("tests",'scripts','sl_bayes.py',
                                  '*png','*nc','w*ipynb','logs','trash'))

setup(
    name="sealeveltools",
    version='0.0.1',
    description='A project to handle sea level data (altimetry, tide-gauges, models) and statistical exploitation tools',
    license='',
    author='Julius Oelsmann',
    author_email='julius.oelsmann@tum.de',
    packages=packages,
    url="https://gitlab.lrz.de/iulius/sea_level_tool.git",
    install_requires=['scipy', 'matplotlib','pandas','xarray',
                      'numpy','eofs','seaborn','sympy'],
    python_requires='>=3.6',
    #package_data={    },
    #entry_points={    },
    #setup_requires=["pytest-runner"],
)

