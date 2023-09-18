# <img src="https://www.flaticon.com/svg/static/icons/svg/824/824695.svg" width="3.5%">  sealeveltools


sealeveltools is a Python package to which bundles frequently applied and required operations to geophysical data. Its main focus is the exploitation of 4-D climate variables (in particular 3-D sea level data) and provides a range of operators for statistical analysis (time series analysis), spatial operators (mapping, interpolating ...) and visualisation in 2-D. Sealeveltools is based on, preserves and extends the functionalities of xarray.

Check out the `getting started guide <https://gitlab.lrz.de/iulius/sea_level_tool/-/blob/master/sealeveltools_tutorial.ipynb>`.

Features
========

- Simple math and statistical operations 
    * Different fitting algorithms for trend/seasonality and uncertainty estimation
    * Correlations
    * EOFs
- Plotting and Visualization
- Spatial manipulations
    * Regridding, smoothing 
    * Combining data in space
- Time operators
- Custom functions, miscellaneous


Installation
============

### Linux

Install PIP

    $ apt install python3-pip

Install miniconda

    $ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    $ bash Miniconda3-latest-Linux-x86_64.sh

Install the package with

    $ pip install git+https://gitlab.lrz.de/iulius/sea_level_tool.git@master#egg=sealeveltools

Install two further required packages with conda:

    $ conda install -c conda-forge cartopy

    $ conda install -c conda-forge iris

## Usage/Getting started


use sealeveltools in python3

from sealeveltools.sl_class import *

now you have the sl() class to get started

type sl().info() to show all of its functions

Check out the `getting started guide <https://gitlab.lrz.de/iulius/sea_level_tool/-/blob/master/sealeveltools_tutorial.ipynb>`.

to get an overview of the functionailities

*Note: A full documenation is not yet available








