# <img src="https://www.flaticon.com/svg/static/icons/svg/824/824695.svg" width="3.5%">  sealeveltools

<details> 
<summary>SVG code</summary>

```
@sample.svg
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1440 320"><path fill="#273036" fill-opacity="1" d="M0,224L24,202.7C48,181,96,139,144,144C192,149,240,203,288,186.7C336,171,384,85,432,74.7C480,64,528,128,576,170.7C624,213,672,235,720,256C768,277,816,299,864,277.3C912,256,960,192,1008,154.7C1056,117,1104,107,1152,106.7C1200,107,1248,117,1296,144C1344,171,1392,213,1416,234.7L1440,256L1440,320L1416,320C1392,320,1344,320,1296,320C1248,320,1200,320,1152,320C1104,320,1056,320,1008,320C960,320,912,320,864,320C816,320,768,320,720,320C672,320,624,320,576,320C528,320,480,320,432,320C384,320,336,320,288,320C240,320,192,320,144,320C96,320,48,320,24,320L0,320Z"></path></svg>
@sample.svg
```

</details>


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








