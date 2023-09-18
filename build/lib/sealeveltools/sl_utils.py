import xarray as xr
import pandas as pd
import os
import os.path
from os import path


def load_testfiles():
    dirname=os.path.dirname(__file__)
    dire  =  os.path.join(dirname, 'data/')

    if path.exists(dire):
        sat=xr.open_dataset(dire+'sat_data.nc')
        tg=xr.open_dataset(dire+'tg_data.nc')
        gnss=xr.open_dataset(dire+'gps_data.nc')    
        gia=xr.open_dataset(dire+'GIA_data.nc')
        ci=pd.read_csv(dire+'climate_indices.csv',index_col=[0])
        ci.index=pd.to_datetime(ci.index.values)
        return sat,tg,gnss,gia,ci
    else:
        print('No testfiles available yet. include them in side-package directory: sealeveltools/sealeveltools/data/')