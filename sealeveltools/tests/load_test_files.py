from os import listdir
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy import signal
import xarray as xr
import os
import iris
from sealeveltools.sl_class import *
import codecs


def load_sat_old(name='balticplus_gridded',syst='xr',fileDir='/home/oelsmann/Julius/Scripts/balticplus/exploit_functions'):

    """
    load data-sets
    """


    #fileDir = os.path.dirname(os.path.realpath('__file__'))
    if 'north' in name:
        
        if name=='AVISO_north_sea':
            
            filedir='/home/oelsmann/Julius/Scripts/north_sea/data/aviso/'
            ds=xr.open_dataset(filedir+name+'_cut.nc')            
        elif name =='SLcci_merged_north_sea':
            
            filedir='/home/oelsmann/Julius/Scripts/north_sea/data/slcci/'
            ds=xr.open_dataset(filedir+name+'_cut.nc')        
        else:
        
            filedir='/home/oelsmann/Julius/Scripts/north_sea/data/grids/'
            ds=xr.open_dataset(filedir+name+'.nc')

    else:
    

        if name=='balticplus_gridded':
            filedir=os.path.join(fileDir, '../dataset_lfs/balticplus_gridded/monthly_grids_KO8/')
            name_in='baltic_gridded_merged.nc'
            name='100_rem_res_new_flag_grids'
            name_in='baltic_gridded_merged_'+name+'.nc'  
        elif name=='slcci_gridded':
            filedir=os.path.join(fileDir, '../dataset_lfs/slcci_gridded/SeaLevel-ECV/V2.0_20161205/')
            name_in='SLcci_merged_baltic.nc'
        elif name=='aviso_gridded':
            filedir=os.path.join(fileDir, '../dataset_lfs/aviso_gridded/')
            name_in='AVISO_baltic.nc'
        else:
            filedir=os.path.join(fileDir, '../dataset_lfs/balticplus_gridded/monthly_grids_KO8/')
            name_in='baltic_gridded_merged_'+name+'.nc'       

        if syst=='xr':
            ds=xr.open_dataset(filedir+name_in)
        else:
            ds = iris.load(filedir+name_in)

    return ds



def test_merge_gridded_data(time_dim_type='pandas',save=True, out=False,name=''):
    """
    Reads in files in '../dataset_lfs/balticplus_gridded/monthly_grids_KO8'
    and outputs a merged 3D-xarray file (baltic_gridded_merged.nc)
    
    Parameters: time_dim_type: str, optional
                    'pandas' for pandas.DatetimeIndex or 'numpy' for numpy.datetime[ns] data-type
   
    
    returns:      
    """
    
    

    filedir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/'+name
    
    os.listdir(filedir)
    time_array=pd.date_range(start='1995-05-15',end='2019-06-15', freq='M')
    np_time_array=np.arange('1995-05', '2019-06', dtype='datetime64[M]').astype('datetime64[ns]')
    
    sshr_data=np.empty([len(time_array),143,60])*np.nan
    ssh_std_data=np.empty([len(time_array),143,60])*np.nan
    num_obs_data=np.empty([len(time_array),143,60])*np.nan
    num_used_obs_data=np.empty([len(time_array),143,60])*np.nan
    
    sshr_data
    i=0
    for filename in time_array:
        month=filename.month
        if month < 10:
            month_str=str(0)+str(month)
        else:
            month_str=str(month)

        grid_in=xr.open_dataset(filedir+'/'+str(filename.year)+'_'+month_str+'.nc')
        
        ssh=grid_in.ssh.values
        ssh_std=grid_in.ssh_std.values
        num_obs=grid_in.num_obs.values
        num_used_obs=grid_in.num_used_obs.values
        
        sshr=ssh[0:-1].reshape(143,60)
        ssh_stdr=ssh_std[0:-1].reshape(143,60)
        num_obsr=num_obs[0:-1].reshape(143,60)
        num_used_obsr=num_used_obs[0:-1].reshape(143,60)
        
        
        #sshr=np.expand_dims(sshr,axis=2)

        sshr_data[i,:,:]=sshr
        ssh_std_data[i,:,:]=ssh_stdr
        num_obs_data[i,:,:]=num_obsr
        num_used_obs_data[i,:,:]=num_used_obsr
        
        
        i=i+1
    #sshr_data[:,:,0]
    if time_dim_type=='pandas':
        tar=time_array
    else:
        tar=np_time_array
    
    lonr=grid_in.lon.values[0:-1].reshape(143,60) 
    latr=grid_in.lat.values[0:-1].reshape(143,60) 
    ds = xr.Dataset({'ssh': (['time','x', 'y'],  sshr_data), 
                     'ssh_std': (['time','x', 'y'],  ssh_std_data),
                    'num_obs': (['time','x', 'y'],  num_obs_data),
                    'num_used_obs': (['time','x', 'y'],  num_used_obs_data),},
                         coords={'lon': (['x', 'y'], lonr),
                        'lat': (['x', 'y'], latr),
                        'time':tar}) 
    ds
    ds.ssh.attrs={'standard_name': 'sea_surface_height', 'long_name': 'sea surface height','_FillValue' : 'nan'}
    ds.ssh_std.attrs={'standard_name': 'ssh_standard_deviation', 'long_name': 'ssh standard_deviation','_FillValue' : 'nan'}
    ds.num_obs.attrs={'standard_name': 'Number_of_theoretical_usable_observation', 'long_name': 'Number of theoretical usable observations','_FillValue' : 'nan'}
    ds.num_used_obs.attrs={'standard_name': 'Number_of_used_observation', 'long_name': 'Number of used observations','_FillValue' : 'nan'}
    
    ds.lon.attrs={'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east', '_CoordinateAxisType': 'Lon'}
    ds.lat.attrs={'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north', '_CoordinateAxisType': 'Lat'}
    ds.time.attrs={'standard_name':  'time','axis': 'T'}
    
    
    #fileDir = os.path.dirname(os.path.realpath('__file__'))
    #filedir = os.path.join(fileDir, '../dataset_lfs/balticplus_gridded/monthly_grids_KO8')
    filedir='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/balticplus_gridded/monthly_grids_KO8'
    
    if save:
        ds.to_netcdf(filedir+'/baltic_gridded_merged_'+name+'.nc')
    if out:
        return ds
  
def load_big_data(par='msl_0001',extend='',opt='standard'):
    """
    open SLP etc. dataset
    """
    if opt=='standard':
        big_data=xr.open_dataset('/home/oelsmann/Julius/baltic_plus/old_data/data/aux/adaptor.mars.internal-1578589481.8398316-18892-25-4250771e-ddbf-48ba-9d9d-18024e7e56b2.nc')

        if par=='all':
            data=big_data.rename({'latitude':'lat','longitude':'lon'})    
        else:
            data=big_data[par].rename({'latitude':'lat','longitude':'lon'})  
        lon=data['lon'].values
        lon_new=lon
        #lon_new[lon_new>180]=lon_new[lon_new>180]-360.
        #data=data.assign_coords(lon=lon_new)
        #data=data.assign_coords(lon=(((data.lon + 180) % 360) - 180))
        if extend=='':
            print('get all')
        else:
            if extend[0]<0:
                print('here')
                domain = (
                 ((data.coords["lat"] > extend[2]) &
                  (data.coords["lat"] < extend[3]) &
                 (data.coords["lon"] > 0) &
                 (data.coords["lon"] < extend[1]) |
                 (data.coords["lon"] > 360+extend[0]) &
                 (data.coords["lon"] < 360) &
                 (data.coords["lat"] > extend[2]) &
                  (data.coords["lat"] < extend[3]))
                )        
            else:
                domain = (
                 ((data.coords["lat"] > extend[2]) &
                  (data.coords["lat"] < extend[3]) &
                 (data.coords["lon"] > extend[0]) &
                 (data.coords["lon"] < extend[1]) )
                )
            data = data.where(domain,drop=True) 
        #data = data.dropna(dim='lon',how='all')
        #data = data.dropna(dim='lat',how='all')
        #data=data.transpose('time','lat','lon')
    elif opt=='ym':
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_ym_baltic_box.nc').rename({'longitude':'lon','latitude':'lat'})
    elif opt=='JJA':
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_JJA_baltic_box.nc').rename({'longitude':'lon','latitude':'lat'})
    
    elif opt=='DJF':
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_DJF_seas_shift_ym_box.nc').rename({'longitude':'lon','latitude':'lat'})

    elif opt=='trend':
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_ym_trend_box.nc').rename({'longitude':'lon','latitude':'lat'})        
        data=data.drop('time_bnds')
        data=data.squeeze(dim='time')  
    elif opt=='trend_DJF':    
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_DJF_trend_box.nc').rename({'longitude':'lon','latitude':'lat'})                 
        data=data.drop('time_bnds')
        data=data.squeeze(dim='time')
    elif opt=='trend_JJA':          
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_JJA_trend_box.nc').rename({'longitude':'lon','latitude':'lat'})        
        data=data.drop('time_bnds')
        data=data.squeeze(dim='time')   
    elif opt=='standard_DJF':    
        dirr='/home/oelsmann/Julius/baltic_plus/data/aux/'
        data=xr.open_dataset(dirr+'ERA5_DJF_seas_shift_ym_box.nc').rename({'longitude':'lon','latitude':'lat'})                 
        data=data.drop('time_bnds')
        #data=data.squeeze(dim='time')
    elif opt=='standard_JJA':          
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_JJA_box.nc').rename({'longitude':'lon','latitude':'lat'})        
        data=data.drop('time_bnds')
        #data=data.squeeze(dim='time')           
    elif opt=='mean_DJF':    
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_DJF_box_avg.nc').rename({'longitude':'lon','latitude':'lat'})                 
        data=data.drop('time_bnds')
        data=data.squeeze(dim='time')
    elif opt=='mean_JJA':          
        dirr='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/atmospheric_fields_ERA5/eval/'
        data=xr.open_dataset(dirr+'ERA5_JJA_box_avg.nc').rename({'longitude':'lon','latitude':'lat'})        
        data=data.drop('time_bnds')
        data=data.squeeze(dim='time')          
        
    return data

def test_merge_gridded_data_params(time_dim_type='pandas',save=True, out=False,name='',params=['ssh','ssh_std','num_obs','num_used_obs','sig0_fitting_error','helm_p','mean_time_obs','mean_time_obs_std']):
    """
    Reads in files in '../dataset_lfs/balticplus_gridded/monthly_grids_KO8'
    and outputs a merged 3D-xarray file (baltic_gridded_merged.nc)
    
    Parameters: time_dim_type: str, optional
                    'pandas' for pandas.DatetimeIndex or 'numpy' for numpy.datetime[ns] data-type
   
    
    returns:      
    """
    
    

    filedir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/'+name
    
    os.listdir(filedir)
    time_array=pd.date_range(start='1995-05-15',end='2019-06-15', freq='M')
    np_time_array=np.arange('1995-05', '2019-06', dtype='datetime64[M]').astype('datetime64[ns]')
    
    sshr_data=np.empty([len(time_array),143,60])*np.nan

    i=0
    DATA=[sshr_data]*len(params)
    ATTRS=[]
    for filename in time_array:
        month=filename.month
        if month < 10:
            month_str=str(0)+str(month)
        else:
            month_str=str(month)

        grid_in=xr.open_dataset(filedir+'/'+str(filename.year)+'_'+month_str+'.nc',drop_variables ='id_mission')
        
        jj=0
        for var in params:
            small_arr=DATA[jj]
            
            small_arr[i,:,:]=grid_in[var].values[0:-1].reshape(143,60)
            DATA[jj]=copy.deepcopy(small_arr)
            ATTRS.append(grid_in[var].attrs)      
            jj=jj+1
            
        i=i+1
    #sshr_data[:,:,0]

    if time_dim_type=='pandas':
        tar=time_array
    else:
        tar=np_time_array
    
    lonr=grid_in.lon.values[0:-1].reshape(143,60) 
    latr=grid_in.lat.values[0:-1].reshape(143,60) 
    _all=[]
    jj=0
    for var in params:
        
        ds = xr.Dataset({var: (['time','x', 'y'],  DATA[jj])},
                             coords={'lon': (['x', 'y'], lonr),
                            'lat': (['x', 'y'], latr),
                            'time':tar})         

        ds.attrs=ATTRS[jj]
        ds.lon.attrs={'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east', '_CoordinateAxisType': 'Lon'}
        ds.lat.attrs={'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north', '_CoordinateAxisType': 'Lat'}
        ds.time.attrs={'standard_name':  'time','axis': 'T'}
        jj=jj+1
        _all.append(ds)

    
    if save:
        xr.merge(_all,compat='override').to_netcdf('/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/balticplus_gridded/monthly_grids_KO8'+'/baltic_gridded_merged_'+name+'.nc')
    if out:
        return xr.merge(_all,compat='override')     
    
 
    
def load_sat_test(name='100_grids',syst='xr'):
    
    """
    load data-sets
    """
    
    
    #fileDir = os.path.dirname(os.path.realpath('__file__'))

    
    #filedir=os.path.join(fileDir, '../dataset_lfs/balticplus_gridded/monthly_grids_KO8/') 
    
    name_in='baltic_gridded_merged_'+name+'.nc'
    filedir='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/balticplus_gridded/monthly_grids_KO8/'
    if syst=='xr':
        ds=xr.open_dataset(filedir+name_in)       
    else:
        ds = iris.load(filedir+name_in)
        
    return ds
    

def load_north_sea(time_dim_type='pandas',save=True, out=False,name='',
                   params=['ssh','ssh_std','sig0_fitting_error','num_used_obs','num_obs',
                           'mean_time_obs']):
    """
    Reads in files in '/nfs/DGFI36/altimetry/North_Sea_plus/'
    and outputs a merged 3D-xarray file (North_sea_gridded_merged.nc)
    
    Parameters: time_dim_type: str, optional
                    'pandas' for pandas.DatetimeIndex or 'numpy' for numpy.datetime[ns] data-type
   
    
    [sla, var_sla, var_sigma_0,num_used_obs, num_theo_obs,num_itt,mean(time_of_obs)];
    
    returns: xr.dataset
    """
    

    
    filedir='/nfs/DGFI36/altimetry/North_Sea_plus/'+name
    grid=h5py.File('/nfs/DGFI36/altimetry/North_Sea_plus/LEVEL_10_SEA_LEVEL_GRID_North_SEA.mat')
    half_monthly=False
    if name == 'grids02_02':
        data_temp=h5py.File(filedir+'/2009_07.mat', 'r')
    
    elif name =='grids02_04':
        data_temp=h5py.File(filedir+'/2005_07_14.mat', 'r')        
        half_monthly=True
    else:
        data_temp=h5py.File(filedir+'/2009_07_16.mat', 'r')

    
    
    os.listdir(filedir)
    
    if half_monthly:
        dirrr=os.listdir(filedir)

        dates=[]
        for date in dirrr:
            date_real=pd.to_datetime(date[0:4]+'-'+date[5:7]+'-'+date[8:10])
            dates.append(date_real)
        time_array=pd.DatetimeIndex(dates)
        sshr_data=np.empty([len(time_array),data_temp['Z'].shape[1]])*np.nan
    else:
        
        time_array=pd.date_range(start='1995-05-15',end='2019-06-15', freq='M')
        np_time_array=np.arange('1995-05', '2019-06', dtype='datetime64[M]').astype('datetime64[ns]')

        sshr_data=np.empty([len(time_array),data_temp['Z'].shape[1]])*np.nan

    i=0
    DATA=[sshr_data]*len(params)
    ATTRS=[]
    for filename in time_array:
        month=filename.month
        if month < 10:
            month_str=str(0)+str(month)
        else:
            month_str=str(month)
        #print(glob.glob(filedir+'/'+str(filename.year)+'_'+month_str+'*.mat'))
        
        if half_monthly:
            day=filename.day
            if day < 10:
                day_str=str(0)+str(day) 
            else:
                day_str=str(day) 
   
            file_mat=glob.glob(filedir+'/'+str(filename.year)+'_'+month_str+'_'+day_str+'.mat')[0]
        
        else:
            file_mat=glob.glob(filedir+'/'+str(filename.year)+'_'+month_str+'*.mat')[0]
                    
            

        grid_in=h5py.File(file_mat)

        jj=0
        for var in params:
            small_arr=DATA[jj]
            
            small_arr[i,:]=grid_in['Z'][jj,:]
            DATA[jj]=copy.deepcopy(small_arr)
            ATTRS.append({})      
            jj=jj+1
            
        i=i+1
    #sshr_data[:,:,0]

    if time_dim_type=='pandas':
        tar=time_array
    else:
        tar=np_time_array
    
    lonr=grid['koord'][0,:]
    latr=grid['koord'][1,:]
    _all=[]
    jj=0
    for var in params:
        
        ds = xr.Dataset({var: (['time','x'],  DATA[jj])},
                             coords={'lon': (['x'], lonr),
                            'lat': (['x'], latr),
                            'time':tar})         

        ds.attrs=ATTRS[jj]
        ds.lon.attrs={'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east', '_CoordinateAxisType': 'Lon'}
        ds.lat.attrs={'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north', '_CoordinateAxisType': 'Lat'}
        ds.time.attrs={'standard_name':  'time','axis': 'T'}
        jj=jj+1
        _all.append(ds)

    
    if save:
        xr.merge(_all,compat='override').to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/data/grids/'+'north_sea_gridded_merged_'+name+'.nc')
    if out:
        return xr.merge(_all,compat='override')

def load_tg(kind='psmsl',filedir='/home/oelsmann/Julius/Scripts/balticplus/exploit_functions'):
    """
    load psmsl or gesla tide gauge data
    corrected with DAC (Carrere and Lyard as in ZOI-paper)
    """
    #fileDir = os.path.dirname(os.path.realpath('__file__'))
    if kind =='psmsl':
        filedir=os.path.join(filedir, '../dataset_lfs/psmsl/')
        #tg=xr.open_dataset(filedir+'PSMSL_DAC_corrected_name.nc')
        tg=xr.open_dataset(filedir+'PSMSL_DAC_corrected_name_datetupdate.nc')


    elif kind =='gesla':
        print('not yet integrated')
    elif kind=='psmsl_full':
        # no  DAC corrected long time series
        tg=xr.open_dataset('/home/oelsmann/Julius/Scripts/vlad/data/VLAD/WP1/PSMSL/PSMSL_full_time_no_correction.nc')
        
    elif kind=='bafg':
        # DAC corrected bafg files
        tg=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/tgs/bafg/TG_TnwThw_DAC_corrected_3std.nc')
        
    elif kind=='bafg_psmsl':
        # load merged tg
        ds=load_tg(kind='bafg')
        psmsl=load_tg(kind='psmsl')
        psmsl
        tg=xr.concat([psmsl, ds], dim="x")    
    return tg



def load_DAC():
    """
    load ECMWF DAC
    """
    
    DAC=xr.open_dataset('/home/oelsmann/data/VLAD/WP1/aux_data/DAC/DAC_monthly_mean/DAC_mm_all.nc')
    time_array=pd.date_range(start='1/1/1993',end='08/1/2019', freq='M')
    DAC_cut=DAC.dac[:-1,:,:]
    DAC_cut['time']=time_array
    DAC_cut=DAC_cut.rename({'latitude':'lat','longitude':'lon'})
    lon=DAC_cut.lon.values
    lon[lon>180]=lon[lon>180]-360
    DAC_cut['lon']=lon

    return DAC_cut
    

def load_ci(filedir='/home/oelsmann/Julius/Scripts/balticplus/exploit_functions'):
    """
    read teleconnection indices from NOAA
    https://www.climate.gov/news-features/understanding-climate/climate-variability-north-atlantic-oscillation
    """
    #fileDir = os.path.dirname(os.path.realpath('__file__'))
    filedir='/home/oelsmann/data/projects/balticplus/dataset_lfs/climate_indices/'

    tele=pd.read_csv(filedir+'climate_indices',header=15,delim_whitespace=True)
    arctic_osc=pd.read_csv(filedir+'arctic_oscillation',delim_whitespace=True,names=['year','month','data'])
    tele=tele.rename(columns={'yyyy':'year','mm':'month'})
    tele['day']=15
    arctic_osc['day']=15
    index_arctic=pd.to_datetime(arctic_osc[['year','month','day']])
    inex_tele=pd.to_datetime(arctic_osc[['year','month','day']])
    tele['AO']=arctic_osc['data']
    tele['date']=inex_tele
    tele=tele.set_index('date')

    return tele.drop(['year', 'month','Expl.','Var.','day'], axis=1)



###

def read_grdc_details(text):
    """
    read certain parameters
    """
    s=text
    start = 'GRDC-No.:'
    end = '# Ri'
    GRDC_No=float(s[s.find(start)+len(start):s.rfind(end)])


    start = 'River: '
    end = '# Station: '
    River=s[s.find(start)+len(start):s.rfind(end)]


    River=River.replace('\r\n', '')
    River=River.replace(' ', '')
    
    start = '# Station: '
    end = '# Country:'
    Station =    s[s.find(start)+len(start):s.rfind(end)]
    Station=Station.replace('\r\n', '')
    Station=Station.replace(' ', '')
            
    start = '# Country:'
    end = '# Latitude (DD):'
    Country=s[s.find(start)+len(start):s.rfind(end)]
    Country=Country.replace('\r\n', '')
    Country=Country.replace(' ', '')
    
    start = '# Longitude (DD):'
    end = '# Catchme'
    lon=float(s[s.find(start)+len(start):s.rfind(end)])

    start = '# Latitude (DD): '
    end = '# Longitude (DD): '
    lat=float(s[s.find(start)+len(start):s.rfind(end)])

    start = '# Catchment area (km):'
    end = '# Altitude ('
    Catchment=float(s[s.find(start)+len(start):s.rfind(end)])

    start = '# Next downstream station:'
    end = '# Remarks: '
    Next_stat=s[s.find(start)+len(start):s.rfind(end)]
    
    Next_stat=Next_stat.replace('\r\n', '')
    Next_stat=Next_stat.replace(' ', '')
    
    
    
    if Next_stat=='-':
        Next_stat=np.nan
    else:
        Next_stat=float(Next_stat)
    
    return GRDC_No,River,Station,Country,lon,lat,Catchment,Next_stat


def load_runoff(make=False,load=True,out=True,start='1990'):
    """
    
    
    """
    
    if load:

        new_coup=xr.open_dataset('/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/aux/river_discharge/discharge_grdc_baltic_'+start)
        new_trend=xr.open_dataset('/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/aux/river_discharge/discharge_grdc_baltic_trend'+start)
    elif make:

        Period=pd.date_range(start='1/1/1900', end='1/08/2019',freq='M')
        filedir='/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/aux/river_discharge/raw_grdc/'

        data_vec=np.empty([len(Period),len(os.listdir(filedir))])*np.nan
        SPEC=[]
        DATA=[]
        i=0
        for filename in os.listdir(filedir):
            ff=pd.read_csv(filedir+filename,sep=';',delimiter=';',comment='#')
            ff['YYYY-MM-DD']=pd.to_datetime(ff['YYYY-MM-DD'])
            ff=ff.set_index('YYYY-MM-DD')
            ff=ff.drop(['hh:mm'],axis=1)

            ff=ff[ff[' Calculated']!=-999.000]
            ff=ff.resample('M').mean()
            ff=ff['1/1/1900' : '1/08/2019']
            data_vec[np.isin(Period,ff.index),i]=ff[' Calculated']

            f=codecs.open(filedir+filename, 'r', encoding='utf-8',
                             errors='ignore') 
            text=f.read()


            GRDC_No,River,Station,Country,lon,lat,Catchment,Next_stat=read_grdc_details(text)
            SPEC.append([GRDC_No,River,Station,Country,lon,lat,Catchment,Next_stat])

            print(i),
            i=i+1

        specs=pd.DataFrame(SPEC,columns=['GRDC_No','River','Station','Country','lon','lat','Catchment','Next_stat'])    

        ds = xr.Dataset({'discharge': (['time','x'],  data_vec)},
                             coords={'lon': (['x'], specs['lon']),
                            'lat': (['x'], specs['lat']),
                            'time':Period.astype('datetime64[ns]'),
                            'lat': (['x'], specs['lat']),
                             'GRDC_No': (['x'], specs['GRDC_No']),
                             'River': (['x'], specs['River']),
                             'Station': (['x'], specs['Station']),
                             'Country': (['x'], specs['Country']),
                             'Catchment': (['x'], specs['Catchment']),
                             'Next_stat': (['x'], specs['Next_stat'])})


        ds.discharge.attrs={'standard_name': 'discharge', 'long_name': 'mean monthly discharge',
                            '_FillValue' : 'nan','units' : 'm^3/s'}
        ds.lon.attrs={'standard_name': 'longitude', 'long_name': 'longitude', 'units': 'degrees_east',
                      '_CoordinateAxisType': 'Lon'}
        ds.lat.attrs={'standard_name': 'latitude', 'long_name': 'latitude', 'units': 'degrees_north',
                      '_CoordinateAxisType': 'Lat'}
        ds.time.attrs={'standard_name':  'time','axis': 'T'}
        ds=ds.loc[dict(time=slice(start+'-01-01', '2019-06-01'))]
        
        ds_mean=ds.mean(dim='time')
        ds_domain=ds_mean.groupby("River").max()

        arr=np.isin(ds_mean.discharge.values,ds_domain.discharge.values)
        ds_mean=ds_mean.discharge[arr]

        new=ds.discharge[:,np.isin(ds.lon.values,ds_mean.lon.values)*np.isin(ds.lat.values,ds_mean.lat.values)]
        new=new.where(new.count(dim='time')>240,drop=True)
        name='100_rem_res_new_flag_grids_test11'
        baltic=load_sat(name=name)['ssh']
        baltic,new_coup=sl(baltic).couple(new)
        new_coup.data.to_netcdf('/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/aux/river_discharge/discharge_grdc_baltic_'+start)
        new_trend=sl(new_coup).trend(monthly=True).data
        new_trend.to_netcdf('/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/aux/river_discharge/discharge_grdc_baltic_trend'+start)
                            
    if out:                    
        return new_coup,new_trend


def load_CMR(opt='data'):
    if opt=='data':
        ds=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/vlm/Frederikse_2020/data/total.nc')
        lon=ds.lon.values
        lon[lon>180]=lon[lon>180]-360
        ds=ds.assign_coords(lon=lon)
        CMR=ds
        time_array=pd.date_range(start=str(CMR.time.values.min()),end=str(CMR.time.values.max()+1), freq='Y')
        CMR['time']=time_array
        data=CMR
        domain1=  ((data.coords["lon"] > 0) )
        domain2=  ((data.coords["lon"] <= 0) )

        datagt=data.where(domain1,drop=True)
        datasm=data.where(domain2,drop=True)
        CMR=xr.concat([datasm,datagt],dim='lon')
    elif opt=='trend':
        CMR=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/vlm/Frederikse_2020/data/CMR_trend_1995_2019.nc')
    return CMR    

def load_gps(kind='ULR'):
    """
    load different GPS data
    
    ULR is the ITRF2014
    
    """
    
    if kind=='ULR':
        GPS=pd.read_csv('/home/oelsmann/Julius/Scripts/vlad/data/VLAD/WP1/GNSS_VLM/vertical_velocities_table-2.txt',skiprows=13,header=None,delim_whitespace=True,names=['Site','DOMES','Lon','Lat','T_GPS','Data','V_GPS','S_GPS','MODEL'])


        data=GPS['V_GPS']
        data2=GPS['S_GPS']
        lonnew=GPS['Lon']
        latnew=GPS['Lat']
        
        ds = xr.Dataset({'trend': (['x'],  data),'trend_un': (['x'],  data2)},
                             coords={'lon': (['x'], lonnew),
                            'lat': (['x'], latnew)})         
        
    elif kind=='NGL':
        ds=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/vlm/Blewitt_2016/Blewitt_NGL_MIDAS.nc')
        lon=ds.lon.values
        lon[lon>180]=lon[lon>180]-360
        ds.assign_coords(lon=lon)
        
    
    return ds




def make_gps(version='Lahtinen'):
    """
    here we take the GPS solution from
    
    Lahtinen, S., Jivall, L., Häkli, P. et al. Densification of the ITRF2014 
    position and velocity solution in the Nordic and Baltic countries. 
    GPS Solut 23, 95 (2019). https://doi.org/10.1007/s10291-019-0886-3
    

    """
    if version=='Lahtinen':
        GNSS=pd.read_csv('/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/gps/GNSS_Lahtinen_2019.csv',sep=';')

        ds = xr.Dataset({'trend': (['x'],  GNSS['vu'].values),'trend_un': (['x'],  GNSS['su'].values)},
                         coords={'lon': (['x'],GNSS['lon'].values ),
                        'lat': (['x'],GNSS['lat'].values )}) 

        ds.attrs={'GNSS solution': 'ITRF2014', 'link': 'https://doi.org/10.1007/s10291-019-0886-3',
                                'cite' : 'Lahtinen, S., Jivall, L., Häkli, P. et al. Densification of the ITRF2014 position and velocity solution in the Nordic and Baltic countries. GPS Solut 23, 95 (2019). https://doi.org/10.1007/s10291-019-0886-3'}
        ds['trend_un'].attrs={'standard_name': 'trend_un', 'long_name': 'trend uncertainties WN + FN','_FillValue' : 'nan', 'units': 'mm/year','info': 'Hector'}
        ds['trend'].attrs={'standard_name': 'trend', 'long_name': 'linear trend (OLS)' ,'_FillValue' : 'nan', 'units': 'mm/year'}    



        GIA=pd.read_csv('/home/oelsmann/Julius/Scripts/north_sea/data/vlm/Blewitt_2016/midas.IGS14.txt',names=['ID',
                                                                                                           'version','first_epoch','last_epoch','duration',
           'number of epochs of data','# of good data',
           '# velocity sample pairs','east','north','up',
           'east_un','north_un','up_un','east_off','north_off','up_off',
           'east_frac_outl','north_frac_outl','up_frac_outl',
           'east_std_vel_pairs','north_std_vel_pairs','up_std_vel_pairs','# steps assumed'],
                    header=None,delim_whitespace=True)
    elif version=='NGL':
        Coord=pd.read_csv('/home/oelsmann/Julius/Scripts/north_sea/data/vlm/Blewitt_2016/llh.out',
                          names=['ID','lat','lon','height'],
                        header=None,delim_whitespace=True)


        GIA.sort_values(by=['ID'],inplace=True)
        Coord.sort_values(by=['ID'],inplace=True)
        ARR=np.isin(Coord['ID'],GIA['ID'])
        Coord_cut=Coord[ARR]
        ARR2=np.isin(GIA['ID'],Coord_cut['ID'])
        GIA=GIA[ARR2]

        LAT=Coord_cut['lat']
        LON=Coord_cut['lon']

        LON=LON+360.
        #LON=-LON

        ds = xr.Dataset({'trend': (['x'],  GIA['up'].values*1000.),
                         'trend_un': (['x'],  GIA['up_un'].values*1000.),
                         'first_epoch': (['x'],  GIA['first_epoch'].values),
                         'last_epoch': (['x'],  GIA['last_epoch'].values),
                         'duration': (['x'],  GIA['duration'].values)},

                         coords={'lon': (['x'],LON.values ),
                        'lat': (['x'],LAT.values )}) 

        ds.attrs={'data info' : 'NGL MIDAS trends', 'link': 'http://geodesy.unr.edu/',
                                'cite' : 'Caron, L., & Ivins, E. R. (2019). "A baseline Antarctic GIA correction for space gravimetry". Earth and Planetary Science Letters, 115957. doi: 10.1016/j.epsl.2019.115957'}
        ds['trend_un'].attrs={'standard_name': 'trend_un', 'long_name': 'trend uncertainties (GPS-NGL)','_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        ds['trend'].attrs={'standard_name': 'trend', 'long_name': 'linear trend (GPS-NGL)' ,'_FillValue' : 'nan', 'units': 'mm/year'}  

        ds['first_epoch'].attrs={'standard_name': 'first_epoch', 'long_name': 'time series first epoch, in decimal year format.' ,'info': '(See http://geodesy.unr.edu/NGLStationPages/decyr.txt for translation to YYMMMDD format)','_FillValue' : 'nan', 'units': 'decimalyearfromat'} 
        ds['last_epoch'].attrs={'standard_name': 'last_epoch',   'long_name': 'time series last epoch, in decimal year format' ,'info': '(See http://geodesy.unr.edu/NGLStationPages/decyr.txt for translation to YYMMMDD format)','_FillValue' : 'nan', 'units': 'decimalyearfromat'} 
        ds['duration'].attrs={'standard_name': 'duration',       'long_name': 'time series duration (years)' ,'_FillValue' : 'nan', 'units': 'years'} 

        ds

        ds.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/data/vlm/Blewitt_2016/Blewitt_NGL_MIDAS.nc')

    
    return ds

    
def make_GIA(opt='NGK2016'):
    """
    make different GIA sets
    
    """
    
    if opt=='NGK2016':
        GIA=pd.read_csv('/home/oelsmann/data/projects/balticplus/dataset_lfs/GIA_models/NKG2016/NKG2016LU_abs.dat',
                        delim_whitespace=True,names=['lat','lon','trend'])    
        GIA_un=pd.read_csv('/home/oelsmann/data/projects/balticplus/dataset_lfs/GIA_models/NKG2016/NKG2016LU_StdUnc.dat',
                        delim_whitespace=True,names=['lat','lon','trend_un']) 


        ds = xr.Dataset({'trend': (['x'],  GIA['trend'].values),'trend_un': (['x'],  GIA_un['trend_un'].values)},
                         coords={'lon': (['x'],GIA['lon'].values ),
                        'lat': (['x'],GIA['lat'].values )}) 

        ds.attrs={'GIA + GNSS solution': 'ITRF2008', 'link': 'https://www.lantmateriet.se/sv/Kartor-och-geografisk-information/gps-geodesi-och-swepos/Referenssystem/Landhojning/',
                                'cite' : 'Vestøl, O., Ågren, J., Steffen, H. et al. NKG2016LU: a new land uplift model for Fennoscandia and the Baltic Region. J Geod 93, 1759–1779 (2019). https://doi.org/10.1007/s00190-019-01280-8'}
        ds['trend_un'].attrs={'standard_name': 'trend_un', 'long_name': 'trend uncertainties','_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        ds['trend'].attrs={'standard_name': 'trend', 'long_name': 'linear trend' ,'_FillValue' : 'nan', 'units': 'mm/year'}  

    elif opt=='ICE-6GD':
        #from http://www.atmosp.physics.utoronto.ca/~peltier/data.php
        GIA = xr.open_dataset('/home/oelsmann/Julius/data/aux/GIA/ICE/drad.12mgrid_512.nc')
        GIA=GIA.rename({'Lat':'lat','Lon':'lon'})
        GIA=GIA.rename({'Drad_250':'trend'})
        GIA['trend_un']=GIA['trend']*0
        
        lon=GIA.lon.values
        lon[lon>180]=lon[lon>180]-360
        GIA['lon']=lon
        ds=GIA
        
    elif opt=='Caron':        
        GIA=pd.read_csv('/home/oelsmann/Julius/Scripts/north_sea/data/vlm/Caron_2018/GIA_maps_Caron_et_al_2018',
                    skiprows = 7,header=None,delim_whitespace=True,names=['Latitude','Longitude','VLM (mm/yr) exp.',
                                                                          'VLM (mm/yr) std.','Geoid rate (mm/yr) exp.',
                                                                          'Geoid rate (mm/yr) std.',
                                                                          'Gravity rate (mm/yr WHE) exp.',
                                                                          'Gravity rate (mm/yr WHE) std.']
        )


        LAT=GIA['Latitude']
        LON=GIA['Longitude']

        LON[LON>180.]=LON[LON>180.]-360.
        LAT=(LAT-90.)*-1


        ds = xr.Dataset({'trend': (['x'],  GIA['VLM (mm/yr) exp.'].values),
                         'trend_un': (['x'],  GIA['VLM (mm/yr) std.'].values),
                         'geoid_rate': (['x'],  GIA['Geoid rate (mm/yr) exp.'].values),
                         'geoid_rate_un': (['x'],  GIA['Geoid rate (mm/yr) std.'].values),
                         'gravity_rate': (['x'],  GIA['Gravity rate (mm/yr WHE) exp.'].values),
                         'gravity_rate_un': (['x'],  GIA['Gravity rate (mm/yr WHE) std.'].values)                 

                        },
                         coords={'lon': (['x'],LON.values ),
                        'lat': (['x'],LAT.values )}) 

        ds.attrs={'data info' : 'GIA + GNSS + GRACE + Bayes', 'link': 'https://vesl.jpl.nasa.gov/solid-earth/gia/',
                                'cite' : 'Caron, L., & Ivins, E. R. (2019). "A baseline Antarctic GIA correction for space gravimetry". Earth and Planetary Science Letters, 115957. doi: 10.1016/j.epsl.2019.115957'}
        ds['trend_un'].attrs={'standard_name': 'trend_un', 'long_name': 'trend uncertainties (GIA, Caron2018)','_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        ds['trend'].attrs={'standard_name': 'trend', 'long_name': 'linear trend (GIA, Caron2018)' ,'_FillValue' : 'nan', 'units': 'mm/year'}  

        ds['geoid_rate'].attrs={'standard_name': 'geoid_rate',           'long_name': 'Geoid rate (mm/yr) exp.' ,'_FillValue' : 'nan', 'units': 'mm/year'} 
        ds['geoid_rate_un'].attrs={'standard_name': 'geoid_rate_un',     'long_name': 'Geoid rate (mm/yr) std.' ,'_FillValue' : 'nan', 'units': 'mm/year'} 
        ds['gravity_rate'].attrs={'standard_name': 'gravity_rate',       'long_name': 'Gravity rate (mm/yr WHE) exp.' ,'_FillValue' : 'nan', 'units': 'mm/yr WHE'} 
        ds['gravity_rate_un'].attrs={'standard_name': 'gravity_rate_un', 'long_name': 'Gravity rate (mm/yr WHE) std.' ,'_FillValue' : 'nan', 'units': 'mm/yr WHE'} 

        #ds

        #ds.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/data/vlm/Caron_2018/GIA_maps_Caron_et_al_2018.nc')

    return ds




def set_settings():
    """
    set global options
    structure motivated by Frederikse et al. 2020
    """
   
    print('Define settings...')
    global settings2
    settings2 = {}
    # filedir.
    
    dirname='/home/oelsmann/Julius/Scripts/vlad_globcoast/scripts'
    settings2['dir_sat_tg']    =  os.path.join(dirname, '../vlm_sat_tg/')
    settings2['dir_gps']       =  os.path.join(dirname, '../vlm_gps/')
    settings2['dir_gia']       =  os.path.join(dirname, '../vlm_gia/')
    settings2['dir_gdr']       =  os.path.join(dirname, '../vlm_gdr/')
    settings2['dir_external']  =  os.path.join(dirname, '../external/')    
    return


def load_sat_tg(kind='ales_gesla_rmse',var='stats',lvl='all'):
    """
    load sat-tg vlm trend datasets
    """
    set_settings()
    if var=='stats':
        add_= 'zoi_'    # load trend + stats
    else:
        add_= 'timser_' # load monthly time-series

    if kind=='ales_gesla_corr':
        ds=xr.open_dataset(settings2['dir_sat_tg']+'ales_gesla/ales_gesla_'+add_+'corr.nc')
    if kind=='ales_gesla_rmse':
        ds=xr.open_dataset(settings2['dir_sat_tg']+'ales_gesla/ales_gesla_'+add_+'rmse.nc')
    if kind=='ales_psmsl_corr':
        ds=xr.open_dataset(settings2['dir_sat_tg']+'ales_psmsl/ales_psmsl_'+add_+'corr.nc')
    if kind=='ales_psmsl_rmse':
        ds=xr.open_dataset(settings2['dir_sat_tg']+'ales_psmsl/ales_psmsl_'+add_+'rmse.nc')
    if lvl=='all':
        return ds
    else:
        ds=ds.where(ds.level==lvl).dropna(dim='x',how='all')
    if var is not 'stats':
        ds=ds*-1.
    return ds