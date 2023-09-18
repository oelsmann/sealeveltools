#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 16:15:59 2019

@author: oelsmann
"""


import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit
from sealeveltools.sl_stats.resources.hector.analyse_timeseries_function import *
from sealeveltools.sl_stats.sl_stats import *
from sealeveltools.sl_stats.sl_zoi import *
from sealeveltools.sl_stats.sl_kendall import *
from sealeveltools.sl_plots import *
from sealeveltools.sl_misc import *
from eofs.xarray import Eof
from pprint import pprint
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning)

class slt_object:
    """
    - parent class of sl()    
    - accepts xr, pd or float objects
    - extracts necessary information for statistical tools to be applied (sl)
    
    information on data-set
    helps to understand and pack the data to be used by sl()
    attributes:            
        data: input data
    properties:
        typ: data-type
        shape: data-shape
        flat_x: data-values (observations)
        flat_y: pandas.Datetime time-vector (if existing)
        flat_numeric_y: numeric time-vector (if existing)

    
    """
    def norm_pd(self,convert=''):
        """
        find and set a datetime index or reject
        
        """
        if convert=='xr':
            print('convert to pd.DataFrame')
            self.data=pd.DataFrame(self.data.values,index=self.data.time.values,columns=[self.data.name])
            return self.data
        
        else:
            if str(type(self.data.index))== "<class 'pandas.core.indexes.datetimes.DatetimeIndex'>":  
                # convert series to DataFrame
                if str(type(self.data)) == "<class 'pandas.core.series.Series'>":
                    print('convert to DataFrame')
                    self.data=pd.DataFrame(self.data,columns=[self.data.name])       
                return self.data
            elif len(self.shape) == 1:
                raise TypeError(f" data has no defined timesteps ")                
            else:
                # search for time vector in data
                print('update time-index')
                jj=0
                for i in self.data.iloc[0,:].values:

                    if str(type(i)) == "<class 'str'>" or str(type(i)) == "<class 'pandas._libs.tslibs.timestamps.Timestamp'>":
                        index=pd.to_datetime(self.data.iloc[:,jj].values)
                        _values=self.data.values[:,~jj]                    
                        self.data=pd.Series(_values,index=index) 
                        break
                    jj=jj+1   

                return self.data
        
      
    def __init__(self,*args,convert_pd=False):   
        
        if args==():
            data=1.
        else:
            if str(type(args[0]))=="<class 'sealeveltools.sl_class.sl'>":
                self.data=args[0].data 
            else:
                data=args[0]        
                self.data = data

                if self.typ=='pd':
                    self.data=self.norm_pd()
                if (self.typ=='xra' or self.typ=='xr') and 'time' in self.data.dims and len(self.shape)==1 and len(self.data.dims)==1 and convert_pd:
                    # when it's only one timeseries, without coordinates
                    self.data=self.norm_pd(convert='xr')
                
    def __call__(self,*args,**kwargs):
        """
        here the functionality of pandas is added:
        if sl receives an unknown function call
        __getattrs__ assigns the associated pandas function to self.data
        the method self.data is then equal to self.<function>
        this function can be called with pandas arguments:
        self.<function>(*args,**kwargs)
        when the __call__ is invoked
        self.data is then updated with 
        self.data=self.<function>(*args,**kwargs)
        and the original sl class is returned
        
        
        """
        
        #for ar in *args:
        #    if 
        
        self.data=self.data(*args,**kwargs)
        return self    
        
    def __repr__(self):
        return repr(self.data)
    
    
    # functions
    
    def flat(self,kind='values',style='np'):
        """
        returns flattened array of 
        
        Parameters
        
        ------------
        
        kind: values, lon, lat
        
        
        """      
        if self.typ == 'xra' or self.typ == 'xr':      
            if self.typ == 'xra':
                dat=self.data[self.var]
            elif self.typ == 'xr':
                dat=self.data
            if len(self.shape) > 2:
       
                if kind=='values':
                    flat_x= getattr(dat,kind).reshape(self.shape[0],np.prod(self.shape[1:]))
                else:
                    flat_x= getattr(dat,kind).values  

                    if len(flat_x.shape) > 1:
                  
             
                        flat_x=flat_x.flatten()     
                
                    else:
                        lon, lat = np.meshgrid(dat.lon.values, dat.lat.values)
                        
                        if kind=='lon':
                            flat_x=lon.flatten()
                        elif kind=='lat':
                            flat_x=lat.flatten()
            elif len(self.shape) == 2 and 'time' not in self.data.dims:     
                if kind=='values':
                    flat_x= getattr(dat,kind).flatten()                   
                else:
                    
                    if 'lat' in dat.dims:
                        lon, lat = np.meshgrid(dat.lon.values, dat.lat.values)
                        
                        if kind=='lon':
                            flat_x=lon.flatten()
                        elif kind=='lat':
                            flat_x=lat.flatten()                                      
                    else:
                        flat_x= getattr(dat,kind).values.flatten()  
                
                
            else:     
                if kind=='values':
                    flat_x= getattr(dat,kind)
                else:
                    flat_x= getattr(dat,kind).values


        elif self.typ == 'pd':
            if kind=='values':
                flat_x= getattr(self.data[self.var],kind)
            else:
                flat_x= getattr(self.data[self.var],kind).values
                
        elif self.typ == 'pd_multi':  
            # data structured like:
            # mission lat lon data
            #
            #
            if kind=='values':
                flat_x= getattr(self.data[self.var],kind).flatten()
            else:
                if kind=='lat':                    
                    a=pd.DataFrame(self.data.index.get_level_values(1))
                elif kind=='lon':
                    a=pd.DataFrame(self.data.index.get_level_values(2))
                elif kind=='mission':  
                    a=pd.DataFrame(self.data.index.get_level_values(0))
                if style=='np':
                    flat_x=a.loc[(a.shift() != a).values.flatten()].values.flatten()
                else:
                    flat_x=a.loc[(a.shift() != a).values.flatten()]                    

                            
            
            
        elif self.typ =='dask':
            if kind=='values':
                flat_x= self.data.iloc[:,-1] #last row of dask array
            else:
                flat_x= getattr(self.data,kind)          
        
        else:
             flat_x= self.data
        return flat_x
    
    
    # properties
    
    @property
    def typ(self):
        """
        xr, pandas, float
        """
        
        str_typ = str(type(self.data))

        if str_typ=="<class 'pandas.core.frame.DataFrame'>":
            if isinstance(self.data.index, pd.MultiIndex):
                typ='pd_multi'
            else:
                typ='pd'                
            
        elif str_typ=="<class 'pandas.core.series.Series'>":
            typ='pd'                
        elif str_typ=="<class 'float'>" or str_typ=="<class 'numpy.float64'>":
            typ='float'
        elif str_typ=='xarray.core.dataset.Dataset':
            typ='xra'
        elif str_typ== "<class 'xarray.core.dataarray.DataArray'>":
            typ='xr'
        elif str_typ== "<class 'dask.dataframe.core.DataFrame'>":
            typ='dask'    
        elif str_typ== "<class 'list'>":
            typ='list'
        else:
            raise TypeError(f" Wrong data-type ")  
        return typ     
    
    @property    
    def var(self):
        if self.typ == 'xra':
            return list(self.data.keys())[0]
        elif self.typ == 'xr':
            return self.data.name
        elif self.typ == 'pd':    
            return self.data.columns[0]
        else:
            return ''
    
    
    @property
    def shape(self):
        if self.typ == 'xra':
            return self.data[self.var].values.shape            
        elif self.typ == 'xr':
            return self.data.values.shape
        elif self.typ == 'pd' or self.typ == 'pd_multi':
            return self.data.shape
        elif self.typ == 'dask':
            return (self.data.shape[0].compute(),self.data.shape[1].compute())
        else:
            return ()        
        
        #return get_shape(self)   
    @property
    def flat_x(self):
        return self.flat()      

        
    @property
    def flat_y(self):
        if self.typ == 'xr' or self.typ == 'xra':
            if 'time' in self.data.dims:
                timeser = self.data.time
            else: 
                timeser=np.array([0])

        elif self.typ == 'pd':
            timeser =self.data.index
        else:
            timeser = np.array([0])
        if len(timeser) > 1:                
            timeser=pd.to_datetime(timeser.values)    
        return timeser
        #return self.__flat_y
    @property
    def flat_y_numeric(self):
        year=pd.Timedelta('365.2422 days')
        f=year.total_seconds()   
        if len(self.flat_y)>1:
            return (self.flat_y-pd.to_datetime('1990-01-01')).total_seconds().values/f
        else:
            return (pd.to_datetime('2000-01-01')-pd.to_datetime('1990-01-01')).total_seconds()/f                

        #return self.__flat_y_numeric     

            

class sl(slt_object):
    """
    sl() class accepts xarray and pandas like structures
    - inherits attributes from sl_obj
    - applies operations
    
    Squeeze 1 dimensional axis objects into scalars.
    Series or DataFrames with a single element are squeezed to a scalar.
    DataFrames with a single column or a single row are squeezed to a
    Series. Otherwise the object is unchanged.
    This method is most useful when you don't know if your
    object is a Series or DataFrame, but you do know it has just a single
    column. In that case you can safely call `squeeze` to ensure you have a
    Series.
    Parameters
    ----------
    axis : {0 or 'index', 1 or 'columns', None}, default None
        A specific axis to squeeze. By default, all length-1 axes are
        squeezed.
    Returns
    -------
    DataFrame, Series, or scalar
        The projection after squeezing `axis` or all the axes.
    See Also
    --------
    Series.iloc : Integer-location based indexing for selecting scalars.
    DataFrame.iloc : Integer-location based indexing for selecting Series.
    Series.to_frame : Inverse of DataFrame.squeeze for a
    single-column DataFrame.
    """  
    
    #general functions
    def info(self):
        #lists doc and functions
        print(inspect.getdoc(self))
        funct=inspect.getmembers(self, inspect.ismethod)
        print()
        print('use functions:')
        print()
        print(np.asarray(funct[1:])[0:,0])
   
    
    # plot
    
    def plot(self,vector=False,return_fig=False,**kwargs):   
        """
        Parameters
        ----------
        var : 
        
        ranges :
        
        Returns
        -------

        --------
        """
        
        if not 'var' in kwargs:
            kwargs['var']=[self.var]
            if self.typ=='xr':
                kwargs['var']=['no_var']  
        kwargs['vector']=vector        
        if return_fig:
            return plt_xr_map(self.data,**kwargs)
        else:
            plt_xr_map(self.data,**kwargs)
    # normalize data-types
           
    def multi_function(self,*args,**kwargs):
        """
        apply function recursively for all variables in xr-Dataset
        """
        
        variables=list(self.data.keys())
        all_=[]
        for var in variables:
            new=self.data[var]
            new_out=getattr(sl(new), func)(*args, **kwargs)
            all_.append(new_out.data)
        all_=xr.merge(all_,compat='override')
 
        return sl(all_)
          
        
    def norm_data(self,data2):
        """
        normalize data so that
        1. pandas operations can be applied        
        """
        
        if data2.typ=='float':
            return data2
        
        elif data2.typ=='pd':
            data2.data=self.data*0+data2*(self.data*0+1).values[0,:,:,np.newaxis]
        
        else:
            data2.data=self.data*0+data2.data
        
        return data2
    
    # add pandas and xr functions
    
    def apply_pd_func(self,func):
                   
        return getattr(self.data, func)#(*args, **kw)  

    def __getitem__(self,key):
        """
        make sl class subscriptable
        """
        return getattr(self.data,key)
    
    
    def __getattr__(self, func):
        
        """
        hands over pd.methods to be applied on self.data
        
        """
        
        if func=='_ipython_canary_method_should_not_exist_':
            return 'test'
        
        elif self.typ is not 'float':
            self.data=self.apply_pd_func(func)
            #self.data=self.apply_pd_func(func)
            return self#self
        else:
            print('Operand does not work for floats')
            return self
      
        
    # maths and stats
    def __add__(self, other,**kwargs):
        """
        simply add data
        """
        
        if other.var=='trend' and 'time' in  self.data.dims:
            # add trend to data
            print('add trend')
            out=f_linear(self.flat_y_numeric,other.data.values[:,np.newaxis])            
            new=self.data+np.swapaxes(out,0,1) 
            return sl(new)
        else:
            self,other =self.couple(other,**kwargs) 
            return sl(self.data + other.data)
        
    def __sub__(self, other,**kwargs):
        """
        simply sub data
        """
        
        self,other =self.couple(other,**kwargs)        
        return sl(self.data - other.data)    
        
    def __mul__(self, other,**kwargs):
        """
        simply mul data
        """
        self,other =self.couple(other,**kwargs) 
        return sl(self.data * other.data)   
        
    def __truediv__(self, other,**kwargs):
        """
        simply mul data
        """
        self,other =self.couple(other,**kwargs) 
        return sl(self.data / other.data)   
        
                
    def rms(self,data2,s_method='nearest',limit=100.,t_method='time',dropna=True,**kwargs):
        """
        compute rms between datasets
        
        """

        data2=sl(data2)
        if data2.typ =='pd' or data2.typ =='xr':    
            if data2.typ =='pd':
                #corr_array=pd.DataFrame(self.flat_x,index=self.flat_y).corrwith(data2.data)
                raise TypeError(f" Data-type wrong or not integrated ")
            elif data2.typ=='xr':

                self,data2=self.match_xr(data2,s_method=s_method,
                                         limit=limit,t_method=t_method,**kwargs) 
                                         # aranges data such that self is the 
                                         # class with the least coordinates

                one=pd.DataFrame(self.flat_x,index=self.flat_y)
                two=pd.DataFrame(data2.flat_x,index=data2.flat_y)
                
                two=two-two.mean()
                one=one-one.mean()                           
                rms_array=np.sqrt(np.mean(((two-one)**2),axis=0))
                
            data_rms=rms_array.values.reshape(self.shape[1:])
            self.data[0,...]=0
            new_data=self.data[0,...]+data_rms
            new_data.attrs={'standard_name': 'rmse', 'long_name': self.var+' rmse. '+data2.var,
                            '_FillValue' : 'nan', 'units': '','info': ''}
            if dropna:
                
                for dim in new_data.dims:
                    new_data=new_data.dropna(dim=dim)

            self.data=new_data   

        elif data2.typ=='xra':
            
            print('loop through function')
            sl_list=[]
            for var in list(self.data.keys())[:]:
                sl_list.append(sl(self.data[var]).cor(data2).data)
            
        else:
            raise TypeError(f" Data-type wrong or not integrated ")
        return self        
        
        
        return 
        
    def regres():
        """
        regression of two variables
        
        """
        
        
    def cor(self,data2,s_method='nearest',limit=100.,t_method='time',dropna=True,**kwargs):
        """
        correlate two datasets
        
        
        
        """
        
        
        data2=sl(data2)
        if data2.typ =='pd' or data2.typ =='xr': # correlate with pd time series   
            if data2.typ =='pd': 
                if len(data2.data.columns) > 1:  # pd has more than one column
                    print('correlate multiple columns')
                    all_=[]
                    keys_=[]
                    i=0   
                    data3=copy.deepcopy(data2.data)
                    self_copy=copy.deepcopy(self.data)
                    for var2 in data2.data.columns:
                        self_copy2=copy.deepcopy(self_copy)
                        dat=data2.data[var2]
                        new=pd.DataFrame()
                        new[var2]=dat
                        other=sl(new)
                        print(var2)
                        
                        
                        out_self=sl(self_copy2).cor(other,limit=100.,t_method='time',dropna=dropna,**kwargs).data
                        if self.var==None:
                            var_text='variable1'
                        else:
                            var_text=self.var   
                        if self.typ == 'pd':
                        
                            out_self=out_self.rename(var_text+'_cor_'+var2)
                            keys_.append(var_text+'_cor_'+var2)
                        else:
                            out_self.attrs={'standard_name': 'correlation', 'long_name': var_text+' cor. '+var2,
                                '_FillValue' : 'nan', 'units': '','info': ''}
                            out_self=out_self.rename(var_text+'_cor_'+var2)
                        all_.append(out_self)
                        i=i+1
                    if self.typ == 'pd':    
                        dataset=pd.concat(all_,keys=keys_)#,keys=['sh','rg','hh'])
                    else:
                        dataset=xr.merge(all_,compat='override') 
                    
                    return sl(dataset)

                    #i=0
                    #for var in list(dataset.keys()):
                    #    dataset[var].attrs=all_[i].attrs  
                    #    i=i+1
                    #self.data=dataset                                       
                else:
                    
                                                                                                                                                    corr_array=pd.DataFrame(self.flat_x,index=self.flat_y).corrwith(pd.Series(data2.data.values.flatten(),index=data2.data.index))
                    
                    

            elif data2.typ=='xr':

                self,data2=self.match_xr(data2,s_method=s_method,
                                         limit=limit,t_method=t_method,**kwargs) 
                                         # aranges data such that self is the 
                                         # class with the least coordinates

                one=pd.DataFrame(self.flat_x,index=self.flat_y)
                two=pd.DataFrame(data2.flat_x,index=data2.flat_y)
                corr_array=one.corrwith(two)
            if self.typ =='pd':
                print(corr_array)
                print(type(corr_array))
                new_data=corr_array  
            else:

                data_corr=corr_array.values[0:np.prod(self.shape[1:])].reshape(self.shape[1:])
                self.data[0,...]=0         
                new_data=self.data[0,...]+data_corr 
                if self.var==None:
                    var_text='variable1'
                else:
                    var_text=self.var             
                new_data.attrs={'standard_name': 'correlation', 'long_name': var_text+' cor. '+data2.var,
                                '_FillValue' : 'nan', 'units': '','info': ''}
                if dropna:

                    for dim in new_data.dims:
                        new_data=new_data.dropna(dim=dim)

            self.data=new_data   

        elif data2.typ=='xra':
            
            print('loop through function')
            sl_list=[]
            for var in list(self.data.keys())[:]:
                sl_list.append(sl(self.data[var]).cor(data2).data)
                
            #list(self.data.keys())[0]
            
        else:
            raise TypeError(f" Data-type wrong or not integrated ")

        # reshape    
               
        print(self.data.shape)
        return self#.squeeze(dim='time')    
    
    def zoi(self,other,output='zoi_series',opt='cor',how='rel',level=0.8,avg='mean',limit=100.,both=False):
        """
        a) maps two datasets:
            1. self:  usually satellite data
            2. other: tide gauge data
        
        b) computes statistics within a *limit radius around tg
        
            opt: median, or mean within the limit
                 corr, rms, AC absolute or relative thresholds
                 
        c) selects subsets of statistics (abs/rel thresholds)
        
        d) makes average over best performing timeseries
        
        
        Parameters
        ----------
        other: other sl observation dataset
        
        opt: 'cor','rms' or 'AC' - criterium on which ZOI selection is based
        
        how: 'rel' or 'abs' - relative or absolute levels
        
        level: percentile, or absolute value
        
        avg: make 'mean' or 'median' of data in ZOI
        
        output: 'zoi_series' returns the averaged series formed in a ZOI; 'zoi_stats'
                returns statistics and individual series which satisfy the criteria set for the ZOI
        Returns
        -------

        --------
        """        
        
        #if opt=='median' or opt=='mean':
        #elif opt=='cor':
        
        # this function searches for values in radius = limit
        # both datasets are organized such that they cover the same coordinates
        # i.e. TG time series are projected/multiplied on all values of SAT in ZOI
        self,other = self.remap(other,both=True,s_method='zoi',limit=limit)
        
        
        
        self_or=copy.deepcopy(self.data)
        if opt=='out':
            self_cor=self        
        else:
            self_cor=getattr(self, opt)(other,dropna=False) # e.g. opt='cor' or 'rms'
        if output=='zoi_stats':
            if both:
                return self_cor,other            
            else:
                return self_cor        
        
        elif output=='zoi_series':
            out=self_cor.data.groupby('idx').apply(lambda x: zoi_function(x,kind=opt,how=how,level=level,merge=''))
            # check e.g. where the best correlated data are

            if avg=='mean':
                data=self_or.loc[:,np.isnan(out)].groupby('idx').mean(dim='x') # mean along x
            elif avg=='median':
                data=self_or.loc[:,np.isnan(out)].groupby('idx').median(dim='x') # mean along x
            lon=other.data.lon.groupby('idx').mean(dim='x')
            lat=other.data.lat.groupby('idx').mean(dim='x')
            data=data.assign_coords(lon=lon,lat=lat)
            data
            self.data=data
            return self

        
        
        
        

    # statistics
    

        
    def fltr(self,how='std',mod1=2,mod2='',win_size='30D',win_type=None):
        
        """
        different filter
        includes here also oulier rejection
        """
        if how =='std':

            other=copy.deepcopy(self.data)
            if mod2=='detrend':
                self=self.detrend()

            
            tt=abs(self.data-self.data.mean(dim='time'))<mod1*self.data.std(dim='time')

            tt=xr.where(tt, other, np.nan)

            self.data=tt
            
        if how =='abs_std':
            # scale by absolute standard-dev = mean std of set
            other=copy.deepcopy(self.data)
            if mod2=='detrend':
                self=self.detrend()

            abs_std=self.data.std(dim='time').mean()
            tt=abs(self.data-self.data.median(dim='time'))<mod1*abs_std

            tt=xr.where(tt, other, np.nan)

            self.data=tt
            
        elif how=='MAD':
            #median absolute deviation threshold
            other=copy.deepcopy(self.data)
            if mod2=='detrend':
                self=self.detrend()

            
            tt=abs(self.data-self.data.median(dim='time'))<mod1*1.4826*abs(self.data-self.data.median(dim='time')).mean(dim='time')


            tt=xr.where(tt, other, np.nan)

            self.data=tt            
        elif how=='abs':
            #absolute threshold
 
            other=copy.deepcopy(self.data)
            if mod2=='detrend':
                self=self.detrend()            
            tt=abs(self.data-self.data.median(dim='time'))<mod1
            tt=xr.where(tt, other, np.nan)

            self.data=tt            
                        
            
            
        elif how=='rolling':
            kwargs={'win_size':win_size,'win_type':win_type}
            out=xr.apply_ufunc(sl_fltr_rolling, self.data,self.data.time,
                  input_core_dims=[["time"],["time"]],output_core_dims=[['var']],
                         kwargs=kwargs,dask = 'allowed', vectorize = True)
            self.data=out
        return self
    
    
    def detrend(self,monthly=False,**kwargs):
        """
        subtract trend and annual cycle
        
        **kwargs ={'de_season' : 'False','semi' : 'False'}
        """
        #kwargs.setdefault('detrend', True)        
        kwargs['dtrend']=True
        kwargs['de_season']=True
        kwargs['semi']=False
        kwargs['de_season_without_trend']=False
        kwargs['trend_only']=False
        kwargs['hector']=False
        kwargs['kendall']=False
        kwargs['hector_out']=False
        kwargs['monthly']=monthly
        kwargs['min_samples']=20.
        
        out=trend_sub(self,**kwargs) # function in sl_stats

        shp=sl(out).shape
        

        if len(shp)>2:
            detrend_arr=annual_cycle(self.flat_y_numeric,out['offset'].values[0,:,:,np.newaxis],
                     out['trend'].values[0,:,:,np.newaxis],
                     out['acos'].values[0,:,:,np.newaxis],
                     out['asin'].values[0,:,:,np.newaxis])        
            self.data=self.data-np.swapaxes(np.swapaxes(detrend_arr,0,2),1,2)
            
        elif len(shp)==2:
            detrend_arr=annual_cycle(self.flat_y_numeric,out['offset'].values[:,:,np.newaxis],
                     out['trend'].values[:,:,np.newaxis],
                     out['acos'].values[:,:,np.newaxis],
                     out['asin'].values[:,:,np.newaxis])             
            self.data=self.data-np.swapaxes(np.swapaxes(detrend_arr,0,1),0,2)
        else:
            detrend_arr=annual_cycle(self.flat_y_numeric,out['offset'].values[:,np.newaxis],
                     out['trend'].values[:,np.newaxis],
                     out['acos'].values[:,np.newaxis],
                     out['asin'].values[:,np.newaxis])             
            self.data=self.data-np.swapaxes(detrend_arr,0,1)
                  
        
        self.data.attrs={'standard_name': self.var, 'long_name': 'detrended '+self.var,'_FillValue' : 'nan'}                   
        return self  
    

    
    def trend(self,how='leastsq',de_season=True,semi=False,
                       de_season_without_trend=False,trend_only=False,min_samples=20.,kendall=False, hector=False,
                       model='AR1',hector_out=False,monthly=False):
        """
        compute trend and statistics
        
        
        Parameters
        ----------
        
        how: str, default 'leastsq',
            or 'MLE', 'kendall'
        
        de_season: subtract seasonal component (in case of de-seasoning the data)
         
        semi: compute semi-annuals cycle (sinusoidal)
        
        de_season_without_trend:  only remove annual cylce
       
        trend_only: trend fit only (no periodic components)
        
        min_samples: minimum number of required samples
        
        kendall: bool, default False
            compute trends using kendall's tau
                
        hector: bool, default False
            Turn on hector trend uncertainty computation
        
        model: WN, AR1, FNWN, RWFLWN, PLWN, GGMWN
        
        hector_out: bool, default True
                    True: compute all statistics with hector (trend, cylces, uncertainties)
                    False: only compute uncertainties with hector, and trend with linear fit
        
        monthly: if True then calculates a monthly-wise seasonal cycle, Amplitude is (min-max)/2
        
        Returns: sl() object containing trend + uncertainty estimates for var
        -------        
        
        """
        if how=='MLE':
            hector=True
            hector_out=True
        elif how=='kendall':
            kendall=True            
        kwargs={'de_season':de_season,'semi':semi,
                       'de_season_without_trend':de_season_without_trend,
                        'trend_only':trend_only,'min_samples':min_samples,'kendall':kendall,
                'hector':hector,'model':model,'hector_out':hector_out,
                        'monthly':monthly}
        
        return sl(trend_sub(self,**kwargs))  
        
    def eof(self,n_eofs=6,**kwargs):
        """
        apply function from eofs.xarray
        
        """
        solver = Eof(self.data)
        
        all_=[]
        output={}
        eof_pattern=solver.eofs()
        var=solver.varianceFraction()
        for mode in range(n_eofs):
            ds=eof_pattern.where((eof_pattern.mode==mode),drop=True).squeeze(dim='mode')
            mode=mode+1
            ds=ds.rename(str(mode)+'. eof')
            ds.attrs['long_name']=str(mode)+'. EOF ('+str(np.round(var[mode-1].values*100,2))+'% expl. var)'
            all_.append(ds)
        eof_pattern=xr.merge(all_,compat='override')
        pcs=solver.pcs(npcs=n_eofs,**kwargs)
        output['eof_pattern']=sl(eof_pattern)
        output['pcs']=pcs
        output['solver']=solver
        return output
        
    
    # time operators
    
    def selmon(self,months):
        
        months=np.asarray(months)
        out=self.data
        if self.typ == 'xr' or self.typ == 'xra':

            self.data=out[np.isin(pd.to_datetime(out.time.values).month,months),...]

        elif self.typ == 'pd':
            self.data=out[np.isin(pd.to_datetime(out.time.values).month,months)].dropna()
        else:
            '! no pandas or xarray data-type !'        
        return self         
        

    
    def yearmean(self,):
        if self.typ == 'xr' or self.typ == 'xra':
            self.data=self.data.resample(time="Y").mean()
        elif self.typ == 'pd':
            self.data=self.data.resample('Y').mean()
            self.data.columns=self.data.columns +' - annual mean'
        else:
            '! no pandas or xarray data-type !'
        return self
 
    def monmean(self,):
        if self.typ == 'xr' or self.typ == 'xra':
            self.data=self.data.resample(time="M").mean()
        elif self.typ == 'pd':
            self.data=self.data.resample('M').mean()
        else:
            '! no pandas or xarray data-type !'
        return self       
  
    def seasmean(self,seas='DJF'):
        """
        DJF belongs to year of JF
        
        """
        
        
        
        ll=['DJF','MAM','JJA','SON']
        ll2=[12,3,6,9]
        ll3=[3,6,9,12]
        
        if self.typ == 'xr' or self.typ == 'xra':
            month=ll3[ll.index(seas)]
            out=self.data.resample(time="QS-DEC").mean()
            if self.typ == 'xra': 
                out=out[self.var].shift(time=1)
            else:
                out=out.shift(time=1)
                
            time_new=out[pd.to_datetime(out.time.values).month==month,...]
            self.data=time_new.resample(time="Y").mean()
            
            
            
            
        elif self.typ == 'pd':
            month=ll2[ll.index(seas)]
            out=self.data.resample("QS-DEC").mean()
            out=out[out.index.month==month].dropna().shift(2, freq ='MS').resample('Y').mean()
            out.columns=out.columns +'-'+seas
            self.data=out
            
        else:
            '! no pandas or xarray data-type !'        
        return self   
    def yearmon(self,how='mean',op=None):
        """
        Multi-year monthly statistical values
        
        Parameters:
        --------------
        how: mean, min, max, std
        op: None (default), sub to subtract annual cycle, or amplitude 
        """
        
        out=copy.deepcopy(self.data)
        if self.typ == 'xr' or self.typ == 'xra':

            for month in np.arange(13)[1:]:
                mon_index=pd.to_datetime(out.time.values).month==month
                out[mon_index,...]=getattr(out[mon_index,...], how)(dim='time')

            if op=='sub':
                # de_season
                out=sl(baltic).data-out

            elif op=='amplitude':    
                # make amplitude
                out=(out.max(dim='time')-out.min(dim='time'))/2
        
        
        return sl(out)
        
        
    def timavg(self):
        if self.typ == 'xr' or self.typ == 'xra':
            self.data=self.data.mean(dim=['time'])
        elif self.typ == 'pd':
            self.data=self.data.mean(axis=0)
        else:
            '! no pandas or xarray data-type !'
        return self
    
 
    # remapping - re-aranging - combinations
    
    def box(self,extend=[0,50,30,66],drop=True):
        """
        lonlatbox selection
        """
        data=self.data
        domain = (
         ((data.coords["lat"] > extend[2]) &
          (data.coords["lat"] < extend[3]) &
         (data.coords["lon"] > extend[0]) &
         (data.coords["lon"] < extend[1]) )
        )
        if drop:
            self.data=data.where(domain,drop=drop)
        else:
            self.data=data.where(domain,np.nan)            
        return self
    
    
    def radial_smoothing(self,stdon=True,L_scale=150.,exponent=1,function='cos'):
        """
        smooth values in a certain radius,
        weigth by uncertainties
        
        ----------
        Parameter
        
        self: sl()-object
        
        stdon: bool, default: True,
            include uncertainty weighting
        
        L_scale: float, default: 150.,
            lenght scale of interpolation in km
        
        exponent: float, default: 1.,
            ()
        
        function: str, default: 'cos', 
            distance weigthing function
       
        """
        variable=sl(self['trend'])

        if stdon:
            # get uncertainty of variable
            if 'trend_un' in self.data:
                var_un=sl(self['trend_un'])
            elif 'trend_err' in self.data:
                var_un=sl(self['trend_err'])
            else:
                raise ValueError("Dataset must contain 'trend_un', oder 'trend_err' variables ")    
            var_un, out2=var_un.remap(var_un,both=True,s_method='zoi',limit=L_scale)
            text='err. weighting'
        else:
            var_un=None
            text='no err. weighting'
        variable, out2=variable.remap(variable,both=True,s_method='zoi',limit=L_scale)
        # use original dataset to finally overwrite it
        out2=weight_vals_radially(variable,original=out2,uncertainty=var_un,exponent=exponent,
                                 lenght_scale=L_scale,function=function)
        out2['trend'].attrs['long_name']='trend '+str(L_scale)+' km smoothing; '+text
        return out2  
    
    def grid_pd(self,grid_size):
        """
        for data arrays with or without time dimensions
        
        """
        
        lon=self.flat(kind='lon')
        lat=self.flat(kind='lat')
        flat_x=self.flat_x
        max_lat=np.max(lat)
        min_lat=np.min(lat)
        max_lon=np.max(lon)
        min_lon=np.min(lon)

        grid_x, grid_y = np.mgrid[max_lat:min_lat:-grid_size,min_lon:max_lon:grid_size]
        max_leng=np.max([np.prod(grid_x.shape),flat_x.shape[-1]])    
        # maximal size of data in flattened space-direction

        if 'time' in self.data.dims:
            time_=True
            _dum=np.empty([flat_x.shape[0],max_leng])
        else:
            time_=False
            _dum=np.empty([1,max_leng])            
        _dum[:,0:flat_x.shape[-1]]=flat_x
        
        #dummy array
        
        dat=pd.DataFrame(_dum)
        points=np.column_stack((lon,lat))
        # old coordinates

        if time_:
            dat=dat.apply(lambda x: grid_pd_loop(x,flat_x.shape[-1],grid_x, grid_y,points),axis=1)     
        else:
            dat=grid_pd_loop(dat.loc[0,:],flat_x.shape[-1],grid_x, grid_y,points) 
        
        latnew=np.arange(max_lat,min_lat,-grid_size)
        lonnew=np.arange(min_lon,max_lon,grid_size)

        if time_:
            self.data=reconstruct_data(self,dat.loc[:,0:(np.prod(grid_x.shape)-1)].values.reshape(dat.shape[0],grid_x.shape[0],grid_x.shape[1]),latnew, lonnew,time_=time_)
        else:
            self.data=reconstruct_data(self,dat.loc[0:(np.prod(grid_x.shape)-1)].values.reshape(grid_x.shape[0],grid_x.shape[1]),latnew, lonnew,time_=time_)            
        
        return self

    def couple(self,other,limit=100.,s_method='nearest',swap=True,**kwargs):
        """
        remapping function:
            - couples two dataset in time and space
        
        Parameter:
        
        self: sl() object to be coupled; self has higher resolution than other
        other: sl() objet with lower resolution; provides coordinates on which self is projected onto
        limit: max. spatial distance where data is taken into account
        s_method: space mapping method
           - nearest

             data2 has usually less space dimensions than self, 
             if not data2 and self will be swapped during matching
             to speed up the process

           - mean

             finds stations/observations in limit radii around data2!
             calcs. the mean of the surrounding observations
             here no swap so data2 should contain less measurements (e.g. data2 is 
             a tide gauge dataset, and self a gridded altimetry set)  
           - median

             same as above for median
        
        
        """
        
        other=sl(other)
        
        if other.typ != 'float':
            
            if 'time' not in other.data.dims or 'time' not in self.data.dims:
                rmptime=False 

            else:
                rmptime=True
            if self.typ=='xra': #to be fixed, add loop
                self.data=self.data[self.var]
            
            self,other=self.remap(other=other,both=True,rmptime=rmptime,grid_size=.5,s_method=s_method,
                  limit=limit,t_method='time',swap=swap,**kwargs)
            
        return self,other    
    
    def remap(self,other=None,both=False,rmptime=False,grid_size=.5,s_method='nearest',
              limit=100.,t_method='time',swap=True,stdon=False,**kwargs):
               
        """
        remap to standard cartesian lon,lat grid
        resolutions

        Parameters
        ----------
        other = None (default), 
            other sl or xr data-array as target map
            
        res = float, default 0.1 
            resolution in degrees
        
        s_method: str, default 'nearest
            remapping method: 'nearest','median','mean','zoi','sfltr'
        
        stdon: bool, default False
            include std of observation for method 'sfltr'            
            
        Returns
        -------
        sl() object       
        """

  
        if other is None:
            # regridding no coupling
            if s_method == 'sfltr':
                if self.typ == 'xr':
                    raise TypeError("input must be xarray, or pd.DataFrame with vars: trend, trend_un, lon, lat")            
                # takes also exponent=1,function='cos from **kwargs
                self.data=self.radial_smoothing(stdon=stdon,L_scale=limit,**kwargs)
            
            else:
                if self.typ == 'xra':                
                    all_=[]
                    i=0
                    for var in list(self.data.keys()):
                        dat=self.data[var]
                        other=sl(dat)
                        out0=other.grid_pd(grid_size).data
                        all_.append(out0)
                        all_[i].attrs=copy.deepcopy(out0).attrs
                        i=i+1                    
                    dataset=xr.merge(all_,compat='override') 
                    i=0
                    for var in list(dataset.keys()):
                        dataset[var].attrs=all_[i].attrs  
                        i=i+1
                    self.data=dataset
                else:
                    self=self.grid_pd(grid_size)
            return self      
        else:
            # couling no regridding
            self,_dum=self.match_xr(sl(other),s_method=s_method,
                         limit=limit,t_method=t_method,rmptime=rmptime,swap=swap,**kwargs)
            if both:
                return self.squeeze(),_dum.squeeze()
            else:
                return self.squeeze()

        
        
            
            
    def match_pd_time(self,flat_xnew,flat_y,flat_xtarget,targettime,smooth='',**kwargs):
        """
        match pd.arrays in time 
        """
        
        s_frame=pd.DataFrame(flat_xnew).set_index(flat_y)      # DataFrame which needs to be interpolated on targettime
        
        if smooth=='rolling':
            s_frame=s_frame.rolling(str(kwargs['freq'])+'D').mean().shift(-int(kwargs['freq']/2),freq='D')
        
        sapp_frame=pd.DataFrame(flat_xtarget).set_index(targettime) # nan DataFrame with targettime indices

        flat_xout=s_frame.append(sapp_frame).sort_index(axis='index').interpolate(**kwargs).loc[targettime]   
   
        return flat_xout.loc[~flat_xout.index.duplicated(keep='first')]

        
        
    
    def match_xr(self,data2,rmptime=True,swap=True,**kwargs):
        """
        match two datasets in space if rmptime=True also match in time

   
        data2:  sl().class
        kwargs: s_method='nearest',limit=100.,t_method='time'
        
        s_method: space mapping method
                   - nearest
                   
                     data2 has usually less space dimensions than self, 
                     if not data2 and self will be swapped during matching
                     to speed up the process
      
                   - mean
                     
                     finds stations/observations in limit radii around data2!
                     calcs. the mean of the surrounding observations
                     here no swap so data2 should contain less measurements (e.g. data2 is 
                     a tide gauge dataset, and self a gridded altimetry set)  
                   - median
                     
                     same as above for median
                   
               
        
        returns: self,data2
            must have the same shape i.e. same time and space dimensions
        
        """
        if data2.typ=='xra':
            data2.data=data2.data[data2.var]
        if self.typ=='pd_multi':
            self,data2=match_pd_multi_xr(self,data2,rmptime=rmptime,method=kwargs['t_method'],**kwargs)                   
            return self,data2

        else:
            if self.shape == data2.shape and kwargs['s_method'] == 'nearest':
                # don't interpolate at all
                return self,data2
            else:

                # check which has more data
                _swap=False

                sshape,st=check_spacetime_dims(self)
                dshape,dt=check_spacetime_dims(data2)


                if np.prod(sshape) < np.prod(dshape) and kwargs['s_method'] == 'nearest' and swap:

                    # swap variables 
                    # arange that data2 always has less obs. in space than self
                    # swap == True
                    # allows to swap
                    print('swap')
                    self,data2 = data2,self
                    st,dt=dt,st
                    sshape,dshape=dshape,sshape
                    _swap=True


                # dist.index = fitting index of data2
                # dist.idx = fitting index of self
                # dist.dist = distance in m

                if kwargs['s_method'] =='zoi':
                    # search for all observations in radius
                    dist=interp_pandas_like(self,data2,method='all',limit=kwargs['limit']).fillna(method='ffill')

                    dist_2_idx=dist.level.dropna().values.astype(int) # index over which data can be grouped by
                    dist_1_idx=dist.idx.values.astype(int)

                    dist_2_distance=dist.dist.dropna().values #distance of samples                
                    
                    flat_x2=data2.flat_x[...,dist_2_idx] 
                    flat_xnew=self.flat_x*np.nan
                    flat_xnew=self.flat_x[...,dist_1_idx]

                    kwargs={'limit': 2,'limit_direction': 'both','method':kwargs['t_method'],'freq':40}
                    # settings for time series interpolation
                    if 'time' in self.data.dims:
                        time_=True
                        flat_xnew=self.match_pd_time(flat_xnew,self.flat_y,flat_x2[...,dist_2_idx]*np.nan,data2.flat_y,
                                                     smooth='',**kwargs).values
                    else:
                        print('no time')
                        time_=False
                    latnew=self.flat(kind='lat')[dist_1_idx] #!falsch
                    lonnew=self.flat(kind='lon')[dist_1_idx]


                    latnewtg=data2.flat(kind='lat')[dist_2_idx] #!falsch
                    lonnewtg=data2.flat(kind='lon')[dist_2_idx]

                    # datasets will have the same shape
                
                    self.data=reconstruct_irreg_data(data2,flat_xnew,latnew,lonnew,
                                                     index=dist_2_idx,coords_in={'dist': (['x'], dist_2_distance)},
                                                     time_=time_) # give index to groupby
#
                    data2.data=reconstruct_irreg_data(data2,flat_x2,latnewtg,
                                                      lonnewtg,index=dist_2_idx,
                                                      coords_in={'dist': (['x'], dist_2_distance)})
                     
                else:

                    if kwargs['s_method'] =='nearest':
                        # map by finding tthe closest values
                        dist=interp_pandas_like(self,data2,method=kwargs['s_method'],limit=kwargs['limit']) 
                        dist_2_idx=dist.index
                        flat_x2=data2.flat_x*np.nan
                        flat_x2[...,dist.index]=data2.flat_x[...,dist.index]    
                        flat_xnew=self.flat_x[...,dist.idx.values.astype(int)]

                    elif kwargs['s_method'] =='mean' or kwargs['s_method'] =='median':
                        dist=interp_pandas_like(self,data2,method='all',limit=kwargs['limit'])
                        series_f=np.vstack(dist.groupby(level=[0]).apply(lambda x: match_xr_group(x,self,data2,
                                                                                                  kwargs['s_method']))).squeeze()

                        dist_2_idx=dist.level.dropna().values.astype(int)

                        flat_x2=data2.flat_x*np.nan
                        flat_x2[...,dist_2_idx]=data2.flat_x[...,dist_2_idx]    
                        flat_xnew=series_f

                    elif method=='cor':
                        print('not yet implemented')



                    if rmptime:
                        #also remap in time
                        kwargs={'limit': 2,'limit_direction': 'both','method':kwargs['t_method'],'freq':40}
                        flat_xnew=self.match_pd_time(flat_xnew,self.flat_y,flat_x2[...,dist.index]*np.nan,data2.flat_y,smooth='',**kwargs).values

                        flat_x1=data2.flat_x*np.nan
                        flat_x1[...,dist.index]=flat_xnew

                        self.data=from_flat_to_data(data2.data,flat_x1,data2.shape)
                        data2.data=from_flat_to_data(data2.data,flat_x2,data2.shape)

                    else:               
                        #do not remap in time

                        flat_x1=np.empty([st,data2.flat_x.shape[-1]])*np.nan
                        flat_x1[...,dist_2_idx]=flat_xnew


                        self.data=from_flat_to_data(data2.data,flat_x1,dshape,adjust_time=True,time=self.flat_y)  
                        print(kwargs['s_method'])
                        print(self.var)
                        print(str(kwargs['limit']))
         
                        
                        self.data.attrs['long_name']=kwargs['s_method']+' '+str(self.var)+' in '+str(kwargs['limit'])+' km'
                        # when mean or median self is larger file (of which median will be applied) 
                        # and data2 is the target station file
                        # 
                        data2.data=from_flat_to_data(data2.data,flat_x2,data2.shape)
                        
                        
            if _swap:

                return data2,self
            else:
                return self,data2        

    
    
    def misc(self,opt='ke_constituent'):
        """
        sl().misc (miscellaneous):
        
        Parameter:
        
        -------------------
        opt: - 'ke_constituent': coastal exposure coefficient after Lobeto at al. 2018
             - 'dist_coast': dist_coast for every grid point
             
        
        """
        
        
        if opt=='ke_constituent':
            # works only for data aranged like (time x space, i.e. 2D)
            
            lon=self.flat(kind='lon')
            lat=self.flat(kind='lat')
            land_mask=xr.open_dataset('/home/oelsmann/Julius/Scripts/vlad/data/aux/NAVO-lsmask-world8-var.dist5.5.nc')
            # needs to be added to sltools
            land_mask=land_mask['dst'].where(land_mask['dst'] == 0. ,1)
            cons=[]
            i=0
            for coord1 in np.vstack([lat,lon]).T:
                constituent=make_coastal_exp_constituent(coord1,land_mask,radius=300,x_num=360,start=5,reso=100)
                cons.append(constituent)
                i=+1
                print(i)
            self.data=self.data.assign_coords(ke_const=(self.data.dims[1], cons))
        elif opt=='dist_coast':
            """
            get distance to coast for all the corrdinates 
            use file in '/DGFI8/D/coast_distance/dist2coast_1deg_v2.grd'
            
            implemented only for xr or xra
            """
            data=xr.open_dataset('/DGFI8/D/coast_distance/dist2coast_1deg_v2.grd')

            lon=self.flat(kind='lon')
            lat=self.flat(kind='lat')

            max_lat=np.max(lat)
            min_lat=np.min(lat)
            max_lon=np.max(lon)
            min_lon=np.min(lon)

            factor=63./111. # for 2D lon lat interpolation; here for baltic sea: approximate distance ratio between 1° lon/lat

            domain = (
             ((data.coords["y"] > min_lat-1) &
              (data.coords["y"] < max_lat+1) &
             (data.coords["x"] > min_lon-1) &
             (data.coords["x"] < max_lon+1) )
            )    

            data = data.where(domain,drop=True) 

            lon_grid=data.x.values *factor# 0 - 360
            lat_grid=data.y.values 

            x=data.z.values.flatten()

            grid_x, grid_y = np.meshgrid(lon_grid,lat_grid) # grid_x = lon

            #grid_x, grid_y = np.meshgrid(lat_grid,lon_grid)
            # maximal size of data in flattened space-direction
            points=np.column_stack((grid_x.flatten(),grid_y.flatten()))

            grid = griddata(points, x, (lon*factor,lat), method='nearest').flatten()
            #ds=reconstruct_irreg_data(self,grid,lat,lon,index=[],time_=False)
            if 'time' in self.data.dims:
                grid=grid.reshape(self.data.shape[1:])  
            else:    
                grid=grid.reshape(self.data.shape[:])

            _dummy=copy.deepcopy(self.data)
            _dummy[:]=0
            new=_dummy+grid
            new=new.rename('dist_coast')
            _all=[self.data,new]
            _all=[self.data,new]
            self.data=xr.merge(_all)


            
        return self
    
    
    
class sat:
    """
    object contains time series and information for altimetry measurement
    
    """
    
    def __init__(self,mission,data,loc,dist_coast,frequency,correction_version,box_length):
        self.mission = mission
        self.loc = loc 
        self.data = data
        self.dist_coast =dist_coast
        self.frequency =frequency
        self.correction_version =correction_version
        self.box_length =box_length
        
        
        
class tidegauge:
    """
    object contains time series and information for tide gauge measurement
    
    """
    def __init__(self,name,data,loc,info,months,flag,tgindex):
        self.name = name
   
        if (flag == 0) or (flag == 2) or (flag == 3): # flag = 1: valid
            self.data = data
        else:
            self.data = np.nan              
        self.loc = loc #lat lon
        self.info = info
        self.months = months
        self.tgindex = tgindex
    def correct_xtr(self):  #exclude 999%percentiles
        self.data=gauss_percentiles(self.data)

        
        
def gauss_percentiles(data):
    
    gauss_fit = norm.fit(data.sealevel[~np.isnan(data.sealevel)])
    quant_gauss_up=norm.ppf(0.9998,gauss_fit[0],gauss_fit[1])
    quant_gauss_down=norm.ppf(0.0002,gauss_fit[0],gauss_fit[1])
    correct=data.mask(data.sealevel>quant_gauss_up)
    correct=correct.mask(correct.sealevel<quant_gauss_down)    
    return correct        
