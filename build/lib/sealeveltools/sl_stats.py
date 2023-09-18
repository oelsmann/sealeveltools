import pandas as pd
import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
import warnings
import copy
import math
from scipy import signal
from scipy.optimize import curve_fit
import xarray as xr
from sympy import *
import sympy

from scipy import stats
from sympy import *
import sympy

#-----------------------------------------
# operators to be applied on xr-data, or pandas-timeseries

# def understand_xr(ds):
    

    

# def detrend(ds,deseason=True)




# def corr
# def add
# def sub 
#
#
#
#
#

def sl_pd_series(ydata,xdata):
    """
    xdata - time-vector
    ydata - data-vecttor
    """
    
    xdata=pd.to_datetime(xdata)
    series=pd.Series(ydata, index=xdata)    
    series = series[pd.notnull(series)]    
    return series

def sl_trend_prepare(ydata,xdata,**kwargs):
    series=sl_pd_series(ydata,xdata)
    series=series[~np.isnan(series)].dropna()
  
    if len(series.dropna())<10:
        out=[np.nan]*10
    else:
        out=t_fit_annual_cycle(series,**kwargs)
        #[popta[1],Uncer_minann2[1],AMv,UncAm,y_minann_and_trend,
                #pd.Series(y_mintrend, index =series.index)]#,pd.Series(y_minann_and_trend, index =series.index)]
    
    
    return np.array(out[0:8])



def trend_sub(self,**kwargs):      
    out=xr.apply_ufunc(sl_trend_prepare, self.data,self.data.time,
                      input_core_dims=[["time"],["time"]],output_core_dims=[['var']],
                             kwargs=kwargs,dask = 'allowed', vectorize = True)

    names=['trend','trend_un','AC','AC_unc','offset','acos','asin','phase']
    long_names=['linear trend','trend uncertainties','annual cycle','annual cylce uncertainties',
                'offset','acos','asin','annual cycle phase phase']
    units=['m/year','m/year','m','m','m','m','m','rad']
    i=0
    ds_array=[]
    if self.typ=='xra':
        print('auto-select first variable ',self.var)
        out=out[self.var]
    
    
    if len(self.shape)>2:
        for name in names:
            ds_array.append(out[:,:,i].rename(name).astype(float))
            ds_array[i].attrs={'standard_name': [names[i]], 'long_name': long_names[i],'_FillValue' : 'nan', 'units': units[i]}
            i=i+1
        out1=xr.merge(ds_array,compat='override')
    else:

        for name in names:
            ds_array.append(out[:,i].rename(name).astype(float))
            ds_array[i].attrs={'standard_name': [names[i]], 'long_name': long_names[i],'_FillValue' : 'nan', 'units': units[i]}
            i=i+1
        out1=xr.merge(ds_array,compat='override') 
        
    out1=out1.assign_coords({"time": self.flat_y.mean()})
    out1=out1.expand_dims('time')      
    return out1





#-------------------------------------------
# main fit



def t_fit_annual_cycle(series,C=0.001,P=1.5,**kwargs):
    """
    fit annual cycle and trend 
    use prais-winston method
    
    Parameters:
    ------------
    series
    
    autocorrelation
    
    Returns:
    ------------
    
    popt : array
        Optimal values for the parameters so that the sum of the squared residuals of 
        f(xdata, *popt) - ydata is minimized
        
        popt[0:3] : offset, trend, ...
    
    """
    de_season=kwargs['de_season']
    semi=kwargs['semi']
    cubic=kwargs['cubic']    
    de_season_without_trend=kwargs['de_season_without_trend']
    trend_only=kwargs['trend_only']
        
    
    series=series[~np.isnan(series)]
    year=pd.Timedelta('365.2422 days')
    f=year.total_seconds()   
    xdata=(series.index-pd.to_datetime('1990-01-01')).total_seconds().values/f
    ydata=series.values
    
    
    
    if semi:
        popt, pcov = curve_fit(annual_cycle_semi, xdata, ydata)   
        y_minann=ydata-annual_cycle_semi(xdata,0,0,popt[2],popt[3],popt[4],popt[5])
        DoF=len(xdata)-6 #Degrees of Freedom N-(A,B,C,D)
    elif trend_only:
        popt=[0]*10
        pcov=[0]*10
        poptb, pcovb = curve_fit(ftrend_only, xdata, ydata)  
        popt[0:2]=poptb[0:2]
        pcov[0:2]=pcov[0:2]
        y_minann=ydata  
        DoF=len(xdata)-2 #Degrees of Freedom N-(A,B,C,D)
    else:    
        popt, pcov = curve_fit(annual_cycle, xdata, ydata)
        y_minann=ydata-annual_cycle(xdata,0,0,popt[2],popt[3])
        DoF=len(xdata)-4 #Degrees of Freedom N-(A,B,C,D)

        
    t_critical=stats.t.ppf(1-0.025,DoF) #95 % confidence interval    
    # option to start preis winston


      
        
    SE_noCO=np.sqrt(np.diag(pcov))
    Uncer_noCO=SE_noCO*t_critical 
    # subtract annual cycle

    if de_season:
        y_mintrend =ydata-annual_cycle(xdata,popt[0],popt[1],popt[2],popt[3])    

    else:
        y_mintrend =ydata-annual_cycle(xdata,popt[0],popt[1],0,0)

    if de_season_without_trend:
        y_mintrend =ydata-annual_cycle(xdata,0,0,popt[2],popt[3])    



    #plt.plot(xdata,annual_cycle(xdata,0,popt[1],0,0))
    #plt.plot(xdata,y_mintrend)

    # now estimate trend
    if cubic:
        #print len(y_minann)
        popta, pcova = curve_fit(trend_only_and_cubic, xdata, y_minann,bounds=([-np.inf,-np.inf,-np.inf,1.,-np.inf], [np.inf,np.inf,np.inf,25.,np.inf]),max_nfev=100000)       
        #popta, pcova = curve_fit(annual_cycle_semi, xdata, ydata)
        #a + b*x + c*np.cos((2*np.pi*x/d)+e)
        y_mintrend =ydata-trend_only_and_cubic(xdata,0,0,popta[2],popta[3],popta[4]) 
        #print popta
    else:        
        if trend_only:
            popta=popt
            pcova=pcovb
       
        else:
            popta, pcova = curve_fit(ftrend_only, xdata, y_minann) #popta offset + trend

    y_minann_and_trend=ydata-annual_cycle(xdata,0,popta[1],popt[2],popt[3])

    SE_a=np.sqrt(np.diag(pcova))    
    DoF1=len(xdata)-2 #two options
    Uncer_minann2=SE_a*stats.t.ppf(1-0.025,DoF1)    
    # error propagation for phase + amp
    c=Symbol('c')
    d=Symbol('d')       
    amp=sympy.sqrt(c**2+d**2)
    ph=sympy.atan(c/d)
    
    if trend_only:
        UncAm=0
        Uncph=0
        AMv=0
        PHv=0
    else:
        UncAm=error_prop(amp,[c,d],np.array([popt[2],popt[3]]),np.array([Uncer_noCO[2],Uncer_noCO[3]]))
        Uncph=error_prop(ph,[c,d],np.array([popt[2],popt[3]]),np.array([Uncer_noCO[2],Uncer_noCO[3]]))

        AMv=AMP(popt[2],popt[3])
        PHv=PH(popt[2],popt[3])        

    if cubic:
        UncAm=popta[3]
        AMv=popta[2]
    ##print popt


    return [popt[1],Uncer_minann2[1],AMv,UncAm,popt[0],popt[2],popt[3],PHv,y_minann_and_trend,
            pd.Series(y_mintrend, index =series.index)]#,pd.Series(y_minann_and_trend, index =series.index)]

        
        


        

def AMP(C,D):
    return np.sqrt(C**2+D**2)

def PH(C,D):
    if C>0:
        return np.arctan(D/C)+np.pi
    else: 
        return np.arctan(D/C)

def error_prop(f,varlist,vals,errs):
    n=len(varlist)
    df_at_x=[]
    for i in range(n):
        sig = f.diff(varlist[i])
        t=sig
        for j in range(n):
            t=t.subs({varlist[j]:vals[j]})
        df_at_x.append(t)
    error1 = sympy.sqrt(np.sum((np.asarray(df_at_x)**2)*(errs**2)))
    
    return error1

# fit functions

def ftrend_only(x,a,b):
    return  a + b*x

def trend_only_and_cubic(x,a,b,c,d,e): #,c,d):
    return  a + b*x + c*np.cos((2*np.pi*x*d)+e) #+c*(x**2) #+d*(x**3)

def trend_only_vec(x,a,b):
    return  a*x[:,3] + b*x[:,0]

def two_gauss( x, c1, mu1, sigma1, c2, mu2, sigma2):
    res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return res

def annual_cycle(x,a,b,c,d):
#    year=pd.Timedelta('365.2422 days')
#    f=year.total_seconds()    
    return  a + b*x + c*np.cos(2*np.pi*x) + d*np.sin(2*np.pi*x)

def annual_cycle_semi(x,a,b,c,d,e,f):
#    year=pd.Timedelta('365.2422 days')
#    f=year.total_seconds()    
    return  a + b*x + c*np.cos(2*np.pi*x) + d*np.sin(2*np.pi*x) + e*np.cos(4*np.pi*x) + f*np.sin(4*np.pi*x)

def f_linear(x,a):
    return a*x   


def annual_cycle_vec(x,a,b,c,d):
#    year=pd.Timedelta('365.2422 days')
#    f=year.total_seconds()    
    return  a*x[:,3] + b*x[:,0] + c*x[:,1] + d*x[:,2]






