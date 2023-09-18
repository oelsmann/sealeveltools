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
import os
import glob
from scipy import stats
from sympy import *
import sympy
from scipy.interpolate import griddata
from sealeveltools.sl_stats.resources.hector.analyse_timeseries_function import *
from sealeveltools.sl_stats.sl_kendall import *
from sealeveltools.sl_stats.resources.hector.analyse_and_plot_mod import *
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

def sl_fltr_rolling(ydata,xdata,**kwargs):
    series=sl_pd_series(ydata,xdata)
    if type(kwargs['win_size'])=='str':
        series=series.rolling(kwargs['win_size'],win_type=kwargs['win_type']).mean()
    else:
        series=series.rolling(kwargs['win_size'],win_type=kwargs['win_type']).mean()    
    return series
    

def sl_pd_series(ydata,xdata,monthly=False):
    """
    xdata - time-vector
    ydata - data-vecttor
    """
    xdata=pd.to_datetime(xdata)
    series=pd.Series(ydata, index=xdata)    
    if monthly:
        # create x_matrix with size 12 x len(xdata), and diagonal 1
        month=series.index.month[0]-1
        X_matrix=np.tile(np.identity(12), (int((len(xdata)+12)/12), 1))[month:len(xdata)+month,:] # shift matrix by first month
        X_matrix=X_matrix[pd.notnull(series),:]
        series = series[pd.notnull(series)] 
        return series,X_matrix
    else:
        series = series[pd.notnull(series)]    
        return series

def sl_trend_prepare(ydata,weight_data,xdata,**kwargs):
    """
    
    
    """
    
    if kwargs['monthly']:
        series,X_matrix=sl_pd_series(ydata,xdata,monthly=True)
        series=series[~np.isnan(series)].dropna()
        X_matrix=X_matrix[~np.isnan(series),:]
    else:
        series=sl_pd_series(ydata,xdata)
        series=series[~np.isnan(series)].dropna()        
        X_matrix=[]
    if kwargs['weights'] is not None:
        weight_data=pd.Series(weight_data).dropna().values
        
    if len(series.dropna())<kwargs['min_samples']:
        out=[np.nan]*17
        if kwargs['hector']==True: 
            return np.array(out[0:15])
        elif kwargs['kendall']==True: 
            return np.array(out[0:7])
        else:
            return np.array(out[0:9])
    else:        
        if kwargs['kendall']==True:            
            out=kendall_model(series,**kwargs)
            return np.asarray(out)
        else:                
            out=t_fit_annual_cycle(series,X_matrix=X_matrix,weight_data=weight_data,**kwargs)
            #[popta[1],Uncer_minann2[1],AMv,UncAm,y_minann_and_trend,
                    #pd.Series(y_mintrend, index =series.index)]#,pd.Series(y_minann_and_trend, index =series.index)]
            if kwargs['hector']==True:
                # estimate appropiate uncertainties with differentt noise models
                # based on the residuals (detrended + deseasoned)

                series=out[9]
                name=str(1)+str(np.random.normal())[3:]+'_series' # name of series = random

                file_name,trend,trend_err,AM,AM_err,AIC,BIC,driving_noise,WN_frac,M_frac,M_spec =t_fit_hector(series,rolling_mean=False,reprocess=False,
                                                         resample_freq='30',name=name,model=kwargs['model'])
                # WN, AR1, FLWN, RWFLWN, PLWN, GGMWN
                out[2]=trend_err
                if kwargs['hector_out']==True:
                    out[0]=trend
                    out[3]=AM
                    out[4]=AM_err

                return np.array(out[0:9]+[AIC,BIC,driving_noise,WN_frac,M_frac,M_spec])

            else:
                return np.array(out[0:9])



def trend_sub(self,**kwargs):    
    """
    sub-function to calculate trends and uncertainties
    using different statistical models
    
    ------------------------
    compute trend and statistics


    Parameters
    ----------

    de_season: subtract seasonal component (in case of de-seasoning the data)

    semi: compute semi-annuals cycle (sinusoidal)

    de_season_without_trend:  only remove annual cycle

    trend_only: trend fit only (no periodic components)

    min_samples: minimum number of required samples
    
    kendall: bool, default False
        compute trends using kendall's tau

    hector: bool, default True
        Turn on hector trend uncertainty computation

    model: WN, AR1, FNWN, RWFLWN, PLWN, GGMWN

    hector_out: bool, default True
                True: compute all statistics with hector (trend, cylces, uncertainties)
                False: only compute uncertainties with hector, and trend with linear fit

    monthly: if True then calculates a monthly-wise seasonal cycle, Amplitude is (min-max)/2

    ----------
    
    Returns: sl() object containing trend + uncertainty estimates for var
    
    """
    add_=''
    if kwargs['weights'] not in self.data:
        weight_data = self.data
    else:
        weight_data=self.data[kwargs['weights']]
        try:
            self.data = self.data[kwargs['var']]
        except ValueError:
            print("Specify the variable (var=<varname>) of which the trend should be computed, and make sure the variable is in the dataset")
        add_=' (weighted)'

    out=xr.apply_ufunc(sl_trend_prepare, self.data,weight_data,self.data.time,
                          input_core_dims=[["time"],["time"],["time"]],output_core_dims=[['var']],
                                 kwargs=kwargs,dask = 'allowed', vectorize = True)

    if kwargs['hector'] and not kwargs['monthly']:
        names=['trend','trend_un','trend_un '+kwargs['model'],'AC','AC_unc','offset','acos','asin','phase','AIC','BIC','driving_noise',
        'WN_frac','M_frac','M_spec']
        
        long_names=['linear trend','trend uncertainties OLS','trend uncertainties '+kwargs['model'],'annual cycle','annual cycle uncertainties',
                    'offset','acos','asin','annual cycle phase','AIC','BIC','driving_noise',
        'WN fraction','Model fraction','Model specs.']

        units=['m/year','m/year','m/year','m','m','m','m','m','degree','','','',
        '','','']
        units_more=['','','','','','','','','0° degree minimum at 12.31.','','','',
        '','','']   
        
    elif kwargs['monthly']:
        names=['trend','trend_un','trend_un_AR1','AC','AC_unc','max_mon','acos','asin','min_mon']
        long_names=['linear trend','trend uncertainties OLS','trend uncertainties AR1',
                    'annual cycle','annual cylce uncertainties',
                    'maximum month','acos','asin','minimum month']

        units=['m/year','m/year','m/year','m','m','counts','m','m','counts']
        units_more=['','','','','','','','','']
        if kwargs['monthly'] and kwargs['hector']:
                names=['trend','trend_un','trend_un '+kwargs['model'],'AC','AC_unc','max_mon','acos','asin',
                       'min_mon','AIC','BIC','driving_noise',
                'WN_frac','M_frac','M_spec']

                long_names=['linear trend'+add_,'trend uncertainties OLS','trend uncertainties AR1',
                            'annual cycle','annual cylce uncertainties',
                            'maximum month','acos','asin',
                            'minimum month','AIC','BIC','driving_noise',
                'WN fraction','Model fraction','Model specs.']

                units=['m/year','m/year','m/year','m','m','counts','m','m','counts','','','','','','']
                units_more=['','','','','','','','','','','','','','','']
                    
    elif kwargs['kendall']:
        #trend_seasonal, uncertainty_seasonal, upper, lower confidence bounds (95%, seasonal), sig_seasonal,
        #                    trend_not_seasonal,sig_not_seasonal
        
        names=['trend','trend_un','upper_bound','lower_bound','sig','trend_noseas','sig_noseas']
        long_names=['linear trend','trend uncertainties Kendall','Upper confidence bound',
                    'Lower confidence bound','Significance',
                    'linear trend (no seasonal)','Significance (no seasonal)']

        units=['m/year','m/year','m/year','m/year','','m/year','']
        units_more=['','','','','','','']

    else:
        names=['trend','trend_un','trend_un_AR1','AC','AC_unc','offset','acos','asin','phase']
        long_names=['linear trend'+add_,'trend uncertainties OLS','trend uncertainties AR1',
                    'annual cycle','annual cylce uncertainties',
                    'offset','acos','asin','annual cycle phase']

        units=['m/year','m/year','m/year','m','m','m','m','m','degree']
        units_more=['','','','','','','','','0° degree minimum at 12.31.']

    
    i=0
    ds_array=[]
    if self.typ=='xra':
        print('auto-select first variable ',self.var)
        out=out[self.var]
    

    for name in names:
        
        ds_array.append(out[...,i].rename(name).astype(float))
        ds_array[i].attrs={'standard_name': [names[i]], 'long_name': long_names[i],'_FillValue' : 'nan', 'units': units[i],'info': units_more[i]}
        i=i+1
    out1=xr.merge(ds_array,compat='override')    
    
    if kwargs['hector']:
        fileDir=os.path.dirname(os.path.realpath(__file__))
        new=os.path.join(fileDir, 'resources/hector/')
        fileList = glob.glob(new+'obs_files/*seri*') + glob.glob(new+'mom_files/*seri*') + glob.glob(new+'pre_files/*seri*')

        # Iterate over the list of filepaths & remove each file.
        # Remove the timeseries text file in hector
        
        for filePath in fileList:
            try:
                #os.remove(filePath)
                #print(filePath)
                1+1
            except:
                print("Error while deleting file : ", filePath)

    return out1




#-------------------------------------------
# main fit


def t_fit_annual_cycle(series,X_matrix=[],weight_data=None,C=0.001,P=1.5,**kwargs):
    
    """
    fit annual cycle and trend 
    use prais-winston method
    
    Parameters:
    ------------
    series: pandas time series to fit trend
    
    X_matrix: stacked identity matrix for monthly annual cycle evaluation
    
    weight_data: data weights
            default = None
    
    Returns:
    ------------
    
    popt : array
        Optimal values for the parameters so that the sum of the squared residuals of 
        f(xdata, *popt) - ydata is minimized
        
        popt[0:3] : offset, trend, ...
    
    """
    de_season=kwargs['de_season']
    semi=kwargs['semi']    
    de_season_without_trend=kwargs['de_season_without_trend']
    trend_only=kwargs['trend_only']
    hector=kwargs['hector']
    hector_out=kwargs['hector_out']
    monthly=kwargs['monthly']
    
    series=series[~np.isnan(series)]
    year=pd.Timedelta('365.2422 days')
    f=year.total_seconds()   
    xdata=(series.index-pd.to_datetime('1990-01-01')).total_seconds().values/f
    ydata=series.values
    
    if kwargs['weights']==None:
        weight_data=None
    if semi:
        popt, pcov = curve_fit(annual_cycle_semi, xdata, ydata,sigma=weight_data)   
        #y_minann=ydata-annual_cycle_semi(xdata,0,0,popt[2],popt[3],popt[4],popt[5])
        DoF=len(xdata)-6 #Degrees of Freedom N-(A,B,C,D)
    elif trend_only:
        popt=[0]*10
        pcov=[0]*10
        poptb, pcovb = curve_fit(ftrend_only, xdata, ydata,sigma=weight_data)  
        popt[0:2]=poptb[0:2]
        pcov=pcovb
        #y_minann=ydata  
        DoF=len(xdata)-2 #Degrees of Freedom N-(A,B,C,D)
    elif monthly:
        # fit offset trend and monthly annual cycle
        popt, pcov = curve_fit(annual_cycle_mon, np.c_[xdata,X_matrix], ydata,sigma=weight_data)  
        #y_minann=ydata-annual_cycle_monthly(np.c_[xdata,X_matrix],0,1,popt[2],popt[3],popt[4],popt[5],
        #                             popt[6],popt[7],popt[8],popt[9],popt[10],popt[11],popt[12],popt[13])
        n_parameter=1+12
        DoF=len(xdata)-n_parameter #Degrees of Freedom N-(A,B,C,D)        
        
        
    else:    
        popt, pcov = curve_fit(annual_cycle, xdata, ydata,sigma=weight_data)
        #y_minann=ydata-annual_cycle(xdata,0,0,popt[2],popt[3])
        DoF=len(xdata)-4 #Degrees of Freedom N-(A,B,C,D)

        
    t_critical=stats.t.ppf(1-0.025,DoF) #95 % confidence interval    
    # option to start preis winston     
    # SE - standard error     
    #see https://en.wikipedia.org/wiki/Student%27s_t-distribution#Confidence_intervals
    
    SE_noCO=np.sqrt(np.diag(pcov))
    Uncer_noCO=SE_noCO*t_critical 
   
    # SE with autocorrelation AR1:
    # see Santer, B.D., Wigley, T.M.L., Boyle, J.S., Gaffen, D.J., Hnilo, J.J., Nychka, D., Parker, D.E., Taylor, K.E., 2000. Statistical significance of trends and trend differences in layer-average atmospheric temperature time series. J. Geophys. Res.105 (D6), 7337–7356. http://dx.doi.org/10.1029/1999JD901105.
    
    lag1=series.autocorr(lag=1)
    Uncer_noCO_AR1=Uncer_noCO[1]*np.sqrt((1+lag1)/(1-lag1))
    
    # subtract annual cycle

    if de_season:
        if monthly:
            y_mintrend=ydata-annual_cycle_mon(np.c_[xdata,X_matrix],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],
                                         popt[6],popt[7],popt[8],popt[9],popt[10],popt[11],popt[12])  
        else:
            y_mintrend =ydata-annual_cycle(xdata,popt[0],popt[1],popt[2],popt[3])  
                        
    else:
        y_mintrend =ydata-annual_cycle(xdata,popt[0],popt[1],0,0) 
        # only subtract trend

    if de_season_without_trend:
        y_mintrend =ydata-annual_cycle(xdata,0,0,popt[2],popt[3])    

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
        min_mon=PHv*360/(2*np.pi)
    elif monthly:
        PHv=np.nan
        AMv=(np.max(popt[1:])-np.min(popt[1:]))/2
        UncAm = np.sqrt((0.5**2)*(Uncer_noCO[1:][np.argmax(popt[1:])])**2+(0.5**2)*(Uncer_noCO[1:][np.argmin(popt[1:])]**2))
        min_mon=np.argmin(popt[1:])
        max_mon=np.argmax(popt[1:])
        #print(popt[1:])
        popt[1]=popt[0] # trend
        Uncer_noCO[1] = Uncer_noCO[0] # trend uncertainty
        
        
        popt[2]=np.nan
        popt[3]=np.nan
        
    else:
        UncAm=error_prop(amp,[c,d],np.array([popt[2],popt[3]]),np.array([Uncer_noCO[2],Uncer_noCO[3]]))
        Uncph=error_prop(ph,[c,d],np.array([popt[2],popt[3]]),np.array([Uncer_noCO[2],Uncer_noCO[3]]))

        AMv=AMP(popt[2],popt[3])
        PHv=PH(popt[2],popt[3])        
        min_mon=PHv*360/(2*np.pi)
    

    
    # Uncertainties are instantaneousely computed when fitting all signal components!
    #['trend','trend_un','trend_un_AR1','AC','AC_unc','offset','acos','asin','phase','detrended and deseasoned timeseries']
    if hector_out: # compute full stats with hector
        ser_out=series        
    else:
        ser_out=pd.Series(y_mintrend,index =series.index)
    
    if monthly:
        return [popt[1],Uncer_noCO[1],Uncer_noCO_AR1,AMv,UncAm,max_mon,popt[2],popt[3],
                min_mon,ser_out]                
    else:
        
        return [popt[1],Uncer_noCO[1],Uncer_noCO_AR1,AMv,UncAm,popt[0],popt[2],popt[3],
                min_mon,ser_out]

        
        


        

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

def annual_cycle_monthly(x,a,b,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12):
#    year=pd.Timedelta('365.2422 days')
#    f=year.total_seconds()   

    
    cycle=det_dot(x[:,1:],np.array([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12]))
  
    return a + b*x[:,0] + cycle

def annual_cycle_mon(x,b,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12):
#    year=pd.Timedelta('365.2422 days')
#    f=year.total_seconds()   

    
    cycle=det_dot(x[:,1:],np.array([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12]))
  
    return b*x[:,0] + cycle


def f_linear(x,a):
    return a*x   


def annual_cycle_vec(x,a,b,c,d):
#    year=pd.Timedelta('365.2422 days')
#    f=year.total_seconds()    
    return  a*x[:,3] + b*x[:,0] + c*x[:,1] + d*x[:,2]

def det_dot(a, b):
    """
    
    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)


## hector


def hec_test():
    
    print(os.path.abspath(__file__))
    
def t_fit_hector(series,rolling_mean=False,reprocess=False,
                 resample_freq='30',name='test',model='PLWN',force_name=False):
    """
    make uncertainty estmation with Hector software:
    - > starts the function analyse_timeseries (includes outlier 
        detection and parameter fit of linear model
        [trend semi+annual cycle + offsets] and noise model)
        chose among a variety of noise models:
            WN White Noise
            FN Flicker Noise
            PL Power-law Noise
            RW Random Walk
            GGM Generalised Gauss-Markov noise model
            AR1 ARMA(1,0) first-order autogressive noise model
        standard: PLWN, or model
        
        The preferred model is the one with the minimum AIC/BIC value. Note that these are
        relative measures between various choices, not absolute criteria.
    """
   

    if rolling_mean:
        series=series.rolling(resample_freq+'D').mean().resample('30D').mean().dropna()  
        resample_freq='30'
    else:

        #print 'Resample series: ',resample_freq+'D'
        series=series.resample(resample_freq+'D').mean().dropna()

    index=series.index.to_julian_date().values
    index=index-(index[0]-50084.)
    data=series.values*1000 #now values in mm
    text=np.stack((index,data),axis=1)
    fileDir=os.path.dirname(os.path.realpath(__file__))
    new=os.path.join(fileDir, 'resources/hector/')

    file_name=new+'obs_files/'+name+'.mom'
    
    if resample_freq=='0.5':
        np.savetxt(file_name,text,header='sampling period '+resample_freq+'',fmt=['%.1f','%.4f'])    
    else:
        np.savetxt(file_name,text,header='sampling period '+resample_freq+'.0',fmt=['%.1f','%.4f'])


    os.chdir(new)

    out=analyse_timeseries_function(name,model,fraction=True,force_name=force_name)

    out = [float(i) for i in out]
    os.chdir(fileDir)
    #output = [trend,trend_error,N,LogL,aic,bic,bic_c,Sa_cos,Sa_sin, \
    #           Sa_cos_sigma, Sa_sin_sigma, Ssa_cos, Ssa_sin, Ssa_cos_sigma, \
	#			    			   Ssa_sin_sigma,std]
    
    AMv=AMP(out[7],out[8])

           #trend,        trend_err,     AM,      AM_err,WN fraction, PL amp fraction, spectral index
    if model=='PLWN':
        return name,out[0]/1000,out[1]/1000,AMv/1000,0.,out[4],  out[5],       out[15],       out[16],out[17],out[18]
    elif model=='AR1':
        return name,out[0]/1000,out[1]/1000,AMv/1000,0.,out[4],out[5],out[15],out[16],out[16],out[17]

    elif model=='FLWN':
        return name,out[0]/1000,out[1]/1000,AMv/1000,0.,out[4],out[5],out[15],out[16],out[16],out[17]

    elif model=='RWFLWN':
        return name,out[0]/1000,out[1]/1000,AMv/1000,0.,out[4],out[5],out[15],out[16],out[17],out[18]

    elif model=='GGMWN':
        return name,out[0]/1000,out[1]/1000,AMv/1000,0.,out[4],out[5],out[15],out[16],out[17],out[18]
        
    else:
        #print(out)
        return name,out[0]/1000,out[1]/1000,AMv/1000,0.,out[4],out[5],out[15],0,0,0


def hector_estimate_noise_models(series,noisemodels = ['GGMWN','PLWN','FNWN','RWFNWN','WN','AR1'],name='test',force_name=False):
    """ estimate different noise models and plot data
    
    
    """
    specs = []
    models = []
    for noisemodel in noisemodels:
        print(noisemodel)
        filename,trend,trend_err,AM,AM_err,AIC,BIC,driving_noise,WN_frac,M_frac,M_spec =t_fit_hector(series,rolling_mean=False,reprocess=False,
                                                             resample_freq='30',name=name,model=noisemodel,force_name=force_name)
        fileDir=os.path.dirname(os.path.realpath(__file__))
        new=os.path.join(fileDir, 'resources/hector/')    

        os.chdir(new)        
        estimate_and_model_spectrum(filename+'.mom',noisemodel=noisemodel)
        if force_name:
            dat_model = pd.read_csv(new+'modelspectrum'+name+'.out',delim_whitespace=True,header=None)
            dat_est = pd.read_csv(new+'estimatespectrum'+name+'.out',delim_whitespace=True,header=None)
        else:
            dat_model = pd.read_csv(new+'modelspectrum.out',delim_whitespace=True,header=None)
            dat_est = pd.read_csv(new+'estimatespectrum.out',delim_whitespace=True,header=None)
        os.chdir(fileDir) 
        specs.append([noisemodel,trend,trend_err,AM,AM_err,AIC,BIC,driving_noise,WN_frac,M_frac,M_spec])
        models.append([noisemodel,dat_model,dat_est])
    specs = pd.DataFrame(specs,columns = ['model','trend','trend_un','AC','AC_unc','AIC','BIC','driving_noise',
        'WN_frac','M_frac','M_spec'])
    
    best_model = [False] * len(specs)
    best_model[np.argmin(specs['AIC']+specs['BIC'])]=True
    specs['best_model'] = best_model
    return specs,models


# time interp.

# remapping

def grid_pd_loop(x,leng,grid_x, grid_y,points):

    grid = griddata(points, x[:leng].values, (grid_y, grid_x), method='linear').flatten()

    x.loc[:len(grid)-1]=grid

    return x

# constructing



def reconstruct_data(self,data,latnew,lonnew,time_=True):
   
    if self.var=='':
        var='new_var'
    else:
        var=self.var
    if time_:
        ds = xr.Dataset({var: (['time','lat', 'lon'],  data)},
                             coords={'lon': (['lon'], lonnew),
                            'lat': (['lat'], latnew),
                            'time':self.data.time.values}) 
    else:

        ds = xr.Dataset({var: (['lat', 'lon'],  data)},
                             coords={'lon': (['lon'], lonnew),
                            'lat': (['lat'], latnew)})         
        

    ds.attrs=self.data.attrs   
    return ds

def reconstruct_irreg_data(self,data,latnew,lonnew,index=[],
                           coords_in={},time_=True):
    """
    construct irregularly gridded dataset
    
    Parameters:
    
    data:
    
    latnew
    
    lonnew
    
    index=[]
    
    coords_in={}
    
    time_=True    
    
    """
    if self.var=='':
        var='new_var'
    else:
        var=self.var
        
    spacedim,timedim=check_spacetime_dims(self)
    if timedim==1:
        time_=False
    
    coords={'lon': (['x'], lonnew),'lat': (['x'], latnew)}
    var_dict = {var: (['x'],  data)}
    if len(index) > 0:
        print(index)
        coords['idx'] = (['x'], index)
    if time_:
        coords['time'] = self.data.time.values
        var_dict = {var: (['time','x'],  data)}
    coords = {**coords, **coords_in}
    
    ds = xr.Dataset(var_dict,coords=coords)         

    ds.attrs==self.data.attrs

    return ds[var]   

# spatial smoothing functions

def weighted_std(ds):
    
    (average_w,std_w)=weighted_avg_and_std(ds.values, ds.weights_combined.values)    
    ds[:]=std_w
    return ds.mean(dim='x')
    

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))



def spatial_fltr(ds,length_scale,function='cos'):
    """
    apply spatial filter functions
    """    
    w = np.cos((np.pi*ds.dist.values)/(2*length_scale))
    ds=ds.assign_coords(s_weights=ds.dist,dim='x')
    ds=ds.assign_coords(counts=ds.lat,dim='x')
    ds['counts'][:]=ds.count(dim='x').values
    ds['s_weights'][:]=w/np.sum(w)
    return ds

def inverse_weigthing(ds,exponent=1):
    """
    prodcue inversely weighted weigths    
    """
    ds=ds.assign_coords(u_weights=ds.dist,dim='x')
    ds['u_weights'][:]=((1/ds)**exponent)/(np.sum(1/ds)**exponent)
    return ds 



def weight_vals_radially(variable,original=None,uncertainty=None,exponent=1,
                         lenght_scale=150.,function='cos'):
    """
    compute weighted average and std in specified range
    
    """
    # produce weights
    variable_weights=variable.data.groupby('idx').apply(lambda x: spatial_fltr(x,lenght_scale*1000.,function=function))
    
    if uncertainty is None:
        weighted=(variable_weights*variable_weights.s_weights).groupby('idx').sum() # weigth by distance
        variable_weights['weights_combined']=variable_weights.s_weights
        variable_weights['weights_combined'][:]=variable_weights.s_weights.values        
        d_std=variable_weights.groupby('idx').apply(lambda x: weighted_std(x))
        
    else:
        u_weights=uncertainty.data.groupby('idx').apply(lambda x: inverse_weigthing(x)).u_weights # weigth by uncertainty
        weights_combined=u_weights*variable_weights.s_weights        
        weighted=(variable_weights*weights_combined.values).groupby('idx').sum()/(weights_combined.groupby('idx').sum().values)
        ## also compute weighted std            
        variable_weights['weights_combined']=variable_weights.s_weights
        variable_weights['weights_combined'][:]=weights_combined.values
        d_std=variable_weights.groupby('idx').apply(lambda x: weighted_std(x))

      
    _,index = np.unique(original.data['idx'], return_index=True)
    original=original.data.isel(x=index)
    original[:]=weighted.values
    original['counts']=original.dist
    original['counts'][:]=variable_weights.counts.groupby('idx').mean().values
    
    or2=copy.deepcopy(original)
    or2[:]=d_std.values
    or2=or2.rename('trend_std')
    or2.attrs['long_name']='weighted standard deviation, '+str(lenght_scale)+' km smoothing;'
    or2.attrs['units']='mm/year'
    
    all_ = xr.merge([or2,original])
    return all_

# coordinate interp.


#def match_dask_xr():
    
    



def check_spacetime_dims(dat):
    """
    checks space dimensions
    1. dimension must always be time if exists
    
    returns:
    
        sshape: space-dimension
        
        st:     time-dimensions
    """
    
    if dat.typ=='xr':
        if 'time' in dat.data.dims and len(dat.shape) == 1:
            return 1,dat.shape[0]

        elif 'time' in dat.data.dims and len(dat.shape) > 1:
            return dat.shape[1:],dat.shape[0]
        
        else:
            return dat.shape,1
    else:
        ll=len(dat.data)
        return ll,ll
    
def match_pd_multi_xr(self,data2,rmptime=True,s_method='zoi',
                         method='time',limit=100.,**kwargs):
    """
    matches dask array of structure
    
    mission lon lat time obs.
    ...
    ...
    
    with standard xarray
    """
    kwargs={'limit': 2,'limit_direction': 'both','method':method,'smooth':'','freq':40}
    # calculate distances
    dist=interp_pandas_like(self,data2,method='all',limit=limit).fillna(method='ffill')

    #all values needed to map
    lat=pd.DataFrame(self.data.index.get_level_values(1))
    lon=pd.DataFrame(self.data.index.get_level_values(2))
    index=np.isin(lat,self.flat(kind='lat')[dist['idx'].values.astype(int)]).flatten() & np.isin(lon,self.flat(kind='lon')[dist['idx'].values.astype(int)]).flatten()
    # next steps:
    # match individual series in time:

    # options for GESLA and PSMSL
    # a) only PSMSL
    target_x=data2.data[...,0].values*np.nan
    target_time=data2.data.time.values

    
    flat_x_out=self.data[index].groupby(level=[1,2]).apply(lambda x: 
                                                self.match_pd_time(x['sla'].values,pd.DatetimeIndex(x['time'].values),target_x,target_time,smooth=kwargs['smooth'],**kwargs))


    #flat_x_out.reshape()
    #self.data[0:300].groupby(level=[1,2]).apply(lambda x: print(x['sla'].values,pd.DatetimeIndex(x['time'].values))#,x['time'].values,target_x,target_time))

    dist_1_idx=dist['idx'].values.astype(int)
    dist_2_idx=dist['level'].values.astype(int)
    # match in space
    flat_xnew=np.swapaxes(flat_x_out.values.flatten().reshape(len(dist['idx'].drop_duplicates()),len(target_time)),0,1)[:,dist_1_idx]
    # reshaped interpolated array (time x space)

    latnew=self.flat(kind='lat')[dist_1_idx] 
    lonnew=self.flat(kind='lon')[dist_1_idx]

    latnewtg=data2.flat(kind='lat')[dist_2_idx] 
    lonnewtg=data2.flat(kind='lon')[dist_2_idx]

    self.data=reconstruct_irreg_data(data2,flat_xnew,latnew,lonnew,index=dist_2_idx) # give index to groupby
    data2.data=reconstruct_irreg_data(data2,data2.flat_x[:,dist_2_idx],latnewtg,lonnewtg,index=dist_2_idx)
    
    return self, data2        
    
    

def match_xr_group(x,self,data2,func):
    """
    function to match self and data2
    
    func: 'mean' or 'median'
    
    """
 
    #dist_2_idx=x.level.dropna().values.astype(int)
    #dist_self_idx=
    
    #flat_x2=data2.flat_x*np.nan
    #flat_x2[...,dist_2_idx]=data2.flat_x[...,dist_2_idx]    
    
    
    flat_xnew=self.flat_x[...,x.idx.values.astype(int)]
    
    #print(np.nanmean(flat_xnew,axis=(len(flat_xnew.shape)-1)))
    #print(flat_xnew.shape)
    #print(flat_xnew.size)
    #print(len(flat_xnew.shape))    
    if func=='mean':
        arr=np.nanmean(flat_xnew,axis=(len(flat_xnew.shape)-1))    
    elif func=='median':
        arr=np.nanmedian(flat_xnew,axis=(len(flat_xnew.shape)-1))
    
    return arr  
    
def from_flat_to_data(dat,flat_x_in,shp,adjust_time=False,time=[]):
    """
    add back modified flat-x data
    dat - 
    
    """

    if adjust_time:
        if len(time)==0:  
            data=flat_x_in.reshape(shp) 
        else:
            data=flat_x_in.reshape((len(time),)+shp)    
        if 'time' in dat.dims:
            dat=copy.deepcopy(dat[0,...])
            #dat=dat.squeeze(dim='time')
            dat[:]=data   
        else:
            dat = dat.expand_dims(time=time)
            dat=dat*0+data        
    else:
        data=flat_x_in.reshape(shp)
        dat=dat*0+data

    return dat    
 

def min_lon(phi,min_km):
    lonmin=round(min_km/(((math.cos(phi*(2*3.14/360))*6371)*2)*3.14/360),1)
    return lonmin 


def betw_map_coords(x,coord2,method='nearest',limit=100.):
    """
    map coords of x and coord2
    
    Parameters:
    
    ----------
    x: the target coordinate, to be matched with coord2
    
    coord2: arrays of lat and lon data
    
    method: 'nearest' (default)
    
    """
    
    if method=='nearest':

        dist,idx=map_coords(x.values[0:2],coord2,method=method,limit=limit)
        
        x['dist']=dist[0]
        x['idx']=idx[0]
        return x    
    else:
        dist,idx=map_coords(x.values[0][0:2],coord2,method=method,limit=limit)
        # Note this column has 4 rows of values:
        additional = pd.DataFrame({'dist': dist, 'idx':idx})
        new = pd.concat([x, additional], axis=1)
        new[[0,1,'level']]=new[[0,1,'level']].fillna(method='bfill') # addition after pd.version change
        return new

    
def map_coords(coord1,coord2,method='nearest',limit=100.):
    """
    couple coord1 and coord2
    
    """
    minl=min_lon(coord1[0],limit+10)  
    
    select2=np.argwhere((coord2[1] < coord1[1]+minl) * (coord2[1] > coord1[1]-minl)).squeeze()

    if select2.size >1:

        dist=haversine(coord1,[coord2[0][select2],coord2[1][select2]])

        if method=='nearest':        
            idx_=np.argmin(dist)

            if dist[idx_] < limit*1000.:    
                return [dist[idx_]] , [int(select2[idx_])]
            else:
                return [np.nan] , [np.nan]
        else:
            idx_=np.argmin(dist)  
            if dist[idx_] < limit*1000.:    

                return dist[dist<limit*1000.] , (select2[dist<limit*1000]).astype('int')
            else:
                return [np.nan] , [np.nan]            
  
    else:
        
        return [np.nan] , [np.nan]
    

def haversine(coord1, coord2):
    R = 6372800  # Earth radius in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1, phi2 = np.radians(lat1), np.radians(lat2) 
    dphi       = np.radians(lat2 - lat1)
    dlambda    = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + \
        np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    if str(type(a))=="<class 'numpy.float64'>":
        return 2*R*math.atan2(math.sqrt(a), math.sqrt(1 - a))
    else:
        return 2*R*np.arctan(np.sqrt(a), np.sqrt(1 - a))


    
def interp_pandas_like(self,data2,method='nearest',limit=100.):
    """
    made for unstructured griddata
    limit: float - maximum distance in km
    
    returns dist['level','lat','lon','idx']
        
    
    """
   
    #1.
    #if 'lat' is not in self.data.dims:
    #    lat,lon=self.flat(kind='lat'),self.flat(kind='lon')
    #    lat2,lon2=data2.flat(kind='lat'),data2.flat(kind='lon')
    
    lat,lon=self.flat(kind='lat'),self.flat(kind='lon')
    lat2,lon2=data2.flat(kind='lat'),data2.flat(kind='lon') # less coords   

    
    adj_range=1.
    select = np.argwhere((lon2> np.min(lon)-adj_range) * (lon2< np.max(lon)+adj_range) * 
                         (lat2> np.min(lat)-adj_range) * (lat2< np.max(lat)+adj_range )).squeeze()
    coord1 = np.vstack((lat2,lon2))[:,select]
    coord2=[lat,lon]
    # 1. confine region
   
    array=pd.DataFrame(coord1).T
    array['level']=select[array.index.values]
    if method=='nearest':
        dist=array.agg(lambda x: betw_map_coords(x,coord2,method=method,limit=limit),axis=1).dropna()
        dist=dist.set_index(select[dist.index.values])
    else:
        dist=array.groupby(by='level').apply(lambda x: betw_map_coords(x,coord2,method=method,limit=limit))#.dropna()
        dist=dist.loc[dist['dist'].dropna().index]
    
    # return indices of first and second
    
    #dist=dist.set_index(select[dist.index.values])
    
    # dist.columns = [index data2, distance, index self]
    
    # explanation
    # first multiindex : index of array with less samples (e.g. TGs)
    # idx :              index of second array machted to groupindex
    # dist:              distance of measurement
    
    return dist

# constitutents as in lobeto


def adjust_lon_lat_box(lo1,lo2,la1,la2):
    
    if abs(la1) > 90:
        la1=90.*np.sign(la1)
    
    if abs(la2) > 90:
        la2=90.*np.sign(la2)  
    
    #if lo1 < -180:
    #    lo1=lo1+180
    
    if lo2>180:
        lo2=lo2-360
    
    return [lo1,lo2,la1,la2]
 

def ray_cords(coord1,d,phi,reso,start):
    """
    make coords of ray
    """
    npc=True
    R = 6378.1 #Radius of the Earth
    brng = math.radians(phi) #Bearing is 90 degrees converted to radians. 0*np.pi()/180 # in radians
    d = np.linspace(start,d,reso)#15 #Distance in km

    #lat2  52.20444 - the lat result I'm hoping for
    #lon2  0.36056 - the long result I'm hoping for.

    lat1 = math.radians(coord1[0]) #Current lat point converted to radians
    lon1 = math.radians(coord1[1]) #Current long point converted to radians

    if npc:
        lat2 = np.arcsin( math.sin(lat1)*np.cos(d/R) +
             math.cos(lat1)*np.sin(d/R)*math.cos(brng))

        lon2 = lon1 + np.arctan2(math.sin(brng)*np.sin(d/R)*math.cos(lat1),np.cos(d/R)-math.sin(lat1)*np.sin(lat2))
        
    else:
        lat2 = math.asin( math.sin(lat1)*math.cos(d/R) +
             math.cos(lat1)*math.sin(d/R)*math.cos(brng))

        lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat1),math.cos(d/R)-math.sin(lat1)*math.sin(lat2))

    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)
    return lat2,lon2



def get_ray_intersection_distance(coord1,d,data,x_num,reso,start):
    latl=[]
    lonl=[]
    for phi in np.linspace(1,360,x_num):
        lat2,lon2=ray_cords(coord1,d,phi,reso,start)
        latl.append(lat2)
        lonl.append(lon2)        
    return latl,lonl
    
    
    
    
def make_coastal_exp_constituent(coord1,data,radius=300,x_num=360,reso=200,start=1):
    """
    get mean distance of x_num rays, with maximum radius
    coord1=[lat,lon]
    make warning if not enough points are selected
    
    coord1: central coords
    radius: length of ray
    reso:   number of points per ray
    
    
    
    """
    
    minl=min_lon(coord1[0],radius+10)   
    
    minlat=radius/111.+0.2

    extend=adjust_lon_lat_box(coord1[1]-minl,coord1[1]+minl,coord1[0]-minlat,coord1[0]+minlat)
    #print(extend)
    
    datum_trans=False
    
    if (extend[0] > 100.) & (extend[1] < -100):
        # when lon crossing datum line
        domain = (
         ((data.coords["lat"] > extend[2]) &
          (data.coords["lat"] < extend[3]) &
         (data.coords["lon"] > extend[0]) &
         (data.coords["lon"] < 180.) |
         (data.coords["lat"] > extend[2]) &
          (data.coords["lat"] < extend[3]) &
         (data.coords["lon"] >= -180) &
         (data.coords["lon"] < extend[1])
         
         )
        )    
        datum_trans=True
        #print('datum trans')
    else:
        domain = (
         ((data.coords["lat"] > extend[2]) &
          (data.coords["lat"] < extend[3]) &
         (data.coords["lon"] > extend[0]) &
         (data.coords["lon"] < extend[1]) )
        )    
    
    data = data.where(domain,drop=True) 
    
    latl,lonl=get_ray_intersection_distance(coord1,radius,data,x_num,reso,start)
    
    #lonl[lonl<180]=lonl[lonl<180]-360
    
    if datum_trans:
        
        grid_x=pd.DataFrame(latl)
        grid_y=pd.DataFrame(lonl)
        grid_y[grid_y>180]=grid_y[grid_y>180]-360
        
        #print(data)
        #grid_y[grid_y>0 ]=grid_y[grid_y>0]+180
        lon_vals=data.lon.values
        #lon_vals[(lon_vals>0) & (lon_vals<20)]=lon_vals[(lon_vals>0) & (lon_vals<20)]+180
        #print(data)
        #print(lon_vals.max())        
        points=np.column_stack((np.meshgrid(lon_vals,data.lat.values)[0].flatten(),np.meshgrid(lon_vals,data.lat.values)[1].flatten()))
        #points=np.column_stack((data.lon.values,data.lat.values))

        grid = griddata(points, data.values.flatten(), (grid_y, grid_x), method='nearest').flatten()

        
    else:
        grid_x=pd.DataFrame(latl)
        grid_y=pd.DataFrame(lonl)

        points=np.column_stack((np.meshgrid(data.lon.values,data.lat.values)[0].flatten(),np.meshgrid(data.lon.values,data.lat.values)[1].flatten()))
        #points=np.column_stack((data.lon.values,data.lat.values))

        grid = griddata(points, data.values.flatten(), (grid_y, grid_x), method='nearest').flatten()        
        
        
    #ds = xr.Dataset({var: (['x', 'y'],  grid.reshape(x_num,reso))},
    #                     coords={'lon': (['x', 'y'], grid_y),
    #                    'lat': (['x', 'y'], grid_x)}) 
    
    dfram=pd.DataFrame(np.argwhere(grid.reshape(x_num,reso)==0))
    #print(dfram)
    out=dfram.groupby(dfram.columns[0]).apply(lambda x: np.min(x[1]))
    #print(out.values)
    
    if len(out.values) < 1:

        constituent=999.
    else:
        out=np.linspace(start,radius,reso)[out.values]
        out=np.concatenate([np.array((x_num-len(out))*[radius]),out.astype(float)])
        constituent=np.mean(out)        
    return constituent
