import numpy as np
import pandas as pd
from scipy import stats

def kendall_model(timser,**kwargs):
    """
    non parametric trend estimation
    (non)seasonal kendal tau
    
    need at least 100 samples 
    
    Parameters
    ----------

    ser: pd.Series
    
    trend_only: bool, default False
        if False compute seasonal kendall tau
    ----------
    
    Returns: list with [trend_seasonal, uncertainty_seasonal, upper, lower confidence bounds (95%, seasonal), sig_seasonal,
                            trend_not_seasonal,sig_not_seasonal]
    
    """

    if len(timser.dropna()) < 100:
        out=[np.nan]*7
        print('smaller hundred')
    else:
        #timser=pd.Series(data=ser.values,index=pd.to_datetime(ser.index)).dropna()
        kenda_rank_corr_sea,Limit_up,Limit_low,sig,kenda_rank_corr,sig_n=Kendal_test(timser,sig_lvl=95,
                                                                    season=~kwargs['trend_only'],printit=False)  

        out=[kenda_rank_corr_sea,((Limit_up-kenda_rank_corr_sea)+(kenda_rank_corr_sea-Limit_low))/2,
             Limit_up,Limit_low,sig,kenda_rank_corr,sig_n]
   
    return out
    
        
def t_sign(x):
    if x == 0. or np.isnan(x):
        return 0.
    else:
        return x/np.sqrt(x*x)

def std_prob_f(z):
    return (1/np.sqrt(2*np.pi))*np.exp(-(z**2)/2)



def t_transform(series_in):
    series=series_in.resample('M').mean().dropna()
    #series=series_in.dropna()
    x=(series.index-pd.to_datetime('1990-01-01')).total_seconds().values
    y=series.values
    return x,y

def t_transform_months(series_in,high_frequency):
    
    """
    transform in monthly blocks
    if high frequency = False; then make monthly means first
    
    
    """
    if high_frequency:
        series=series_in #.dropna() dont drop it  
        x=[]
        y=[]
        l=1000000000000.
        for i in range(1,13): # loop through 12 months
            sub_series=series.loc[series.index.month==i]
            #x.append((sub_series.index-pd.to_datetime('1990-01-01')).total_seconds().values)
            #y.append(sub_series.values)
            ll=len(sub_series.values)
            if ll < l: #check shortest timeseries
                l=ll        
        print('min_length (in terms of years)', l)       
        for i in range(1,13): # loop through 12 months
            sub_series=series.loc[series.index.month==i]
            x.append((sub_series.index-pd.to_datetime('1990-01-01')).total_seconds().values[:l])
            y.append(sub_series.values[:l])
         
        
    else:
        series=series_in.resample('M').mean() #.dropna() dont drop it
        x=[]
        y=[]
        l=1000000000000.
        for i in range(1,13): # loop through 12 months
            sub_series=series.loc[series.index.month==i]
            x.append((sub_series.index-pd.to_datetime('1990-01-01')).total_seconds().values)
            y.append(sub_series.values)
            ll=len(sub_series.values)

            if ll < l: #check shortest timeseries
                l=ll     
        print(l)
    return pd.DataFrame(x).values[:,:l],pd.DataFrame(y).values[:,:l],l

def stats_test_procedures(data):
    # kendall
    S_seas_arr,beta_seas,Limit_up,Limit_low=Kendal_test(data)
    # parametric - prais-winston
    popta,Uncer_minann2,AMv,UncAm=t_fit_annual_cycle(data.resample('M').mean(),autocorrelation=True)
    

    print('Prais Winston:')
    
    print('trend: ', popta)
    print('Uncer: ', Uncer_minann2)

      
def Kenda_S(y):
    S=0
    n=len(y)
    for k in range(n-1):
        for j in range(n)[k+1:]:
            S=S+t_sign(y[j]-y[k])
    return S

def Kendal_test(series,sig_lvl=95,season=False,printit=False,high_frequency=False):
    """
    calculate Kendall-trend, significance and confidence bounds
    Mann-Kendall test ~ non-parametric test of randomness against trend
    
    based on:
        Hirsch, R. M., Slack, J. R., & Smith, R. A. (1982). Techniques of Trend Analysis for Monthly Water Quality Data. 18(1), 107–121.
        Hirsch, R. M., Slack, J. R., & Geological, U. S. (1984). Rill R •. 20(6), 727–732. https://doi.org/10.1029/WR020i006p00727
        Helsel, B. D. R., & Hirsch, R. M. (n.d.). Statistical Methods in Water Resources.
        Young, I. R., Zieger, S., & Babanin, A. V. (2011). Supporting Online Material for Global Trends in Wind Speed and Wave Height. (March). https://doi.org/10.1126/science.1197219        
        Young, I. R., & Ribal, A. (2019). Supplementary Materials for and wave height. https://doi.org/10.1126/science.aav9527
        
    Calculates Kendall-trend, significance and confidence bounds
    
    Both, the standard (no-autocorrelation) and the seasonal (with autocorr.) trends are computed for comparison
    Decide whether to make monthly mean first or not

    Parameters
    ----------
    series : pd.series
        timeseries to test for trend    
    
    sig_lvl : int, default = 95
        confidence interval of the mk-test
        
    printit: bool, default False
        Print statistics
    high_frequency: bool, default True
        if False: resample to monthly values before computing seasonal MK-test
    
    
    Returns:
    -------
    kenda_rank_corr_sea*(60*60*24*365.2422) : float [mm/yr]
        trend derived with seasonal mk-test
    Limit_up : float [mm/yr]
        upper confidence bound
    Limit_low : float [mm/yr]
        lower confidence bound
    
    sig : int
        if 1: significant trend detected (with seasonal mk-test)
    
    """
    
    
    # Function as described by Young_2011 and Young_2019
    x,y=t_transform(series)
    # test statistic S for significance
    alpha=(1-sig_lvl/100.)
    S=Kenda_S(y)
    n=len(y)
    V_s = np.sqrt(n*(n-1)*(2*n+5)/18)  # variance        
    Z=(S-t_sign(S))/V_s               # standard normal variate Hirsch 1982
                                      # in a two-sided test for trend, the H0 
                                      # shouldbe accepted
                                      # ff [ZI <- z,/2,whereF•(z,/2) = a/2, Ftqbeingthe standard
                                      # normalcumulativedistribution function and a being the size
                                      # of the significance level for the test
    if abs(Z) <= stats.norm.ppf(1-(1-sig_lvl/100.)/2.):
        #print 'significant H_o -> there is no trend'
        sign=0.
    else:                                   
        #print 'not significant H_o -> there is a trend'
        sign=1.
    # kendal rank correlation -> trend (negecting seasonality)
    beta=[]
    for i in range(n-1):
        for j in range(n)[i+1:]:
            beta.append((y[j]-y[i])/(x[j]-x[i]))  
    kenda_rank_corr=np.median(np.asarray(beta))
   
    # 2: mk-seasonal test (with autocorrelation)    
    # Kendall-seasonal-test:
    # sub-devide into month segments!
    # method according to hirsch 1984, young 2011
    
    x,y,l=t_transform_months(series,high_frequency) # subdivide in 12 months-series (with missing values!) x.shape = [months,len(series@month)]
    S_seas_arr=np.empty(12)
    n=np.empty(12)
    V_sq_term1=0
    for i in range(0,12):
        S_seas_arr[i]=Kenda_S(y[i,:])    # seasonal S-statistic to test for significance
        n[i]=np.count_nonzero(~np.isnan(y[i,:]))
        V_sq_term1=V_sq_term1+n[i]*(n[i]-1)*(2*n[i]+5)/18 # first term: seasonal variance as in young 2011 - Supplementary, equation (11)    
    S_seas=sum(S_seas_arr)                    
    # second term sigma(K,r) as in Hirsch 19984
    
    R=np.empty((12,l))
    for g in range(0,12):
        for j in range(l):
            sum_sgn=0
            for i in range(l):
                sum_sgn=sum_sgn+t_sign(y[g,j]-y[g,i])
            R[g,j]=(n[g]+1+sum_sgn)/2                      # respecting missing data
    V_cov_term2_sigma=0
    for g in range(0,11):                                  # all combinations of g and h
        for h in range(0,12)[g+1:]:           
            K_gh=0
            for i in range(l-1):
                for j in range(l)[i+1:]:     
                    K_gh=K_gh+t_sign((y[g,j]-y[g,i])*(y[h,j]-y[h,i]))    
            sum_term=0
            for i in range(l):
                sum_term=sum_term+(R[g,i]*R[h,i])             
            V_cov_term2_sigma=V_cov_term2_sigma+(K_gh+4*sum_term-l*(n[g]+1)*(n[h]+1))/3             
            # according to Hirsch 1984 eq. (14)
    
    # second term done
    # full seasonal variance
    V_s_seas=V_sq_term1+V_cov_term2_sigma
    Z_seas=(S_seas-t_sign(S_seas))/np.sqrt(V_s_seas)

    if abs(Z_seas) <= stats.norm.ppf(1-(1-sig_lvl/100.)/2.):
        #print 'significant H_o -> there is no trend (seas test)'
        sig=0.
    else:                                   
        #print 'not significant H_o -> there is a trend (seas test)'
        sig=1.
    # kendal rank correlation -> trend (checking for seasonality)
    beta_seas=[]
    for i in range(l-1):
        for j in range(l)[i+1:]:
            for g in range(12):    
                beta_seas.append((y[g,j]-y[g,i])/(x[g,j]-x[g,i]))  
    beta_arr=np.asarray(beta_seas)              # this is the trend
    kenda_rank_corr_sea=np.nanmedian(beta_arr)    

    # now calculate upper and lower confidence intervals according to 'statistical methods in water resources' D.R. Helsel and R.M. Hirsch
    # N=number of possible pairwise slopes
    # Ru,Rl are upper and lower ranks in sorted beta (10.5 and 10.6)
    
    Ru=((beta_arr[~np.isnan(beta_arr)]).size+stats.norm.ppf(1-(1-sig_lvl/100.)/2.)*np.sqrt(V_s_seas))/2+1
    Rl=((beta_arr[~np.isnan(beta_arr)]).size-stats.norm.ppf(1-(1-sig_lvl/100.)/2.)*np.sqrt(V_s_seas))/2+1
    
    Limit_up=np.sort(beta_arr[~np.isnan(beta_arr)])[int(round(Ru))]*(60*60*24*365.2422)
    Limit_low=np.sort(beta_arr[~np.isnan(beta_arr)])[int(round(Rl))]*(60*60*24*365.2422)
    
    if printit:
        print('non-seasonal Kendall test:')
        
        print('trend: ', kenda_rank_corr*(60*60*24*365.2422))
        print('Z:     ', Z)
        print('S:     ', S)
        print('V:     ', V_s)
        
        print('seasonal Kendall test:') 
        
        print('trend: ', kenda_rank_corr_sea*(60*60*24*365.2422)) 
        print('Z:     ', Z_seas) 
        print('S_seas:', S_seas)     
        print('term 1:', V_sq_term1) 
        print('term 2:', V_cov_term2_sigma)  
        
        print('Uncer up:   '  ,Limit_up) 
        print('Uncer down: ',Limit_low)
        print('Uncer (avg):', ((kenda_rank_corr_sea*(60*60*24*365.2422)-Limit_low)+Limit_up-kenda_rank_corr_sea*(60*60*24*365.2422))/2) 
        
    #print stats.norm.ppf(1-(1-sig_lvl/100.)/2.)
    
    #if season:
    #    return kenda_rank_corr_sea*(60*60*24*365.2422),Limit_up,Limit_low,sig,beta_arr*(60*60*24*365.2422)
    #else:
    #    return kenda_rank_corr*(60*60*24*365.2422),Limit_up,Limit_low,sig,np.asarray(beta)*(60*60*24*365.2422)      
    return kenda_rank_corr_sea*(60*60*24*365.2422),Limit_up,Limit_low,sig,kenda_rank_corr*(60*60*24*365.2422),sign                  