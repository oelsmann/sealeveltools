import pymc3 as pm
import arviz as az
import numpy as np
import pandas as pd

def bayes_function(x,offset=0,trend=0,A_annual=0,A_semi=0,phi_annual=0,phi_semi=0,m_coeffs=0,mode='harmonic'):#,sigma):
    """
    model data 
    Parameters
    
    offset:     scalar
    trend:      scalar
    A_annual:   scalar or vector
    A_semi:     scalar or vector
    phi_annual: scalar or vector 
    phi_semi:   scalar or vector
    
    """
    if mode=='harmonic':
        annual=A_annual*np.cos(2*np.pi*X1 + phi_annual*np.pi/180)
        semi=A_semi*np.cos(4*np.pi*X1 + phi_semi*np.pi/180) 
        cycle=annual+semi
    elif mode=='month_wise':
        X_matrix=np.tile(np.identity(12), (int((len(x)+12)/12), 1))[0:len(x),:]
        cycle=A_annual*det_dot(X_matrix,m_coeffs)
        
        
    return  offset + trend*x + cycle


def state_space_fit_timeseries(y,X1):
    """
    this one works
    
    """
    with not_so_basic_model:

        # Priors for unknown model parameters
        # sigmas

        sigma_A_zero=pm.HalfNormal('sigma_A_zero', sigma=10)  # sigma annual cycle, RW-process
        sigma = pm.HalfNormal('sigma', sigma=10)              # sigma of white Gaussian noise
        sigma_trend = pm.HalfNormal('sigma_trend', sigma=10)  # sigma of slow varying RW-process

        # constants

        offset = pm.Normal('offset', mu=0, sigma=10)   
        mean_t=pm.Normal('mean_t', mu=0.0, sigma=0.1) 
        phi= pm.Normal('phi', mu=0, sigma=10) 

        # latent variables

        trend=pm.GaussianRandomWalk('trend', mu=mean_t, shape=len(X1), sigma=sigma_trend)
        Azero=pm.GaussianRandomWalk('Azero', mu=0, shape=len(X1), sigma=sigma_A_zero)    

        # model

        Amp_annual=np.exp(Azero)
        annual=Amp_annual*np.cos(2*np.pi*X1 + phi*np.pi/180)    
        mu = offset  + annual + trend

        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    with not_so_basic_model:
        # draw 500 posterior samples
        trace = pm.sample(1000)


def state_space_fit_timeseries_gaussian_random_walk(y,X1):
    """
    this one works
    
    """
    with not_so_basic_model:

        # Priors for unknown model parameters
        # sigmas

        sigma_A_zero=pm.HalfNormal('sigma_A_zero', sigma=10)  # sigma annual cycle, RW-process
        sigma = pm.HalfNormal('sigma', sigma=10)              # sigma of white Gaussian noise
        sigma_trend = pm.HalfNormal('sigma_trend', sigma=10)  # sigma of slow varying RW-process

        # constants

        offset = pm.Normal('offset', mu=0, sigma=10)   
        mean_t=pm.Normal('mean_t', mu=0.0, sigma=0.1) 
        phi= pm.Normal('phi', mu=0, sigma=10) 

        # latent variables

        trend=pm.GaussianRandomWalk('trend', mu=mean_t, shape=len(X1), sigma=sigma_trend)
        Azero=pm.GaussianRandomWalk('Azero', mu=0, shape=len(X1), sigma=sigma_A_zero)    

        # model

        Amp_annual=np.exp(Azero)
        annual=Amp_annual*np.cos(2*np.pi*X1 + phi*np.pi/180)    
        mu = offset  + annual + trend

        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)

    with not_so_basic_model:
        # draw 500 posterior samples
        trace = pm.sample(1000)

def state_space_fit_timeseries_gaussian_processes(y,X1):        

    tg=load_tg(kind='psmsl_full')
    key=tg.where(tg.lat==24.555,drop=True)

    basic_model = pm.Model()
    y=key['sla'].values[:,0][1280:]
    #
    # time vector increment of years
    X1=np.arange(len(y))/12.
    not_so_basic_model = pm.Model()
    y=y-np.nanmean(y)
    y[np.isnan(y)]=0
    t=X1[:,None]

    # normalize data

    first_co2 = y[0]
    std_co2 = np.std(y)
    y_n = (y - first_co2) / std_co2


    with pm.Model() as model:
        # yearly periodic component x long term trend
        η_per = pm.HalfCauchy("η_per", beta=2, testval=1.0)
        ℓ_pdecay = pm.Gamma("ℓ_pdecay", alpha=10, beta=0.075)
        period  = pm.Normal("period", mu=1, sigma=0.05)
        ℓ_psmooth = pm.Gamma("ℓ_psmooth ", alpha=4, beta=3)
        cov_seasonal = η_per**2 * pm.gp.cov.Periodic(1, period, ℓ_psmooth) \
                                * pm.gp.cov.Matern52(1, ℓ_pdecay)
        gp_seasonal = pm.gp.Marginal(cov_func=cov_seasonal)

        # small/medium term irregularities
        η_med = pm.HalfCauchy("η_med", beta=0.5, testval=0.1)
        ℓ_med = pm.Gamma("ℓ_med", alpha=2, beta=0.75)
        α = pm.Gamma("α", alpha=5, beta=2)
        cov_medium = η_med**2 * pm.gp.cov.RatQuad(1, ℓ_med, α)
        gp_medium = pm.gp.Marginal(cov_func=cov_medium)

        # long term trend
        η_trend = pm.HalfCauchy("η_trend", beta=2, testval=2.0)
        ℓ_trend = pm.Gamma("ℓ_trend", alpha=4, beta=0.1)
        cov_trend = η_trend**2 * pm.gp.cov.ExpQuad(1, ℓ_trend)
        gp_trend = pm.gp.Marginal(cov_func=cov_trend)

        # noise model
        η_noise = pm.HalfNormal("η_noise", sigma=0.5, testval=0.05)
        ℓ_noise = pm.Gamma("ℓ_noise", alpha=2, beta=4)
        σ  = pm.HalfNormal("σ",  sigma=0.25, testval=0.05)
        cov_noise = η_noise**2 * pm.gp.cov.Matern32(1, ℓ_noise) +\
                    pm.gp.cov.WhiteNoise(σ)

        # The Gaussian process is a sum of these three components
        gp = gp_seasonal + gp_medium + gp_trend

        # Since the normal noise model and the GP are conjugates, we use `Marginal` with the `.marginal_likelihood` method
        y_ = gp.marginal_likelihood("y", X=t, y=y, noise=cov_noise)

        # this line calls an optimizer to find the MAP
        mp = pm.find_MAP(include_transformed=True)
def changepoint_model(timser,n_changepoints=5,number_mu=0.1,
                      offsets_opt='normal',offsets_std=1):
    """
    model trends and changepoints
    
    """

    X1,y,index, std_data = normalize_timser(timser['vlm'],normalize=True,rmv_nan=True)
    x=X1    
    growth_prior_scale=5 
    changepoint_range=0.9
    basic_model = pm.Model()
    act_number=0

    xmin=np.min(x)
    xmax=np.max(x)
    xhalf=(xmax-xmin)/2.


    
    if len(y) < 100:
        data=[np.nan]*11
        print('smaller hundred')
    else:
    

        with basic_model:
            # Priors for unknown model parameters
            offset = pm.Normal('offset', mu=0, sigma=1)
            trend = pm.Normal('trend', mu=0, sigma=1)
            sigma = pm.HalfNormal('sigma', sigma=1)   
            #if changepoints_prior_scale is None:
            #    changepoints_prior_scale = pm.Exponential('tau', 1.5)
            # number of changepoints = 
            #n_changepoints = pm.HalfNormal('n_changepoints', sigma=1.0).astype(int)+1
            #n_changepoints = pm.Poisson('n_changepoints', mu=0.5)+1
            #n_changepointsv=1
            #n_changepoints=n_changepoints

            #test=np.arange(20000000)-100
            #print(n_changepoints)
            #n_changepointsv=n_changepoints#int(test[test <= n_changepoints])
            #if n_changepoints < 1:
            #    n_changepointsv=1
            #elif n_changepoints ==1.0:
            #    n_changepointsv=1    
            #elif n_changepoints > 1.:
            #    n_changepointsv=2        

            # rate of change 
            #offsets = pm.Uniform('offsets', lower=-4., upper=4., shape=int(n_changepoints))

            #mu_p=pm.HalfNormal('mu_p', sigma=0.5)
            act_number=pm.Poisson('act_number', mu=number_mu)
            #act_number=pm.ChiSquared(1.5)
            if offsets_opt=='normal':
                offsets = pm.Normal('offsets', mu=0, sigma=offsets_std, shape=n_changepoints)  
            mult=np.zeros(n_changepoints)
            arr=np.arange(n_changepoints)
            #print(arr<act_number)
            mult = (arr<act_number) *1 #[arr<act_number]=1
            offsets=offsets*mult


            #s = np.linspace(0, changepoint_range * np.max(x), n_changepoints) # maybe this needs to be changed                                                                      # assumes uniform changepoint distribution
            #s = pm.Uniform('positions', lower=xmin, upper=xmax, shape=int(n_changepoints)) 
            mu_pos=pm.Uniform('mu_pos', lower=xmin, upper=xmax) 

            s = pm.Normal('positions',  mu=mu_pos, sigma=5, shape=n_changepoints)

            A = (x[:, None] >= s) * 1
            offset_change = det_dot(A, offsets)
            mu = offset_change + trend*x + offset     
            Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)
        with basic_model:
            # draw 500 posterior samples
            trace = pm.sample(500, tune=2000, cores=4)
    
        data=eval_trace(trace,std_data)
        
        
    return data


def normalize_timser(data,normalize=True,rmv_nan=False):
    """
    normalizes time-series for state-space model approach
    divide by std and shift starting point to zero    
    
    Parameter:
    
    data: xr with monthly resolution
    
    returns: x,y,norm_factor
    
    """
    # remove nans from start and end
    df=pd.Series(data.values.squeeze(),index=data.time.values)
    first_idx = df.first_valid_index()
    last_idx = df.last_valid_index()
    print('first: ', first_idx,' last: ', last_idx)
    series=df.loc[first_idx:last_idx]
    

    # normalize
    if normalize:
        y = series.values
        first_data = y[0]
        std_data = np.nanstd(y)
        y = (y - first_data) / std_data
    else:
        std_data=1.
    index=np.arange(len(y))
    x=index/12.    
    # remove all nans
    if rmv_nan:
        y_notnan=~np.isnan(y)
        x=x[y_notnan]
        y=y[y_notnan]
        index=index[y_notnan]
    return x,y,index,std_data

def annual_cycle(x,a,b,c,d):#,sigma):
#    year=pd.Timedelta('365.2422 days')
#    f=year.total_seconds()    
    #noise=annual_cycle(x,a,b,c,d,sigma)
    return  a + b*x + c*np.cos(2*np.pi*x) + d*np.sin(2*np.pi*x)# + noise

def annual_cycle_semi(x,a,b,c,d,e,f):#,sigma):
#    year=pd.Timedelta('365.2422 days')
#    f=year.total_seconds()    
    #noise=annual_cycle(x,a,b,c,d,sigma)
    return  a + b*x + c*np.cos(2*np.pi*x) + d*np.sin(2*np.pi*x) + e*np.cos(4*np.pi*x) + f*np.sin(4*np.pi*x)# + noise
def show_trace(trace):
    az.plot_trace(trace)
    print(pm.summary(trace).round(7))
def det_dot(a, b):
    """
    The theano dot product and NUTS sampler don't work with large matrices?
    
    :param a: (np matrix)
    :param b: (theano vector)
    """
    return (a * b[None, :]).sum(axis=-1)