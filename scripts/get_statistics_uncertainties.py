"""
calc all statistics witth hector

"""

from sealeveltools.tests.test_functions import *
from sealeveltools.tests.load_test_files import *

def get_noise_model(name='100_rem_res_new_flag_grids',save=True):
    """
    estimates different 
    
    """

    models=['AR1', 'FNWN', 'PLWN', 'GGMWN']
    all_=[]
    for mod in models:
        print(mod)
        baltic=sl(load_sat(name=name)['ssh'][:,0:2,0:2])
        ds=baltic
        out=ds.trend(hector=True,model=mod)

        data=out.data
        data.to_netcdf('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/'+mod+'.nc')
        all_.append(data)

    i=0
    AIC=[]
    BIC=[]
    for mod in models:    
        AIC.append(all_[i]['AIC'].values.flatten())
        BIC.append(all_[i]['BIC'].values.flatten())
        i=i+1
    pd.DataFrame(AIC).to_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC')
    pd.DataFrame(BIC).to_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/BIC')
    
    
if __name__ == "__main__":
    get_noise_model(name='100_rem_res_new_flag_grids',save=True)
       
    
