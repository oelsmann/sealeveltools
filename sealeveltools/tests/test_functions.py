from sealeveltools.sl_class import *
from sealeveltools.tests.load_test_files import *

### these functions should be added to baltic_plus_functions



def plot_local_maps(outlier_out=False,save=False,remap=False):
    """
    plot maps and series
    
    """
    
    
    GIA=make_GIA()['trend']

    #GNSS=sl(make_gps()['trend'])

    #

    x_abs=40


    i=3
    name='AVISO'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    BALTIC=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+'baltic'+mod+'_mapped_on_AVISO_outl'+str(i)+'sigma.nc'))
    if remap:
        add2='rmp'
        BALTIC=BALTIC.remap(grid_size=.25)
    else:
        add2=''
    AVISOr=load_sat(name='aviso_gridded')
    name2='100_rem_res_new_flag_grids'
    baltic=load_sat(name=name2)['ssh']
    start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
    ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()

    baltic=baltic.loc[dict(time=slice(start[0], ende[0]))]
    AVISOr=AVISOr.loc[dict(time=slice(start[0], ende[0]))]

    psmsl=load_tg()
    psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
    baltic,psl=sl(baltic).couple(psl)
    psl=psl.dropna(dim='x',how='all')

    limit=50.

    psl,GIA = psl.couple(GIA,limit=limit)

    print(psl)
    psl_trend=psl.fltr(mod2='detrend').trend()




    abs_sl=GIA+psl_trend*1000.

    GIA.data=GIA.data/1000
    GIA.data=GIA.data.rename('trend')
    GIA.data
    psl2=copy.deepcopy(psl)
    psl_time=psl2+GIA
    #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected, '+str(i)+'sigma)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle, '+str(i)+'sigma',
                                '_FillValue' : 'nan', 'units': 'm','info': ''}

    abs_sl.data=dat.dropna(dim='x',how='all') 

    unc=BALTIC.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1, '+str(i)+'sigma',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    name='BALTIC'
    mapped='mapped_on_AVISO.nc'

    #plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc],[BALTIC.data]],var=['trend','no_var','AC','AC','no_var','counter'],extend=[8,31,53.5,66],ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5],[0.5,1.]],msize=[0.3,200.],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+mod+mapped[:-3]+str(i)+'sigma+_TG_')   




    if i==3:
        print('now')
        B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
        index_standard=(B_cop*1000. - abs_cop).data.index.values
        vals=(B_cop*1000. - abs_cop).data.values
        index_standard=index_standard[~np.isnan(vals)]             
        indices=abs_sl.data[vals>10].index.values
        abs_cop.data=abs_cop.data.dropna(dim='x',how='all') 
        print(len(index_standard))
    if outlier_out:
        domain = ((~np.isin(abs_sl.data.coords["index"].values,indices)))
        abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')

        domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
        abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')



        #plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc],[BALTIC.data]],var=['trend','no_var','AC','AC','no_var','counter'],extend=[8,31,53.5,66],ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5],[0.5,1.]],msize=[0.3,200.],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+mod+mapped[:-3]+str(i)+'sigma+_TG_')   
        B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
        vals=(B_cop*1000. - abs_cop).data.values
    else:

        domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
        abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')

        B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
        vals_ba=(B_cop*1000. - abs_cop).data


    #RMS=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
    #med=np.round(np.nanmedian(vals),3)
    #count=np.count_nonzero(~np.isnan(vals))
    #ax=plt.subplot(yp,xp,i)  
    #ax.hist(vals,bins=20)
    #ax.text(0.8,0.5,'RMS: '+str(RMS),horizontalalignment='center',
    #  verticalalignment='center', transform=ax.transAxes)
    #ax.text(0.8,0.4,'Median: '+str(med),horizontalalignment='center',
    #  verticalalignment='center', transform=ax.transAxes)
    #ax.text(0.8,0.3,'count: '+str(count),horizontalalignment='center',
    #  verticalalignment='center', transform=ax.transAxes)    
    #ax.set_xlim(x_min,x_max)

    #ax.set_xlabel('diff SAT-TG [mm/year]')
    #if outlier_out:
    #    ax.set_title('BALTIC '+str(i)+'-sigma no_outlier')        
    #else:

    #    ax.set_title('BALTIC '+str(i)+'-sigma')

    i=4

    name='AVISO'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))

    bltc2=sl(AVISO.data['trend'])

    latnew=bltc2.flat(kind='lat')
    lonnew=bltc2.flat(kind='lon')

    data=bltc2.flat(kind='values')
    var='trend'

    print(latnew.shape)

    ds2 = xr.Dataset({var: (['x'],  data)},
                         coords={'lon': (['x'], lonnew),
                        'lat': (['x'], latnew)}) 

    ds2.attrs=bltc.data.attrs

    ds2=ds2.dropna(dim='x',how='all')    
    AVISO=sl(ds)
    AVISO

    domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
    abs_sl.data = xr.where(domain, abs_cop.data, np.nan).dropna(dim='x',how='all')


    print('aviso')
    A_cop,abs_cop=sl(AVISO.data['trend']).couple(abs_sl,limit=50.)
    vals_av=(A_cop*1000. - abs_cop).data

    #RMS=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
    #med=np.round(np.nanmedian(vals),3)
    #count=np.count_nonzero(~np.isnan(vals))
    #ax=plt.subplot(yp,xp,i)  
    #ax.hist(vals,bins=20)
    #ax.text(0.8,0.5,'RMS: '+str(RMS),horizontalalignment='center',
    #  verticalalignment='center', transform=ax.transAxes)
    #ax.text(0.8,0.4,'Median: '+str(med),horizontalalignment='center',
    #  verticalalignment='center', transform=ax.transAxes)
    #ax.text(0.8,0.3,'count: '+str(count),horizontalalignment='center',
    #  verticalalignment='center', transform=ax.transAxes)    
    #ax.set_xlim(x_min,x_max)

    #ax.set_xlabel('diff SAT-TG [mm/year]')
    #if outlier_out:
    #    ax.set_title('AVISO '+str(i)+'-sigma no_outlier')  
    #    add='outl_out'
    #else:
    #    add='all'
    #    ax.set_title('AVISO '+str(i)+'-sigma')

    #if save:
    #    plt.savefig('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AVISO_vs_BALTIC'+str(x_abs)+add+add2)


    index_aviso_nan=vals_av[np.isnan(vals_av)].index.values
    index_aviso_better=vals_ba[abs(vals_av)<abs(vals_ba)].index.values


    for index in index_aviso_better:

        domain = ((np.isin(abs_sl.data.coords["index"].values,index)))
        abs_new = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')/1000.

        lon=abs_new.lon.values[0]
        lat=abs_new.lat.values[0]

        domain = ((np.isin(vals_av.coords["index"].values,index)))    
        vals_av2 =xr.where(domain, vals_av, np.nan).dropna(dim='x',how='all').values[0]
        domain = ((np.isin(vals_ba.coords["index"].values,index)))    
        vals_ba2 =xr.where(domain, vals_ba, np.nan).dropna(dim='x',how='all').values[0]

        print(vals_av2)
        print(vals_ba2)    
        #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

        dat=copy.deepcopy(abs_new)
        dat2=copy.deepcopy(abs_new)

        dat.attrs={'standard_name': 'trend', 'long_name': 'BALTIC SAT and TG trends (diff in mm: '+str(np.round(vals_ba2,4))+')',
                                    '_FillValue' : 'nan', 'units': 'mm/year','info': ''}

        dat2.attrs={'standard_name': 'trend', 'long_name': 'AVISO SAT and TG trends (diff in mm: '+str(np.round(vals_av2,4))+')',
                                    '_FillValue' : 'nan', 'units': 'mm/year','info': ''}

        abs_new1=dat.dropna(dim='x',how='all') 
        abs_new2=dat2.dropna(dim='x',how='all') 


        unc=BALTIC.data['trend_un AR1']*1000.
        unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1, '+str(i)+'sigma',
                                    '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        name='BALTIC'
        mapped='mapped_on_AVISO.nc'




        plt_xr_map([[BALTIC.data,abs_new1],[AVISO.data,abs_new2]],var=['trend','no_var','trend','no_var'],extend=[lon-1.,lon+1.,lat-.5,lat+.5],
                       ranges=[[0.0025,0.006],[0.0025,0.006],[0.0025,0.006],[0.0025,0.006]],msize=[50,600.,50.,600.],edgecolors=[None,'k',None,'k'],cmap='rainbow',
                       save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/TG_zoom',
                       save_name='AVISO_vs_BALTIC_sigma3_'+str(index))    

    i=3
    name='AVISO'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    BALTIC=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+'baltic'+mod+'_mapped_on_AVISO_outl'+str(i)+'sigma.nc'))
    if remap:
        add2='rmp'
        BALTIC=BALTIC.remap(grid_size=.25)
    else:
        add2=''
    AVISOr=load_sat(name='aviso_gridded')
    name2='100_rem_res_new_flag_grids'
    baltic2=load_sat(name=name2)['ssh']
    start=pd.DataFrame([baltic2.time[0].values,AVISOr.time[0].values]).max()
    ende=pd.DataFrame([baltic2.time[-1].values,AVISOr.time[-1].values]).min()

    baltic2=baltic2.loc[dict(time=slice(start[0], ende[0]))]
    AVISOr=AVISOr.loc[dict(time=slice(start[0], ende[0]))]

    psmsl=load_tg()
    psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
    AVISOr,psl=sl(AVISOr).couple(psl)
    psl=psl.dropna(dim='x',how='all')

    for index in index_aviso_better:

        xp=2
        yp=1
        size=4.5   
        x_max=x_abs
        x_min=-x_abs
        fig =plt.figure(figsize=(xp*size*1.3, yp*size))

        domain = ((np.isin(psl_time.data.coords["index"].values,index)))    
        series_tg=psl_time.data[:,domain]

        domain = ((np.isin(baltic.data.coords["index"].values,index)))    
        series_baltic=baltic.data[:,domain]

        domain = ((np.isin(AVISOr.data.coords["index"].values,index)))    
        series_aviso=AVISOr.data[:,domain]


        series_aviso=series_aviso-series_aviso.mean()
        series_baltic=series_baltic-series_baltic.mean()
        series_tg=series_tg-series_tg.mean()
        ax=plt.subplot(yp,xp,1)  

        ax.text(0.8,0.5,'',horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes)

        series_tg.plot(ax=ax,label='tg')
        series_baltic.plot(ax=ax,label='baltic')
        ax.legend()

        ax=plt.subplot(yp,xp,2)  

        ax.text(0.8,0.5,'',horizontalalignment='center',
        verticalalignment='center', transform=ax.transAxes)
        series_tg.plot(ax=ax,label='tg')    
        series_aviso.plot(ax=ax,label='aviso')    

        ax.legend()

        save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/TG_zoom'
        save_name='/AVISO_vs_BALTIC_sigma3_'+str(index)+'series'
        if save:
            plt.savefig(save_dir+save_name)
            
            

def psmsl_gia_long_term_timeseries():
    """
    compute long term time series for 4 regions 100 years + 
    GIA corrected for bayesian analysis
    
    """
    
    
    psl=load_tg(kind='psmsl_full')


    psl

    GIA_trend=make_GIA()['trend']
    GIA_trend_un=make_GIA()['trend_un']

    name2='100_rem_res_new_flag_grids'
    baltic=load_sat(name=name2)['ssh']

    baltic,psl=sl(baltic).couple(psl)
    psl=psl.dropna(dim='x',how='all')

    limit=50.


    psl_o,GIA_trend = psl.couple(GIA_trend,limit=limit)

    names=['S-W Baltic Sea', 'Central / Gotland Basin', 'Gulf of Finland', 'N Baltic Sea']
    data=psl_o.data

    domt=data.time>np.datetime64('1880-01-01')
    data=data.where(domt,drop=True)

    sections=[[9,16,53,60],
             [16,22,53,60],
             [22.5,30,58.5,62],
             [16,24,60,67]]
    i=0
    domains=[]
    for basin in names:

        domain = (
         ((data.coords["lat"] > sections[i][2]) &
          (data.coords["lat"] < sections[i][3]) &
         (data.coords["lon"] > sections[i][0]) &
         (data.coords["lon"] < sections[i][1]) )
        )    


        domains.append(data.where(domain,drop=True))
        i=i+1

    best=[]
    GIAf=[]
    for doo in domains:
        dd=sl(doo).yearmean().trend(trend_only=True)
        GIA_trend=make_GIA()['trend']


        dd,GIA_trendc=sl(dd).couple(GIA_trend,limit=limit)

        flt=doo.count(dim='time')==doo.count(dim='time').max()

        gi=GIA_trendc.where(flt,drop=True)
        new=doo.where(flt,drop=True)
        print(new.shape)
        if new.shape[1]>1:
            print(gi)
            new=new[:,0:1]
            print(gi.data.shape)
            gi=sl(gi.data[0])

        vec=np.empty(new.shape)
        print(gi.data.values)

        print(((int(len(new)/2))*gi.data.values/(1000*12)))
        vec[:,0]=((np.arange(len(new))-int(len(new)/2))*gi.data.values/(1000*12))
        out=(new+vec)


        best.append(out)
        GIAf.append(gi)
    all_=xr.concat(best,dim='x')
    all_.to_netcdf('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_tg/long_term_tg.nc')            

def compare_abs_trend_performance_SAT_TG(x_abs=40,outlier_out=True,save=False,remap=False):
    """
    compare AVISO vs BALTIC+ trends and performance (RMS ...)
    
    """

    all_indices=[]

    xp=4
    yp=1
    size=4.5   
    x_max=x_abs
    x_min=-x_abs
    fig =plt.figure(figsize=(xp*size*1.3, yp*size))

    for i in [1,2,3]:    

        #i=3
        # loop through different sigmas


        GIA=make_GIA()['trend']

        #GNSS=sl(make_gps()['trend'])

        #
        name='AVISO'
        mod='AR1'
        mapped='_mapped_on_baltic.nc'

        BALTIC=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+'baltic'+mod+'_mapped_on_AVISO_outl'+str(i)+'sigma.nc'))

        bltc=sl(BALTIC.data['trend'])

        latnew=bltc.flat(kind='lat')
        lonnew=bltc.flat(kind='lon')

        data=bltc.flat(kind='values')
        var='trend'

        print(latnew.shape)

        ds = xr.Dataset({var: (['x'],  data)},
                             coords={'lon': (['x'], lonnew),
                            'lat': (['x'], latnew)}) 

        ds.attrs=bltc.data.attrs

        ds=ds.dropna(dim='x',how='all')    
        BALTIC=sl(ds)    

        if remap:
            add2='rmp'
            BALTIC=BALTIC.remap(grid_size=.25)
        else:
            add2=''

        AVISOr=load_sat(name='aviso_gridded')
        name2='100_rem_res_new_flag_grids'
        baltic=load_sat(name=name2)['ssh']
        start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
        ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()

        psmsl=load_tg()
        psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
        baltic,psl=sl(baltic).couple(psl)
        psl=psl.dropna(dim='x',how='all')

        limit=50.

        psl,GIA = psl.couple(GIA,limit=limit)
        psl_trend=psl.fltr(mod2='detrend').trend()
        abs_sl=GIA+psl_trend*1000.
        #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

        dat=abs_sl.data
        dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected, '+str(i)+'sigma)',
                                    '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle, '+str(i)+'sigma',
                                    '_FillValue' : 'nan', 'units': 'm','info': ''}

        abs_sl.data=dat.dropna(dim='x',how='all') 

        #unc=BALTIC.data['trend_un AR1']*1000.
        #unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1, '+str(i)+'sigma',
        #                            '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        name='BALTIC'
        mapped='mapped_on_AVISO.nc'

        #plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc],[BALTIC.data]],var=['trend','no_var','AC','AC','no_var','counter'],extend=[8,31,53.5,66],ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5],[0.5,1.]],msize=[0.3,200.],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+mod+mapped[:-3]+str(i)+'sigma+_TG_')   




        if i==1:
            B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
            index_standard=(B_cop*1000. - abs_cop).data.index.values
            vals=(B_cop*1000. - abs_cop).data.values
            index_standard=index_standard[~np.isnan(vals)]   

            index_standard=index_standard[~np.isin(index_standard,[544,563,572])]

            indices=abs_sl.data[vals>10].index.values
            abs_cop.data=abs_cop.data.dropna(dim='x',how='all') 
            print(len(index_standard))
        if outlier_out:
            domain = ((~np.isin(abs_sl.data.coords["index"].values,indices)))
            abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')

            domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
            abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')



            #plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc],[BALTIC.data]],var=['trend','no_var','AC','AC','no_var','counter'],extend=[8,31,53.5,66],ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5],[0.5,1.]],msize=[0.3,200.],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+mod+mapped[:-3]+str(i)+'sigma+_TG_')   
            B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
            vals=(B_cop*1000. - abs_cop).data.values
        else:

            domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
            abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')

            B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
            vals=(B_cop*1000. - abs_cop).data.values


        RMS=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
        med=np.round(np.nanmedian(vals),3)
        count=np.count_nonzero(~np.isnan(vals))
        ax=plt.subplot(yp,xp,i)  
        ax.hist(vals,bins=20)
        ax.text(0.8,0.5,'RMS: '+str(RMS),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8,0.4,'Median: '+str(med),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8,0.3,'count: '+str(count),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)    
        ax.set_xlim(x_min,x_max)

        ax.set_xlabel('diff SAT-TG [mm/year]')
        if outlier_out:
            ax.set_title('BALTIC '+str(i)+'-sigma no_outlier')        
        else:

            ax.set_title('BALTIC '+str(i)+'-sigma')

    i=4

    name='AVISO'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))


    bltc2=sl(AVISO.data['trend'])

    latnew=bltc2.flat(kind='lat')
    lonnew=bltc2.flat(kind='lon')

    data=bltc2.flat(kind='values')
    var='trend'

    print(latnew.shape)

    ds2 = xr.Dataset({var: (['x'],  data)},
                         coords={'lon': (['x'], lonnew),
                        'lat': (['x'], latnew)}) 

    ds2.attrs=bltc.data.attrs

    ds2=ds2.dropna(dim='x',how='all')    
    AVISO=sl(ds2)

    domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
    abs_sl.data = xr.where(domain, abs_cop.data, np.nan).dropna(dim='x',how='all')


    A_cop,abs_cop=sl(AVISO.data['trend']).couple(abs_sl,limit=50.)
    av_index=(A_cop*1000. - abs_cop).data
    vals=av_index.values

    RMS=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
    med=np.round(np.nanmedian(vals),3)
    count=np.count_nonzero(~np.isnan(vals))
    ax=plt.subplot(yp,xp,i)  
    ax.hist(vals,bins=20)
    ax.text(0.8,0.5,'RMS: '+str(RMS),horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes)
    ax.text(0.8,0.4,'Median: '+str(med),horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes)
    ax.text(0.8,0.3,'count: '+str(count),horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes)    
    ax.set_xlim(x_min,x_max)

    ax.set_xlabel('diff SAT-TG [mm/year]')
    if outlier_out:
        ax.set_title('AVISO '+str(i)+'-sigma no_outlier')  
        add='outl_out'
    else:
        add='all'
        ax.set_title('AVISO '+str(i)+'-sigma')

    if save:
        plt.savefig('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AVISO_vs_BALTIC'+str(x_abs)+add+add2)


def correlate_SAT_TG(detrend=False):
    

    for i in [1,2,3,4,5]:
        psmsl=load_tg()
        psl=psmsl['sla']

        AVISO=load_sat(name='aviso_gridded')
        name='100_rem_res_new_flag_grids'
        baltic=load_sat(name=name)
        start=pd.DataFrame([baltic.time[0].values,AVISO.time[0].values]).max()
        ende=pd.DataFrame([baltic.time[-1].values,AVISO.time[-1].values]).min()
        baltic=baltic.loc[dict(time=slice(start[0], ende[0]))]
        AVISO=AVISO.loc[dict(time=slice(start[0], ende[0]))]['sla']
        psl=psl.loc[dict(time=slice(start[0], ende[0]))]


        baltic_mean=baltic.mean(dim='time')
        baltic_std=baltic.std(dim='time')

        ss_std=baltic_mean['ssh_std'].values.flatten()

        #plt.hist(ss_std,bins=100)
        perc_m=np.percentile(ss_std,75)
        ss_std=baltic_std['ssh_std'].values.flatten()

        #plt.hist(ss_std,bins=100)
        perc_std=np.percentile(ss_std,75)
        #np.median(ss_std)

        domain_std = (
             (baltic['ssh_std'] < perc_m + i*perc_std)
         )

        if i==5:
            baltic = baltic['ssh']
            add='standard'
        else:
            baltic = xr.where(domain_std, baltic, np.nan)['ssh']
            add=str(i)+'_sigma'

        if detrend:
            baltic=sl(baltic).detrend()
            AVISO=sl(AVISO).detrend()
            psl=sl(psl).detrend()
            add=add+'detr'



        baltic_cor=sl(baltic).cor(sl(psl)) # enlarge scatter size

        aviso_cor=sl(AVISO).cor(sl(psl))

        dat=baltic_cor.data
        dat.attrs['long_name']='Baltic cor. TG (median: '+str(np.round(dat.median().values,2))+') '+add

        baltic_cor.data=dat
        dat2=aviso_cor.data
        dat2.attrs['long_name']='AVISO cor. TG (median: '+str(np.round(dat2.median().values,2))+')'
        aviso_cor.data=dat2



        plt_xr_map([[baltic_cor.data],[aviso_cor.data]],var=['no_var','no_var'],extend=[8,31,53.5,66],
                       ranges=[[0.5,1.],[0.5,1.]],msize=[200.,200.],cmap='rainbow',edgecolors=['k','k'],
                       save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
                       save_name='baltic_vs_aviso_correlations_'+add)

    
    
    for i in [1,2,3,4,5]:
        psmsl=load_tg()
        psl=psmsl['sla']

        AVISO=load_sat(name='slcci_gridded')
        name='100_rem_res_new_flag_grids'
        baltic=load_sat(name=name)
        start=pd.DataFrame([baltic.time[0].values,AVISO.time[0].values]).max()
        ende=pd.DataFrame([baltic.time[-1].values,AVISO.time[-1].values]).min()
        baltic=baltic.loc[dict(time=slice(start[0], ende[0]))]
        AVISO=AVISO.loc[dict(time=slice(start[0], ende[0]))]['sla']
        psl=psl.loc[dict(time=slice(start[0], ende[0]))]


        baltic_mean=baltic.mean(dim='time')
        baltic_std=baltic.std(dim='time')

        ss_std=baltic_mean['ssh_std'].values.flatten()

        #plt.hist(ss_std,bins=100)
        perc_m=np.percentile(ss_std,75)
        ss_std=baltic_std['ssh_std'].values.flatten()

        #plt.hist(ss_std,bins=100)
        perc_std=np.percentile(ss_std,75)
        #np.median(ss_std)

        domain_std = (
             (baltic['ssh_std'] < perc_m + i*perc_std)
         )

        if i==5:
            baltic = baltic['ssh']
            add='standard'
        else:
            baltic = xr.where(domain_std, baltic, np.nan)['ssh']
            add=str(i)+'_sigma'

        if detrend:
            baltic=sl(baltic).detrend()
            AVISO=sl(AVISO).detrend()
            psl=sl(psl).detrend()
            add=add+'detr'



        baltic_cor=sl(baltic).cor(sl(psl)) # enlarge scatter size

        aviso_cor=sl(AVISO.shift({'time':1})).cor(sl(psl))

        dat=baltic_cor.data
        dat.attrs['long_name']='Baltic cor. TG (median: '+str(np.round(dat.median().values,2))+') '+add

        baltic_cor.data=dat
        dat2=aviso_cor.data
        dat2.attrs['long_name']='SLCCI cor. TG (median: '+str(np.round(dat2.median().values,2))+')'
        aviso_cor.data=dat2
        print(aviso_cor.data)


        plt_xr_map([[baltic_cor.data],[aviso_cor.data]],var=['no_var','no_var'],extend=[8,31,53.5,66],
                       ranges=[[0.5,1.],[0.5,1.]],msize=[200.,200.],cmap='rainbow',edgecolors=['k','k'],
                       save=save,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
                       save_name='baltic_vs_slcci_correlations_'+add)


def plot_uncertainty_statistics():
    """
    visualize 
    
    """

    AIC=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC_all')
    BIC=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/BIC_all')

    AIC.idxmin().hist(label='AIC')
    BIC.idxmin().hist(label='BIC')
    plt.legend()


def plot_trend_and_uncertainties_tgs():
    """
    plot trend: SAT vs TG (corrected for GIA or GNSS)
    plot AC: SAT vs TG 
    
    """
    
    # GNSS 50km
    mod='AR1WN'
    baltic_trend=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/'+mod+'.nc'))

    name='100_final_grid_'+mod
    GNSS=sl(make_gps()['trend'])
    psmsl=load_tg()
    psl=sl(psmsl['sla'])
    name='100_rem_res_new_flag_grids'
    baltic=sl(load_sat(name=name)['ssh']) 

    baltic,psl=baltic.couple(psl)
    psl=psl.dropna(dim='x',how='all')
    limit=50.

    psl,GNSS = psl.couple(GNSS,limit=limit)

    abs_sl=GNSS+psl.fltr(mod2='').trend()*1000.
    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GNSS closest, 50km)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    abs_sl.data=dat.dropna(dim='x',how='all')

    plt_xr_map([baltic_trend.data['trend']*1000,abs_sl.data],var='no_var',msize=[0.3,200.],ranges=[[1.5,6.],[1.5,6.]],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_tg',
               save_name='100_grid_vs_TGfltr_plus_GNSS50km_prelim')

    
    # GNSS 50km median
    mod='AR1WN'
    baltic_trend=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/'+mod+'.nc'))

    name='100_final_grid_'+mod
    GNSS=sl(make_gps()['trend'])
    psmsl=load_tg()
    psl=sl(psmsl['sla'])
    name='100_rem_res_new_flag_grids'
    baltic=sl(load_sat(name=name)['ssh']) 

    baltic,psl=baltic.couple(psl)
    psl=psl.dropna(dim='x',how='all')
    limit=50.

    #psl,GNSS = psl.couple(GNSS,limit=limit)
    GNSS,psl=GNSS.couple(psl,s_method='median',limit=limit)
    abs_sl=GNSS+psl.fltr(mod2='').trend()*1000.
    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GNSS median, 50km)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    abs_sl.data=dat.dropna(dim='x',how='all')

    plt_xr_map([baltic_trend.data['trend']*1000,abs_sl.data],var='no_var',msize=[0.3,200.],ranges=[[1.5,6.],[1.5,6.]],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_tg',
               save_name='100_grid_vs_TGfltr_plus_GNSS50km_median_prelim')

    # GNSS 50km median
    mod='AR1WN'
    baltic_trend=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/'+mod+'.nc'))

    name='100_final_grid_'+mod
    GNSS=sl(make_gps()['trend'])
    psmsl=load_tg()
    psl=sl(psmsl['sla'])
    name='100_rem_res_new_flag_grids'
    baltic=sl(load_sat(name=name)['ssh']) 

    baltic,psl=baltic.couple(psl)
    psl=psl.dropna(dim='x',how='all')
    limit=100.

    #psl,GNSS = psl.couple(GNSS,limit=limit)
    GNSS,psl=GNSS.couple(psl,s_method='median',limit=limit)
    abs_sl=GNSS+psl.fltr(mod2='').trend()*1000.
    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GNSS median, 100km)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    abs_sl.data=dat.dropna(dim='x',how='all')

    plt_xr_map([baltic_trend.data['trend']*1000,abs_sl.data],var='no_var',msize=[0.3,200.],ranges=[[1.5,6.],[1.5,6.]],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_tg',
               save_name='100_grid_vs_TGfltr_plus_GNSS100km_median_prelim')

                
    
    
    # GIA
    GIA=make_GIA()['trend']
    psmsl=load_tg()
    psl=sl(psmsl['sla'])
    name='100_rem_res_new_flag_grids'
    baltic=sl(load_sat(name=name)['ssh']) 

    baltic,psl=baltic.couple(psl)
    psl=psl.dropna(dim='x',how='all')
    limit=50.

    psl,GIA = psl.couple(GIA,limit=limit)

    abs_sl=GIA+psl.fltr(mod2='').trend()*1000.

    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    abs_sl.data=dat.dropna(dim='x',how='all')
    plt_xr_map([baltic_trend.data['trend']*1000,abs_sl.data],var='no_var',msize=[0.3,200.],ranges=[[1.5,6.],[1.5,6.]],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_tg',
               save_name='100_grid_vs_TGfltr_plus_GIA_prelim')


    dat=psl.fltr(mod2='').trend()
    dat.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle',
                                '_FillValue' : 'nan', 'units': 'm','info': ''}

    plt_xr_map([baltic_trend.data['AC'],dat.data['AC']],var='no_var',msize=[0.3,200.],edgecolors=[None,'k'],save=False,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_tg',save_name='100_grid_vs_TGfltr_annual_cycle')


    
    
    

def plot_and_compare_data_cut_time(save=True,outlier_out=False):
    """
    cut timeseries
    plot performance against TGs
    """
    # start from 2001
    GIA=make_GIA()['trend']

    x_abs=10.

    xp=2
    yp=1
    size=4.5   
    x_max=x_abs
    x_min=-x_abs
    fig =plt.figure(figsize=(xp*size*1.3, yp*size))

    for i in [3]:
        psmsl=load_tg()
        psl=psmsl['sla']

        AVISO=load_sat(name='aviso_gridded')
        name='100_rem_res_new_flag_grids'
        baltic=load_sat(name=name)
        start=pd.DataFrame([baltic.time[0].values,AVISO.time[0].values]).max()
        start[0]=pd.Timestamp('2001-12-31 00:00:00')

        ende=pd.DataFrame([baltic.time[-1].values,AVISO.time[-1].values]).min()
        baltic=baltic.loc[dict(time=slice(start[0], ende[0]))]
        AVISO=AVISO.loc[dict(time=slice(start[0], ende[0]))]['sla']
        #psl=psl.loc[dict(time=slice(start[0], ende[0]))]


        baltic_mean=baltic.mean(dim='time')
        baltic_std=baltic.std(dim='time')

        ss_std=baltic_mean['ssh_std'].values.flatten()

        #plt.hist(ss_std,bins=100)
        perc_m=np.percentile(ss_std,75)
        ss_std=baltic_std['ssh_std'].values.flatten()

        #plt.hist(ss_std,bins=100)
        perc_std=np.percentile(ss_std,75)
        #np.median(ss_std)

        domain_std = (
             (baltic['ssh_std'] < perc_m + i*perc_std)
         )

        if i==5:
            baltic = baltic['ssh']
            add='standard'
        else:
            baltic = xr.where(domain_std, baltic, np.nan)['ssh']
            add=str(i)+'_sigma'

        BALTIC=sl(baltic).trend()
        AVISO=sl(AVISO).trend()


        dat=BALTIC.data
        dat.attrs['long_name']='Baltic Trend 2002+'

        BALTIC.data=dat
        dat2=AVISO.data
        dat2.attrs['long_name']='AVISO Trend 2002+'
        AVISO.data=dat2

        psl=psl.dropna(dim='x',how='all')
        #psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]

        limit=50.

        psl,GIA = sl(psl).couple(GIA,limit=limit)
        psl_trend=psl.fltr(mod2='detrend').trend()
        abs_sl=GIA+psl_trend*1000.
        #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

        dat=abs_sl.data
        dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected, '+str(i)+'sigma)',
                                    '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle, '+str(i)+'sigma',
                                    '_FillValue' : 'nan', 'units': 'm','info': ''}

        abs_sl.data=dat.dropna(dim='x',how='all') 




        if i==3:
            B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
            index_standard=(B_cop*1000. - abs_cop).data.index.values
            vals=(B_cop*1000. - abs_cop).data.values
            index_standard=index_standard[~np.isnan(vals)]             
            indices=abs_sl.data[vals>10].index.values
            abs_cop.data=abs_cop.data.dropna(dim='x',how='all') 
            print(len(index_standard))
        if outlier_out:
            domain = ((~np.isin(abs_sl.data.coords["index"].values,indices)))
            abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')

            domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
            abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')



            #plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc],[BALTIC.data]],var=['trend','no_var','AC','AC','no_var','counter'],extend=[8,31,53.5,66],ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5],[0.5,1.]],msize=[0.3,200.],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+mod+mapped[:-3]+str(i)+'sigma+_TG_')   
            B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
            vals=(B_cop*1000. - abs_cop).data.values
        else:

            domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
            abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')

            B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=50.)
            vals=(B_cop*1000. - abs_cop).data.values


        RMS=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
        med=np.round(np.nanmedian(vals),3)
        count=np.count_nonzero(~np.isnan(vals))
        ax=plt.subplot(yp,xp,1)  
        ax.hist(vals,bins=20)
        ax.text(0.8,0.5,'RMS: '+str(RMS),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8,0.4,'Median: '+str(med),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8,0.3,'count: '+str(count),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)    
        ax.set_xlim(x_min,x_max)

        ax.set_xlabel('diff SAT-TG [mm/year]')
        if outlier_out:
            ax.set_title('BALTIC '+str(i)+'-sigma no_outlier')        
        else:

            ax.set_title('BALTIC '+str(i)+'-sigma')

    i=2

    name='AVISO'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'


    domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
    abs_sl.data = xr.where(domain, abs_cop.data, np.nan).dropna(dim='x',how='all')


    A_cop,abs_cop=sl(AVISO.data['trend']).couple(abs_sl,limit=50.)
    vals=(A_cop*1000. - abs_cop).data.values

    RMS=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
    med=np.round(np.nanmedian(vals),3)
    count=np.count_nonzero(~np.isnan(vals))
    ax=plt.subplot(yp,xp,2)  
    ax.hist(vals,bins=20)
    ax.text(0.8,0.5,'RMS: '+str(RMS),horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes)
    ax.text(0.8,0.4,'Median: '+str(med),horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes)
    ax.text(0.8,0.3,'count: '+str(count),horizontalalignment='center',
      verticalalignment='center', transform=ax.transAxes)    
    ax.set_xlim(x_min,x_max)

    ax.set_xlabel('diff SAT-TG [mm/year]')
    if outlier_out:
        ax.set_title('AVISO')  
        add='outl_out'
    else:
        add='all'
        ax.set_title('AVISO')


    if save:
        plt.savefig('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AVISO_vs_BALTIC_sigma3_from_2001-12_psmsl_standard')    

    abs1=abs_sl.data/1000
    abs2=abs_sl.data/1000

    abs2.attrs['long_name']='AVISO linear trends (2001-12+)'
    abs1.attrs['long_name']='BALTIC linear trends (2001-12+)'


    plt_xr_map([[B_trend.data,abs1],[av_trend.data,abs2]],var=['trend','no_var','trend','no_var'],extend=[8,31,53.5,66],
                   ranges=[[0.0025,0.006],[0.0025,0.006],[0.0025,0.006],[0.0025,0.006]],msize=[.2,200.,10.,200.],edgecolors=[None,'k',None,'k'],cmap='rainbow',
                   save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics',
                   save_name='AVISO_vs_BALTIC_sigma3_from_2001-12_map_psmsl_standard')    

def plot_stats_north_sea():
    AIC_all=pd.read_csv('/home/oelsmann/Julius/Scripts/north_sea/eval/trend_statistics/AIC_arfimawn_arwn_north_sea_gridded_merged_grids02_05')
    BIC_all=pd.read_csv('/home/oelsmann/Julius/Scripts/north_sea/eval/trend_statistics/BIC_arfimawn_arwn_north_sea_gridded_merged_grids02_05')

    degrees = 35
    plt.figure(figsize=(8,6))
    AIC_all.idxmin().hist(alpha=0.5,align='left',label='AIC',bins=[0, 1,2,3,4,5,6 ])
    BIC_all.idxmin().hist(alpha=0.5,align='left',label='BIC',bins=[0, 1,2,3,4,5,6 ])

    #locs, labels = xticks()            # Get locations and labels

    #models=['AR(1)', 'ARFIMA(1,d,0)','AR(1)+WN','ARFIMA(1,d,0)+WN', 'FN+WN', 'PL+WN', 'GGM+WN',]
    models=['AR1(1)', 'PL+WN','FN+WN', 'GGM+WN','ARFIMA(1,d,0)']
    plt.xticks(np.linspace(0,5,6), models)#,'AR1WN_h','ARFWN_h']) 


    plt.xticks(rotation=degrees)
    plt.legend()
    plt.title('minimum counts')
    plt.tight_layout()
    plt.savefig('/home/oelsmann/Julius/Scripts/north_sea/plots/stats_uncertainty/AIC_BIC_minimum_count')
    plt.show()


    plt.figure(figsize=(8,6))
    meanAIC=AIC_all.mean(axis=1)
    #(meanAIC-meanAIC.mean()).plot.bar(alpha=0.5,label='AIC',color='b')

    meanBIC=BIC_all.mean(axis=1)
    #(meanBIC-meanBIC.mean()).plot.bar(alpha=0.5,label='BIC',color='orange')


    df = pd.DataFrame({'AIC': meanAIC-meanAIC.mean(),

                       'BIC': meanBIC-meanBIC.mean()})

    ax = df.plot.bar(rot=0)

    plt.legend()

    plt.xticks(np.linspace(0,5,6), models) 


    plt.xticks(rotation=degrees)
    plt.legend()
    plt.title('mean scores')
    plt.tight_layout()

    plt.savefig('/home/oelsmann/Julius/Scripts/north_sea/plots/stats_uncertainty/AIC_BIC_mean_score_anomaly')
    plt.show()




    plt.figure(figsize=(8,6))
    meanAIC=AIC_all.median(axis=1)
    #(meanAIC-meanAIC.mean()).plot.bar(alpha=0.5,label='AIC',color='b')

    meanBIC=BIC_all.median(axis=1)
    #(meanBIC-meanBIC.mean()).plot.bar(alpha=0.5,label='BIC',color='orange')


    df = pd.DataFrame({'AIC': meanAIC-meanAIC.median(),

                       'BIC': meanBIC-meanBIC.median()})

    ax = df.plot.bar(rot=0)

    plt.legend()

    plt.xticks(np.linspace(0,5,6), models) 


    plt.xticks(rotation=degrees)
    plt.legend()
    plt.title('median scores')
    plt.tight_layout()

    plt.savefig('/home/oelsmann/Julius/Scripts/north_sea/plots/stats_uncertainty/AIC_BIC_median_score_anomaly')
    plt.show()

    
def plot_BIC_AIC():
    """
    plot minimum counts and mean anomalies
    
    """
    
    degrees = 35
    AIC=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC')
    BIC=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/BIC')
    AIC2=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC_arfima')
    BIC2=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/BIC_arfima')
    AIC3=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC_arfimawn_arwn')
    BIC3=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/BIC_arfimawn_arwn')
    #AIC4=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC_arfimawn_arwn_hout')
    #BIC4=pd.read_csv('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/BIC_arfimawn_arwn_hout')
    

    
    #AIC_all=pd.concat([AIC,AIC2,AIC3,AIC4],ignore_index=True)
    #BIC_all=pd.concat([BIC,BIC2,BIC3,BIC4],ignore_index=True)

    AIC_all=pd.concat([AIC,AIC2,AIC3],ignore_index=True)
    BIC_all=pd.concat([BIC,BIC2,BIC3],ignore_index=True)
    
    AIC_all=pd.DataFrame(AIC_all.values[:,1:]).loc[[0,4,5,6,1,2,3],:].reset_index(drop=True)

    BIC_all=pd.DataFrame(BIC_all.values[:,1:]).loc[[0,4,5,6,1,2,3],:].reset_index(drop=True)


    
    #['AR1', 'FNWN', 'PLWN', 'GGMWN','ARFIMA','AR1WN','ARFIMAWN','AR1WN_h','ARFIMAWN_h']
    plt.figure(figsize=(8,6))
    AIC_all.idxmin().hist(alpha=0.5,align='left',label='AIC',bins=[0, 1,2,3,4,5,6 ])
    BIC_all.idxmin().hist(alpha=0.5,align='left',label='BIC',bins=[0, 1,2,3,4,5,6 ])

    #locs, labels = xticks()            # Get locations and labels
    
    models=['AR(1)', 'ARFIMA(1,d,0)','AR(1)+WN','ARFIMA(1,d,0)+WN', 'FN+WN', 'PL+WN', 'GGM+WN',]
    plt.xticks(np.linspace(0,7,8), models)#,'AR1WN_h','ARFWN_h']) 
    

    plt.xticks(rotation=degrees)
    plt.legend()
    plt.title('minimum counts')
    plt.tight_layout()
    plt.savefig('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC_BIC_minimum_count')
    plt.show()


    plt.figure(figsize=(8,6))
    meanAIC=AIC_all.mean(axis=1)
    #(meanAIC-meanAIC.mean()).plot.bar(alpha=0.5,label='AIC',color='b')

    meanBIC=BIC_all.mean(axis=1)
    #(meanBIC-meanBIC.mean()).plot.bar(alpha=0.5,label='BIC',color='orange')


    df = pd.DataFrame({'AIC': meanAIC-meanAIC.mean(),

                       'BIC': meanBIC-meanBIC.mean()})

    ax = df.plot.bar(rot=0)

    plt.legend()

    plt.xticks(np.linspace(0,7,8), models) 


    plt.xticks(rotation=degrees)
    plt.legend()
    plt.title('mean scores')
    plt.tight_layout()

    plt.savefig('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC_BIC_mean_score_anomaly')
    plt.show()
    


    
    plt.figure(figsize=(8,6))
    meanAIC=AIC_all.median(axis=1)
    #(meanAIC-meanAIC.mean()).plot.bar(alpha=0.5,label='AIC',color='b')

    meanBIC=BIC_all.median(axis=1)
    #(meanBIC-meanBIC.mean()).plot.bar(alpha=0.5,label='BIC',color='orange')


    df = pd.DataFrame({'AIC': meanAIC-meanAIC.median(),

                       'BIC': meanBIC-meanBIC.median()})

    ax = df.plot.bar(rot=0)

    plt.legend()

    plt.xticks(np.linspace(0,7,8), models) 


    plt.xticks(rotation=degrees)
    plt.legend()
    plt.title('median scores')
    plt.tight_layout()

    plt.savefig('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/AIC_BIC_median_score_anomaly')
    plt.show()
    

   
def plot_compare_gridded_sets_tgs():
    """
    plot AVISO, SLCCI and BALTIC+ trend maps with TG and annual cycle
    
    """
        
    GIA=make_GIA()['trend']

    #GNSS=sl(make_gps()['trend'])

    #
    name='AVISO'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))
    AVISO=AVISO.remap(grid_size=.25)
    
    BALTIC=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+'balticAR1_mapped_on_AVISO.nc'))
    BALTIC=BALTIC.remap(grid_size=.25)
    
    AVISOr=load_sat(name='aviso_gridded')
    name2='100_rem_res_new_flag_grids'
    baltic=load_sat(name=name2)['ssh']
    start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
    ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()

    psmsl=load_tg()
    psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
    baltic,psl=sl(baltic).couple(psl)
    psl=psl.dropna(dim='x',how='all')

    limit=50.

    psl,GIA = psl.couple(GIA,limit=limit)
    psl_trend=psl.fltr(mod2='detrend').trend()
    abs_sl=GIA+psl_trend*1000.
    #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle',
                                '_FillValue' : 'nan', 'units': 'm','info': ''}

    abs_sl.data=dat.dropna(dim='x',how='all')
    unc=AVISO.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    plt_xr_map([[AVISO.data*1000,abs_sl.data],[AVISO.data,psl_trend.data],[unc]],var=['trend','no_var','AC','AC','no_var'],extend=[8,31,53.5,66],
               ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5]],msize=[0.3,200.],edgecolors=[None,'k'],
               save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
               save_name=name+mod+mapped[:-3]+'_TG_')   
    
    unc=BALTIC.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    name='BALTIC'
    mapped='mapped_on_AVISO.nc'
    
    plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc]],var=['trend','no_var','AC','AC','no_var'],extend=[8,31,53.5,66],
               ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5]],msize=[0.3,200.],edgecolors=[None,'k'],
               save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
               save_name=name+mod+mapped[:-3]+'_TG_')   

    GIA=make_GIA()['trend']

    #GNSS=sl(make_gps()['trend'])

    #
    name='SLCCI'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))
    
    BALTIC=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+'balticAR1_mapped_on_SLCCI.nc'))
    BALTIC=BALTIC.remap(grid_size=.25)
    
    AVISOr=load_sat(name='slcci_gridded')
    name2='100_rem_res_new_flag_grids'
    baltic=load_sat(name=name2)['ssh']
    start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
    ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()

    psmsl=load_tg()
    psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
    baltic,psl=sl(baltic).couple(psl)
    psl=psl.dropna(dim='x',how='all')

    limit=50.

    psl,GIA = psl.couple(GIA,limit=limit)
    psl_trend=psl.fltr(mod2='detrend').trend()
    abs_sl=GIA+psl_trend*1000.
    #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle',
                                '_FillValue' : 'nan', 'units': 'm','info': ''}

    abs_sl.data=dat.dropna(dim='x',how='all')
    unc=AVISO.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    plt_xr_map([[AVISO.data*1000,abs_sl.data],[AVISO.data,psl_trend.data],[unc]],var=['trend','no_var','AC','AC','no_var'],extend=[8,31,53.5,66],
               ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5]],msize=[0.3,200.],edgecolors=[None,'k'],
               save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
               save_name=name+mod+mapped[:-3]+'_TG_')   
    
    unc=BALTIC.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    name='BALTIC'
    mapped='mapped_on_SLCCI.nc'
    
    plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc]],var=['trend','no_var','AC','AC','no_var'],extend=[8,31,53.5,66],
               ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5]],msize=[0.3,200.],edgecolors=[None,'k'],
               save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
               save_name=name+mod+mapped[:-3]+'_TG_')   
    
    GIA=make_GIA()['trend']

    #GNSS=sl(make_gps()['trend'])

    #
    name='AVISO'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))
    AVISO=AVISO.remap(grid_size=.25)
    
    BALTIC=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+'balticAR1_mapped_on_AVISO.nc'))
    BALTIC=BALTIC.remap(grid_size=.25)
    
    AVISOr=load_sat(name='aviso_gridded')
    name2='100_rem_res_new_flag_grids'
    baltic=load_sat(name=name2)['ssh']
    start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
    ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()

    psmsl=load_tg()
    psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
    baltic,psl=sl(baltic).couple(psl)
    psl=psl.dropna(dim='x',how='all')

    limit=50.

    psl,GIA = psl.couple(GIA,limit=limit)
    psl_trend=psl.trend()
    abs_sl=GIA+psl_trend*1000.
    #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle',
                                '_FillValue' : 'nan', 'units': 'm','info': ''}

    abs_sl.data=dat.dropna(dim='x',how='all')
    unc=AVISO.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    plt_xr_map([[AVISO.data*1000,abs_sl.data],[AVISO.data,psl_trend.data],[unc]],var=['trend','no_var','AC','AC','no_var'],extend=[8,31,53.5,66],
               ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5]],msize=[0.3,200.],edgecolors=[None,'k'],
               save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
               save_name=name+mod+mapped[:-3]+'_TG_no_fltr')   
    
    unc=BALTIC.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    name='BALTIC'
    mapped='mapped_on_AVISO.nc'
    
    plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc]],var=['trend','no_var','AC','AC','no_var'],extend=[8,31,53.5,66],
               ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5]],msize=[0.3,200.],edgecolors=[None,'k'],
               save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
               save_name=name+mod+mapped[:-3]+'_TG_no_fltr')   

    GIA=make_GIA()['trend']

    #GNSS=sl(make_gps()['trend'])

    #
    name='SLCCI'
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))
    
    BALTIC=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+'balticAR1_mapped_on_SLCCI.nc'))
    BALTIC=BALTIC.remap(grid_size=.25)
    
    AVISOr=load_sat(name='slcci_gridded')
    name2='100_rem_res_new_flag_grids'
    baltic=load_sat(name=name2)['ssh']
    start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
    ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()

    psmsl=load_tg()
    psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
    baltic,psl=sl(baltic).couple(psl)
    psl=psl.dropna(dim='x',how='all')

    limit=50.

    psl,GIA = psl.couple(GIA,limit=limit)
    psl_trend=psl.trend()
    abs_sl=GIA+psl_trend*1000.
    #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected)',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle',
                                '_FillValue' : 'nan', 'units': 'm','info': ''}

    abs_sl.data=dat.dropna(dim='x',how='all')
    unc=AVISO.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    plt_xr_map([[AVISO.data*1000,abs_sl.data],[AVISO.data,psl_trend.data],[unc]],var=['trend','no_var','AC','AC','no_var'],extend=[8,31,53.5,66],
               ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5]],msize=[0.3,200.],edgecolors=[None,'k'],
               save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
               save_name=name+mod+mapped[:-3]+'_TG_no_fltr')   
    
    unc=BALTIC.data['trend_un AR1']*1000.
    unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1',
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    name='BALTIC'
    mapped='mapped_on_SLCCI.nc'
    
    plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc]],var=['trend','no_var','AC','AC','no_var'],extend=[8,31,53.5,66],
               ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5]],msize=[0.3,200.],edgecolors=[None,'k'],
               save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
               save_name=name+mod+mapped[:-3]+'_TG_no_fltr')   
        
    
   
def plot_compare_gridded_sets_tgs_outlier_sigma():
    """
    plot BALTIC+ trend maps with TG and annual cycle and sigma tests
    
    """
    for i in [1,2,3]:    
        GIA=make_GIA()['trend']

        #GNSS=sl(make_gps()['trend'])

        #
        name='AVISO'
        mod='AR1'
        mapped='_mapped_on_baltic.nc'

        BALTIC=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+'baltic'+mod+'_mapped_on_AVISO_outl'+str(i)+'sigma.nc'))
        BALTIC=BALTIC.remap(grid_size=.25)

        AVISOr=load_sat(name='aviso_gridded')
        name2='100_rem_res_new_flag_grids'
        baltic=load_sat(name=name2)['ssh']
        start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
        ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()

        psmsl=load_tg()
        psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
        baltic,psl=sl(baltic).couple(psl)
        psl=psl.dropna(dim='x',how='all')

        limit=50.

        psl,GIA = psl.couple(GIA,limit=limit)
        psl_trend=psl.fltr(mod2='detrend').trend()
        abs_sl=GIA+psl_trend*1000.
        #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

        dat=abs_sl.data
        dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected, '+str(i)+'sigma)',
                                    '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle, '+str(i)+'sigma',
                                    '_FillValue' : 'nan', 'units': 'm','info': ''}

        abs_sl.data=dat.dropna(dim='x',how='all') 

        unc=BALTIC.data['trend_un AR1']*1000.
        unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1, '+str(i)+'sigma',
                                    '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
        name='BALTIC'
        mapped='mapped_on_AVISO.nc'

        plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc],[BALTIC.data]],var=['trend','no_var','AC','AC','no_var','counter'],extend=[8,31,53.5,66],ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5],[0.5,1.]],msize=[0.3,200.],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+mod+mapped[:-3]+str(i)+'sigma+_TG_')   

  
    
def compare_gridded_sets(models=['AR1WN','ARFIMAWN'],mapped='.nc'):
    """
    compare 
    - baltic
    - AVISO
    - SLCCI
   
    
    """
    #mapped='_mapped_on_baltic.nc' #'.nc'
    sets=['AVISO','SLCCI']
    for mod in models:
        for name in sets:
            print(name)
            
            AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))
            if name=='AVISO':
                AVISO=AVISO.remap(grid_size=.25)
            
            AVISO.plot(var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]],msize=10,save=True,
                       save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+'_'+mod+mapped[:-3])
    
    if mapped=='_mapped_on_baltic.nc':
        for mod in models:
            for nname in ['SLCCI','AVISO']:
                
                add=''

                
                name='baltic'+mod+'_mapped_on_'+nname+'.nc'
                baltic_best=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name))

                
                baltic_best=baltic_best.remap(grid_size=.25)
                baltic_best.plot(var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]],save=True,
                           save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name[:-3])                

    else:
        
        for mod in models:

            #add='_hout'
            add=''
            baltic_best=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/trends_altimetry_statistics/'+mod+add+mapped))

            name='100_final_grid_'+mod+add
            baltic_best=baltic_best.remap(grid_size=.25)
            baltic_best.plot(var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]],save=True,
                       save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+'_'+mod+mapped[:-3])

            
            
            
            
            

def make_flag(baltic,var,threshold=1,howt='absolute',sign='gt'):
    baltic_mean=baltic.mean(dim='time')
    baltic_std=baltic.std(dim='time')
    
    print(var)
    print(threshold)
    print(howt)
    if var=='count_cut':
        domain= ((baltic['ssh'].count(dim='time') > threshold))    
        
    else:  
        if howt=='absolute':
            if sign=='gt':
                domain = (
                     (baltic[var] > threshold)
                 )      
            else:
                domain = (
                     (baltic[var] < threshold)
                 )  

        elif howt=='relative':


            if sign=='gt':

                ss_mean=baltic_mean[var].values.flatten()    
                perc_m=np.percentile(ss_mean,25)
                ss_std=baltic_std[var].values.flatten()
                perc_std=np.percentile(ss_std,25)
                domain = (
                     (baltic[var] > perc_m + threshold*perc_std)
                 )
            else:
                
                ss_mean=baltic_mean[var].values.flatten()    
                perc_m=np.percentile(ss_mean,75)
                ss_std=baltic_std[var].values.flatten()
                ss_std=ss_std[~np.isnan(ss_std)]
                perc_std=np.percentile(ss_std,75)
                print(var,perc_std,perc_m,threshold)
                
                domain = (
                     (baltic[var] < perc_m + threshold*perc_std)
                 ) 
        elif howt=='relative_perc':
            """
            percentile
            """

            if sign=='gt':
                perc=100-threshold
                ss_mean=baltic_mean[var].values.flatten()    
                perc_m=np.percentile(ss_mean,perc)
                domain = (
                     (baltic[var] > perc_m)
                 )
            else:
                
                ss_mean=baltic_mean[var].values.flatten()    
                perc_m=np.percentile(ss_mean,threshold)

                print(var,perc_m,threshold)
                
                domain = (
                     (baltic[var] < perc_m)
                 ) 
        elif howt=='trend_err':
            """
            trend_err only for north sea

            """
            trend_set=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/baltic_trend_standard.nc')
            
            if sign=='gt':
                domain = (trend_set.trend_un > threshold)
                
            else:
                domain = (trend_set.trend_un < threshold)

        elif howt=='trend_self':
            """
            trend_self only for north sea
            """
            trend_set=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/baltic_trend_standard.nc')
            
            if sign=='gt':
                domain = (abs(trend_set.trend)-trend_set.trend_un*threshold > 0.)       
            else:
                domain = (abs(trend_set.trend)-trend_set.trend_un*threshold > 0.)
                
                
                
        elif howt=='relative_perc_30km':
            """
            percentile only applied to first 50km
            only implemented for north sea
            """
            #dst=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/grid_coast_dist.nc')
            #(dst['dst'] < 50.)
            if sign=='gt':
                perc=100-threshold
                ss_mean=baltic_mean[var].values.flatten()    
                perc_m=np.percentile(ss_mean,perc)
                domain = (
                     (baltic[var] > perc_m)
                 )
            else:
                
                ss_mean=baltic_mean[var].values.flatten()    
                perc_m=np.percentile(ss_mean,threshold)

                print(var,perc_m,threshold)
                
                domain = (
                     (baltic[var] < perc_m)
                 )                 
                
        elif howt=='individual_relative':
            if sign=='gt':

                domain = (
                     (baltic[var] > baltic_mean[var] + threshold*baltic_std[var])
                 )
            else:
                domain = (
                     (baltic[var] < baltic_mean[var] + threshold*baltic_std[var])
                 )      
            
    return domain

def flag_trend_function(baltic,mod2='detrend',mod1=2,var_all=[],how='abs_std',threshold=[1],
                        howt=['absolute'],sign=['gt'],hector=False,s_model='AR1',semi=False,
                        mode='',set1_name='',monthly=False):
    """
    flag out some data according to var and range    
    
    mode: '' - standard
          'flag_file' returns flag file as well as flagged file
    """
    if 'north' in set1_name:
        basin='north'        
    else:
        basin='baltic'
    print('set1_name is ',set1_name)
    print(basin)
    
    h=0
    baltic=baltic.assign(ssh_std_prod=(baltic['ssh_std'])*np.sqrt(baltic['num_used_obs']))
    ds_std=baltic
    props=''
    dist_coast_only=False
    for var in var_all:
        if '0km' in howt[h]:

            dist_coast_only=True
            print('dist coast is true')
            
        domain_std=make_flag(ds_std,var,threshold=threshold[h],howt=howt[h],sign=sign[h])    
        ds_std = xr.where(domain_std, ds_std, np.nan)

        print(props+'_'+var+'_'+str(threshold[h]).replace('.', ''))
        props=props+'_'+var+'_'+str(threshold[h]).replace('.', '')
        h=h+1
        #dst=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/grid_coast_dist.nc')
        #(dst['dst'] < 50.)    
        
    if dist_coast_only:
        
        dst=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/grid_coast_dist.nc')
        domain = (dst['dst'] < 30.)         
        ds_std = xr.where(domain, ds_std, baltic)
        
    if basin=='baltic':    
        
        bltc=sl(ds_std['ssh'].transpose('time', 'x', 'y'))
        bll_backup=sl(ds_std['ssh'].transpose('time', 'x', 'y'))
    
    elif basin=='north':
        bltc=sl(ds_std['ssh'].transpose('time', 'x'))
        bll_backup=sl(ds_std['ssh'].transpose('time', 'x'))
            
        
    
    if mode=='':
        'other option saves flag file'
        
        if how=='no':
            print('no filter')
            fltr=props
            #
            BALTIC=bltc
            fltr=props+'_test_flag_field'
            flag_map=baltic['ssh'].isin(BALTIC.data)
            #BALTIC=flag_map*baltic['ssh'] 
            
            BALTIC=xr.where(flag_map,baltic['ssh'],np.nan)
            #
            BALTIC=sl(BALTIC).trend(hector=hector,model=s_model,monthly=monthly)
            
            #BALTIC=bltc.trend(hector=hector,model=s_model)
        else:
            BALTIC=bltc.fltr(how=how,mod2=mod2,mod1=mod1).trend(hector=hector,semi=semi,model=s_model,monthly=monthly)
            fltr=props+mod2+how+str(mod1)
        if hector:
            fltr=fltr+s_model
    else:
        fltr=props+mod2+how+str(mod1)
        if basin=='baltic':
            bltc.data.to_netcdf('/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/balticplus_gridded/monthly_grids_KO8/baltic_gridded_merged_'+set1_name+fltr+'_flagged.nc')
            
        else:
            bltc.data.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/north_gridded_merged_'+set1_name+fltr+'_flagged.nc')
                        
            
        BALTIC=bltc
        
        flag_map=baltic['ssh'].isin(BALTIC.data)
        #BALTIC=flag_map*baltic['ssh']
        #xr.where(flag_map,baltic['ssh'],np.nan)
        #print(BALTIC)
        BALTIC=flag_map
        BALTIC.attrs={'flag properties':props}
    return BALTIC,fltr



def cal_trend_and_return(set1_name='100_rem_res_new_flag_grids',save=False,set2_name='AVISO',mod2='detrend',mod1=2,
                how='no',x_abs=8,var_all=[],threshold=[1],howt=['absolute'],sign=['gt'],outlier_out=False,hector=False,mapped_on='AVISO',semi=False,mode='',monthly=False):
    """
    calculate TG and SAT trends and compare
    
    """
    
    if 'north' in set1_name:
        basin='north'
    else:
        basin='baltic'        
    
    all_indices=[]

    xp=2
    yp=1
    size=4.5   
    x_max=x_abs
    x_min=-x_abs
    fig =plt.figure(figsize=(xp*size*1.3, yp*size))


    GIA=make_GIA()['trend']
    name=set2_name
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    
    if set2_name=='AVISO':
        AVISOr=load_sat(name='aviso_gridded')
    elif set2_name=='SLCCI':
        AVISOr=load_sat(name='slcci_gridded')     
    name2=set1_name
    baltic=load_sat(name=name2)
    
    if mode=='flag_file':
        mapped_on=''
    if mapped_on=='':
        print('no map')
        mp_text=''
    else:
        start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
        ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()
        #start[0]=pd.Timestamp('2001-12-31 00:00:00')
        psmsl=load_tg()
        psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
        baltic=baltic.loc[dict(time=slice(start[0], ende[0]))]
        mp_text='_mapped_on_'+mapped_on
    BALTIC,fltr=flag_trend_function(baltic,mod2=mod2,mod1=mod1,var_all=var_all,how=how,threshold=threshold,
                        howt=howt,sign=sign,hector=hector,s_model='AR1',semi=semi,
                                    mode=mode,set1_name=set1_name,monthly=monthly)
    #print(BALTIC)
    if monthly:
        set1_name=set1_name+'monthly'
    if basin=='baltic':
    
        if mode=='flag_file':
            BALTIC.to_netcdf('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+set1_name+mod+mp_text+fltr+'flag_file.nc')  
            print('no')

        else:
            BALTIC.data.to_netcdf('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+set1_name+mod+mp_text+fltr+'.nc')   
            
    else:
        if mode=='flag_file':
            BALTIC.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/'+set1_name+mod+mp_text+fltr+'flag_file_30km.nc')  
            print('no')

        else:
            BALTIC.data.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/'+set1_name+mod+mp_text+fltr+'.nc')          
        
        
        
    return BALTIC
    

    
  
    
def calc_trends(set1_name='100_rem_res_new_flag_grids',save=False,set2_name='AVISO',mod2='detrend',mod1=2,
                how='no',x_abs=8,var_all=[],threshold=[1],howt=['absolute'],
                sign=['gt'],outlier_out=False,mapped_on='AVISO',monthly=False,gps=False):
    """
    calculate TG and SAT trends and compare
    
    """
    max_lon=360.
    add_n=''
    if 'north' in set1_name:
        basin='north'
        save_dir='/home/oelsmann/Julius/Scripts/north_sea/plots/trends_compare/maps'
        plt_name='DGFI'
        gps=False
        exclude=True
        max_lon=16.
        #version='ICE-6GD' # NKG2016
        version='Caron'        
         
        uncert=False    
        tg_data='bafg_psmsl'
        if gps:
            add_n='gps'
        else:
            add_n=version           
        if exclude:
            add_n=add_n+'exc'
        if uncert:
            add_n=add_n+'uncert' 
         
            
        add_n=add_n+str(int(max_lon))
        add_n=add_n+'Dang2014_fulltgs'+tg_data#'noDang2014tgs_eastern_original'
        
        extend=[-4,13,47,60]
    else:
        basin='baltic'
        save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis'
        plt_name='BALTIC+'  
        tg_data='psmsl'
        uncert=False
        extend=[8,31,53.5,66]
        
    
    all_indices=[]

    xp=2
    yp=1
    size=4.5   
    x_max=x_abs
    x_min=-x_abs
    fig =plt.figure(figsize=(xp*size*1.3, yp*size))

    if gps:
        GIA=load_gps(kind='ULR')['trend']
        GIA_trend_un=load_gps(kind='ULR')['trend_un']
        print('gps')
    else:
        
        if basin=='north':
            GIA=make_GIA(opt=version)['trend']   
            GIA_trend_un=make_GIA(opt=version)['trend_un']
        else:
            GIA=make_GIA()['trend']
            GIA_trend_un=make_GIA()['trend_un']
    
    
    name=set2_name
    mod='AR1'
    mapped='_mapped_on_'+basin+'.nc'

    if basin=='baltic':
        if set2_name=='AVISO':
            AVISOr=load_sat(name='aviso_gridded')
        elif set2_name=='SLCCI':
            AVISOr=load_sat(name='slcci_gridded')   

                
    else:
        if set2_name=='AVISO':
            AVISOr=load_sat(name='AVISO_north_sea')
        elif set2_name=='SLCCI':
            AVISOr=load_sat(name='SLcci_merged_north_sea')   


    name2=set1_name
    baltic=load_sat(name=name2)
    
    
    if mapped_on=='':
        print('no map')
        mapped=''
        psl=load_tg(kind=tg_data)
        psl=psl.where(psl.lon<max_lon).dropna(dim='x',how='all')
        
    else:
        start=pd.DataFrame([baltic.time[0].values,AVISOr.time[0].values]).max()
        ende=pd.DataFrame([baltic.time[-1].values,AVISOr.time[-1].values]).min()
        
        print(start)
        print(ende)
        #start[0]=pd.Timestamp('2001-12-31 00:00:00')
        psmsl=load_tg(kind=tg_data)
        psl=psmsl.loc[dict(time=slice(start[0], ende[0]))]
        
        baltic=baltic.loc[dict(time=slice(start[0], ende[0]))]
        mapped='_mapped_on_'+mapped_on    
        psl=psl.where(psl.lon<max_lon).dropna(dim='x',how='all')
    print(baltic)
    if basin=='northdd':
        BALTIC=sl(xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/grids/trends/trends_comparenorth_sea_gridded_merged_grids02_05AVISO_ssh_std_90_count_cut_240_test_flag_field.nc'))
        fltr='240_90rel_sshstd'
        
    else:
        
            
        BALTIC,fltr=flag_trend_function(baltic,mod2=mod2,mod1=mod1,var_all=var_all,how=how,threshold=threshold,
                howt=howt,sign=sign,monthly=monthly,set1_name=set1_name)
    
    if basin=='north':
        BALTIC.data.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/data/grids/trends/trends_compare'+set1_name+name+fltr+'.nc')
        
    
    BALTIC.plot(var='trend',ranges=[[0.0015,0.0035]],cmap='rainbow',
                       save=True,save_dir=save_dir,
                       save_name=set1_name+set2_name+fltr)
    if gps:
        fltr=fltr+'gps'
    
    bltc=sl(BALTIC.data['trend'])
    latnew=bltc.flat(kind='lat')
    lonnew=bltc.flat(kind='lon')
    data=bltc.flat(kind='values')
    var='trend'
    print(latnew.shape)
    ds = xr.Dataset({var: (['x'],  data)},
                         coords={'lon': (['x'], lonnew),
                        'lat': (['x'], latnew)}) 

    ds.attrs=bltc.data.attrs

    ds=ds.dropna(dim='x',how='all')    
    
    bltcu=sl(BALTIC.data['trend_un'])
    latnew=bltc.flat(kind='lat')
    lonnew=bltc.flat(kind='lon')
    data=bltc.flat(kind='values')
    var='trend_un'
    print(latnew.shape)
    dsu = xr.Dataset({var: (['x'],  data)},
                         coords={'lon': (['x'], lonnew),
                        'lat': (['x'], latnew)}) 

    dsu.attrs=bltc.data.attrs

    dsu=dsu.dropna(dim='x',how='all')      
    
    BALTICu=sl(dsu)
    BALTIC=sl(ds)  

    leng=len(BALTIC.data)
    
    baltic,psl=sl(baltic).couple(psl)
    psl=psl.dropna(dim='x',how='all')

    limit=150. # before 50.


    
    print('here baltic')
    
    if gps:
        limit=10.
        psl,GIA = psl.couple(GIA,limit=limit)
        psl_a,GIA_trend_un = psl.couple(GIA_trend_un,limit=limit)
    else:

        
        psl,GIA = psl.couple(GIA,limit=limit)        
        psl_a,GIA_trend_un = psl.couple(GIA_trend_un,limit=limit)
    
    
    #psl_trend=psl.fltr(mod2='detrend').trend()
    
    if uncert:
    
        psl_trend=psl.trend(monthly=monthly,hector=True)    
        

    else:
        psl_trend=psl.trend()        
    
    abs_sl=GIA+psl_trend*1000.
    
    if uncert:
        abs_sl_unc=np.sqrt(GIA_trend_un.data**2+(psl_trend.data['trend_un AR1']*1000.*2)**2)
        dasss=abs_sl.data
        abs_sl=sl(dasss.where((abs(dasss)-abs_sl_unc) > 0).dropna(dim='x',how='all') )
        
        
        
    
    #var=['trend','AC','trend_un','trend_un '+mod,],extend=[8,31,53.5,66],ranges=[[0.0025,0.0055],[0.055,0.095],[0.0015,0.0025],[0.000,0.002]]

    dat=abs_sl.data
    dat.attrs={'standard_name': 'trend', 'long_name': 'SAT and TG trends (GIA corrected, '+fltr,
                                '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    psl_trend.data['AC'].attrs={'standard_name': 'AC', 'long_name': 'SAT and TG annual cycle, '+fltr,
                                '_FillValue' : 'nan', 'units': 'm','info': ''}

    abs_sl.data=dat.dropna(dim='x',how='all') 

    #unc=BALTIC.data['trend_un AR1']*1000.
    #unc.attrs={'standard_name': 'trend_un AR1', 'long_name': 'trend uncertainties AR1, '+str(i)+'sigma',
    #                            '_FillValue' : 'nan', 'units': 'mm/year','info': ''}
    name='BALTIC'
    

    #plt_xr_map([[BALTIC.data*1000,abs_sl.data],[BALTIC.data,psl_trend.data],[unc],[BALTIC.data]],var=['trend','no_var','AC','AC','no_var','counter'],extend=[8,31,53.5,66],ranges=[[2.5,6.],[2.5,6.],[0.05,0.1],[0.05,0.1],[0.000,2.5],[0.5,1.]],msize=[0.3,200.],edgecolors=[None,'k'],save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',save_name=name+mod+mapped[:-3]+str(i)+'sigma+_TG_')   
    
    limit=150   # original (baltic) = 150
    print(BALTIC.data['trend'])
    print(abs_sl)
    B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=limit)
    
    
    
    index_standard=(B_cop*1000. - abs_cop).data.index.values
    
    vals=(B_cop*1000. - abs_cop).data.values
    index_standard=index_standard[~np.isnan(vals)]   

    if basin=='baltic':
    
        index_standard=index_standard[~np.isin(index_standard,[544,563,572])]

        arr=np.array([ 27,  41,  47,  63,  64,  69,  71,  74,  83, 111, 153, 165, 249,
         265, 275, 295, 341, 347, 353, 354, 355, 356, 357, 358, 359, 360,
         361, 362, 363, 364, 365, 384, 405, 407, 412, 418, 428, 431, 442,
         453, 519, 540, 542, 546, 560, 566, 567, 568, 583, 601, 602, 604,
         619])
        index_standard=index_standard[np.isin(index_standard,arr)]

        indices=abs_sl.data[vals>10].index.values
        
    else:
        print('set index')
        
        dang=np.array([344,437,406,461,376,429,387,404,541,569,544,602,422,481,520,554,190,475,592,37])
        
        eastern_channel=np.array([41,83,265,275,363,364,365,430]) # exclude frederikshavn
        
        arr=np.array([344,437,406,461,376,429,387,404,541,569,544,
                      602,422,481,520,554,190,475,592,37,41,83,
                      265,275,363,364,365,430])
        
        #dang=np.load('/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/TG_indices/Tg_mind80perc_data_lt_1_5mmyearerr.npy')
        
        arr = dang
        
        # both eastern plus dangendorf                      
        # tide gauge indices according to selection of Dangendorf et al. 2014
        # except Helgoland, Hörnum and Norderney
        
        abs_cop.data=abs_cop.data.dropna(dim='x',how='all')
                
        domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
        abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')

        av_index=recheck_indices(set2_name=set2_name,name=set2_name,basin=basin,abs_cop=abs_cop,
                                 abs_sl=abs_sl,index_standard=index_standard)
        index_standard=index_standard[np.isin(index_standard,av_index)]
        #index_standard=index_standard[np.isin(index_standard,arr)]
        if exclude:
            exclude_all = [77,271,301,419,437,180,528,406,404,413,344,194]
            index_standard = index_standard[~np.isin(index_standard,exclude_all)]
        ## check if different dataset has same tgs
        
        
    abs_cop.data=abs_cop.data.dropna(dim='x',how='all') 
    print(len(index_standard))


    domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
    abs_sl.data = xr.where(domain, abs_sl.data, np.nan).dropna(dim='x',how='all')

    B_cop,abs_cop=sl(BALTIC.data['trend']).couple(abs_sl,limit=limit)
    
    B_cop_trend_un,abs_copu=sl(BALTICu.data['trend_un']).couple(abs_sl,limit=limit)
    

    
    #abs_cop.data.to_netcdf('/home/oelsmann/Julius/Scripts/balticplus/dataset_lfs/psmsl/psmsl_baltic_absolute_trends.nc')
    
    ba_index=(B_cop*1000. - abs_cop).data
    vals=ba_index.values
    RMS_b=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
    med_b=np.round(np.nanmedian(vals),3)
    count_b=np.count_nonzero(~np.isnan(vals))
    if basin=='baltic':
        print('save netcdf')
        ba_index.to_netcdf('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/flag_tests/presentation/diff_tg_sat_first'+set1_name+set2_name+fltr+add_n)            


        
    ## plot again for presentation style    
    
    pres=True
    
    print('our dataset count: ',count_b)
    x_min=-8.
    x_max=8
    
    datt=abs_sl.data
    datt.attrs={'long_name':'linear trends','units':'mm/year'}
    plt_xr_map([[BALTIC.data*1000,datt]],
           var=['trend','no_var'],extend=extend,ranges=[[1.,4.]],
           msize=[2.,200.],edgecolors=[None,'k'],cmap='rainbow',
                       save=True,save_dir=save_dir,
                       save_name=set1_name+set2_name+fltr+add_n)   

    if basin=='north':
        ba_index.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/plots/trends_compare/diff_tg_sat_north'+set1_name+set2_name+fltr+add_n)
        
        abs_sl.data.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/plots/trends_compare/diff_maps_'+set1_name+set2_name+fltr+add_n)

        B_cop_trend_un.data.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/plots/trends_compare/diff_maps_uncert'+set1_name+set2_name+fltr+add_n)
        
    if pres:
        plt.figure(figsize=(8,5))

        ax=plt.subplot(yp,xp,1)  
        ax.hist(vals,bins=20)
        ax.text(0.6,0.8,'RMS: '+str(RMS_b),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.6,0.7,'Median: '+str(med_b),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.6,0.6,'count: '+str(count_b),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes) 
        
        ax.set_xlim(x_min,x_max)


        ax.set_xlabel('diff SAT-TG [mm/year]')
        ax.set_ylabel('counts')

        if outlier_out:
            ax.set_title(plt_name)        
        else:

            ax.set_title(plt_name)

        i=2

        name=set2_name
        mod='AR1'
        mapped='_mapped_on_baltic.nc'

        if basin=='baltic':

            AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))
        
        else:
            
            if set2_name=='AVISO':

                AVISO=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/aviso/'+'aviso_trend.nc')
                lon=AVISO.lon.values
                lon[lon>180]=lon[lon>180]-360

                AVISO.assign_coords(lon=("lon", lon))

                AVISO=sl(AVISO)
                
            elif set2_name=='SLCCI':
                AVISO=sl(xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/slcci/'+'slcci_trend.nc'))                
                    
            
        
        bltc2=sl(AVISO.data['trend'])

        latnew=bltc2.flat(kind='lat')
        lonnew=bltc2.flat(kind='lon')

        data=bltc2.flat(kind='values')
        var='trend'

        print(latnew.shape)
        limit=150. # new insert not in baltic
        ds2 = xr.Dataset({var: (['x'],  data)},
                             coords={'lon': (['x'], lonnew),
                            'lat': (['x'], latnew)}) 

        ds2.attrs=bltc.data.attrs

        ds2=ds2.dropna(dim='x',how='all')    
        AVISO=sl(ds2)

        domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
        abs_sl.data = xr.where(domain, abs_cop.data, np.nan).dropna(dim='x',how='all')

        A_cop,abs_cop=sl(AVISO.data['trend']).couple(abs_sl,limit=limit)
        av_index=(A_cop*1000. - abs_cop).data
        print(av_index.index.values)
        vals=av_index.values

        if basin=='north':
            av_index.to_netcdf('/home/oelsmann/Julius/Scripts/north_sea/plots/trends_compare/diff_tg_sat_second'+set1_name+set2_name+fltr+add_n)
        elif basin=='baltic':
            print('save netcdf')
            av_index.to_netcdf('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/flag_tests/presentation/diff_tg_sat_second'+set1_name+set2_name+fltr+add_n)            
        
        RMS=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
        med=np.round(np.nanmedian(vals),3)
        count=np.count_nonzero(~np.isnan(vals))
        print('Reference count: ',count)
        ax=plt.subplot(yp,xp,i)  
        ax.hist(vals,bins=20)

        ax.text(0.6,0.8,'RMS: '+str(RMS),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.6,0.7,'Median: '+str(med),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)

        ax.text(0.6,0.6,'count: '+str(count),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes) 
        
        ax.set_xlim(x_min,x_max)

        ax.set_xlabel('diff SAT-TG [mm/year]')
        if outlier_out:
            ax.set_title(name)  
            add='outl_out'
        else:
            add='all'
            ax.set_title(name)

        if save:
            
            if basin=='baltic':
                plt.savefig('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/flag_tests/presentation/'+set1_name+name+'_BALTIC_'+fltr+str(leng))
                
                

                
            else:
                plt.savefig('/home/oelsmann/Julius/Scripts/north_sea/plots/trends_compare/stats/trends_compare'+set1_name+name+fltr+str(leng)+add_n)



    else:
        
        plt.figure(figsize=(8,5))

        ax=plt.subplot(yp,xp,1)  
        ax.hist(vals,bins=20)
        ax.text(0.8,0.5,'RMS: '+str(RMS_b),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8,0.4,'Median: '+str(med_b),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8,0.3,'count: '+str(count_b),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes) 
        ax.text(0.8,0.2,'Fltr: '+fltr,horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)     

        ax.set_xlim(x_min,x_max)


        ax.set_xlabel('diff SAT-TG [mm/year]')
        ax.set_ylabel('counts')

        if outlier_out:
            ax.set_title('BALTIC '+fltr)        
        else:

            ax.set_title('BALTIC '+fltr)

        i=2

        name=set2_name
        mod='AR1'
        mapped='_mapped_on_baltic.nc'

        AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))
        bltc2=sl(AVISO.data['trend'])

        latnew=bltc2.flat(kind='lat')
        lonnew=bltc2.flat(kind='lon')

        data=bltc2.flat(kind='values')
        var='trend'

        print(latnew.shape)

        ds2 = xr.Dataset({var: (['x'],  data)},
                             coords={'lon': (['x'], lonnew),
                            'lat': (['x'], latnew)}) 

        ds2.attrs=bltc.data.attrs

        ds2=ds2.dropna(dim='x',how='all')    
        AVISO=sl(ds2)

        domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
        abs_sl.data = xr.where(domain, abs_cop.data, np.nan).dropna(dim='x',how='all')

        A_cop,abs_cop=sl(AVISO.data['trend']).couple(abs_sl,limit=50.)
        av_index=(A_cop*1000. - abs_cop).data
        vals=av_index.values
        
        RMS=np.round(np.sqrt(np.nanmean(((vals)**2))),4)
        med=np.round(np.nanmedian(vals),3)
        count=np.count_nonzero(~np.isnan(vals))
        ax=plt.subplot(yp,xp,i)  
        ax.hist(vals,bins=20)
        ax.text(0.8,0.5,'RMS: '+str(RMS),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8,0.4,'Median: '+str(med),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)
        ax.text(0.8,0.3,'count: '+str(count),horizontalalignment='center',
          verticalalignment='center', transform=ax.transAxes)  

        ax.set_xlim(x_min,x_max)

        ax.set_xlabel('diff SAT-TG [mm/year]')
        if outlier_out:
            ax.set_title(name+fltr)  
            add='outl_out'
        else:
            add='all'
            ax.set_title(name)

        if save:
            plt.savefig('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/flag_tests/'+set1_name+name+'_BALTIC_'+fltr+str(leng))


                
        
        
        
    return RMS_b,med_b,count_b,RMS,med,count


def sub_select_TG(psl,psl_trend,coverage=0.8,max_pos=284.,trend_err_perc=0.9):
    """
    sub_select TGs by criteria
    
    coverage       : fraction of TGs to consider
    trend_err_perc : maximum allowed uncertainty
    
    returns valid indices
    
    """
    #domain=(psl_trend.data['trend_un']> np.percentile(psl_trend.data['trend_un'].values, trend_err_perc))
    #psl.data = xr.where(domain, psl.data, np.nan).dropna(dim='x',how='all')
    
    #domain = (psl.data.count(dim='time')> coverage * max_pos)
    #psl.data = xr.where(domain, psl.data, np.nan).dropna(dim='x',how='all')
    
    
    #trend_err_perc=0.00125
    domain=(psl_trend.data['trend_un'] < trend_err_perc)
    ttt = xr.where(domain, psl.data, np.nan).dropna(dim='x',how='all')

    #coverage=0.8
    #max_pos=284.
    domain = (ttt.count(dim='time')> coverage * max_pos)
    ttt = xr.where(domain, ttt, np.nan).dropna(dim='x',how='all')

    ttt
    return ttt.index.values    


def recheck_indices(set2_name='',name='',basin='',abs_cop=None,abs_sl=None,index_standard=None):
    
    
    name=set2_name
    mod='AR1'
    mapped='_mapped_on_baltic.nc'

    if basin=='baltic':

        AVISO=sl(xr.open_dataset('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products/'+name+mod+mapped))

    else:

        if set2_name=='AVISO':

            AVISO=xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/aviso/'+'aviso_trend.nc')
            
            lon=AVISO.lon.values
            lon[lon>180]=lon[lon>180]-360
            AVISO.assign_coords(lon=("lon", lon))
            AVISO=sl(AVISO)
        elif set2_name=='SLCCI':
            AVISO=sl(xr.open_dataset('/home/oelsmann/Julius/Scripts/north_sea/data/slcci/'+'slcci_trend.nc'))                



    bltc2=sl(AVISO.data['trend'])

    latnew=bltc2.flat(kind='lat')
    lonnew=bltc2.flat(kind='lon')

    data=bltc2.flat(kind='values')
    var='trend'

    print(latnew.shape)

    ds2 = xr.Dataset({var: (['x'],  data)},
                         coords={'lon': (['x'], lonnew),
                        'lat': (['x'], latnew)}) 


    ds2=ds2.dropna(dim='x',how='all')    
    AVISO=sl(ds2)

    domain = ((np.isin(abs_sl.data.coords["index"].values,index_standard)))
    abs_sl.data = xr.where(domain, abs_cop.data, np.nan).dropna(dim='x',how='all')

    A_cop,abs_cop=sl(AVISO.data['trend']).couple(abs_sl,limit=100.)
    av_index=(A_cop*1000. - abs_cop).data.index.values

    
    return av_index

def check_flags_grid_trends(name='100_rem_res_new_flag_grids_test2',return_it=False,
                            test_params=[['ssh_std'],['sig0_fitting_error'],['helm_p']],threshold=[[1],[1],[1]],
                            outlier_out=False,save=False,remap=False,x_abs=8,mod2='detrend',mod1=2,how='no',
                            set2_name='AVISO',howt=[['absolute'],['absolute'],['absolute']],
                            sign=[['gt'],['gt'],['gt']],hector=False,mapped_on='AVISO',semi=False,mode='',monthly=False):
    
    """
    check performance of flagging for different settings
    
    """        
    jjj=0
    STATS=[]
    jjj=0
    if mode=='flag_file':
        return_it=True
    if return_it:
        for i in range(len(test_params)):   
            baltic_out=cal_trend_and_return(set1_name=name,x_abs=x_abs,outlier_out=outlier_out,
                                                          set2_name=set2_name,save=save,mod2=mod2,mod1=mod1,
                        how=how,var_all=test_params[i],threshold=threshold[i],howt=howt[i],sign=sign[i],hector=hector,mapped_on=mapped_on,semi=semi,mode=mode,monthly=monthly)        
        
            jjj=jjj+1

        return baltic_out  
    else:
        
        
        for i in range(len(test_params)):    
            RMS_b,med_b,count_b,RMS,med,count=calc_trends(set1_name=name,x_abs=x_abs,outlier_out=outlier_out,
                                                          set2_name=set2_name,save=save,mod2=mod2,mod1=mod1,
                        how=how,var_all=test_params[i],threshold=threshold[i],howt=howt[i],sign=sign[i],
                                                          mapped_on=mapped_on,monthly=monthly)
            STATS.append([str(test_params[i]),str(threshold[i]),RMS_b,med_b,count_b,RMS,med,count])
            #i=3
            # loop through different sigmas

            jjj=jjj+1

        return STATS            

def re_check_grid(name='',abss=False,single=False):
    """
    check grid props
    """
    test_params=[['count_cut']]
    threshold=[[150]]
    howt=[['absolute']]*5
    sign=[['lt']]*5

    STATS=check_flags_grid_trends(name=name,return_it=False,
                                test_params=test_params,threshold=threshold,
                                outlier_out=False,save=True,remap=False,x_abs=15,mod2='detrend',mod1=2,how='no',
                                set2_name='AVISO',howt=howt,
                                sign=sign)
    if abss:
        test_params=[['ssh_std','count_cut']]
        threshold=[[0.0044,250]]
        howt=[['absolute','absolute']]*5
        sign=[['lt','lt']]*5
        STATS=check_flags_grid_trends(name=name,return_it=False,
                                    test_params=test_params,threshold=threshold,
                                    outlier_out=False,save=True,remap=False,x_abs=15,mod2='detrend',mod1=2,how='no',
                                    set2_name='AVISO',howt=howt,
                                    sign=sign)
    if single:
        print(single)
    else:
        
        test_params=[['ssh_std','count_cut'],['ssh_std','count_cut'],['ssh_std','count_cut'],['ssh_std','count_cut'],['ssh_std','count_cut']]
        threshold=[[1,240],[1,250],[1,210],[1,220],[1,230]]
        howt=[['relative','absolute']]*5
        sign=[['lt','lt']]*5
        STATS=check_flags_grid_trends(name=name,return_it=False,
                                    test_params=test_params,threshold=threshold,
                                    outlier_out=False,save=True,remap=False,x_abs=15,mod2='detrend',mod1=2,how='no',
                                    set2_name='AVISO',howt=howt,
                                    sign=sign)    
    
    
    name1='100_rem_res_new_flag_grids_test2'
    name2=copy.deepcopy(name)

    baltic1=load_sat(name=name1)
    baltic2=load_sat(name=name2)

    psmsl=load_tg()
    psl=psmsl['sla']

    AVISO=load_sat(name='aviso_gridded')
    name='100_rem_res_new_flag_grids'
    baltic=load_sat(name=name)
    start=pd.DataFrame([baltic.time[0].values,AVISO.time[0].values]).max()
    ende=pd.DataFrame([baltic.time[-1].values,AVISO.time[-1].values]).min()
    baltic=baltic.loc[dict(time=slice(start[0], ende[0]))]
    #baltic1=baltic1.loc[dict(time=slice(start[0], ende[0]))]
    #baltic2=baltic2.loc[dict(time=slice(start[0], ende[0]))]

    AVISO=AVISO.loc[dict(time=slice(start[0], ende[0]))]['sla']
    #psl=psl.loc[dict(time=slice(start[0], ende[0]))]

    baltic_cor1=sl(baltic1['ssh']).cor(sl(psl)) # enlarge scatter size
    baltic_cor2=sl(baltic2['ssh']).cor(sl(psl)) 

    dat1=baltic_cor1.data
    dat1.attrs['long_name']='Baltic cor. TG, test2 (median: '+str(np.round(dat1.median().values,2))+') '
    baltic_cor1.data=dat1

    dat2=baltic_cor2.data
    dat2.attrs['long_name']='Baltic cor. TG, '+name2+' (median: '+str(np.round(dat2.median().values,2))+') '
    baltic_cor2.data=dat2




    plt_xr_map([[baltic_cor1.data],[baltic_cor2.data]],var=['no_var','no_var'],extend=extend,
                   ranges=[[0.5,1.],[0.5,1.]],msize=[200.,200.],cmap='rainbow',edgecolors=['k','k'],
                   save=True,save_dir='/nfs/DGFI36/altimetry/Baltic_SEAL_internal/plots_julius/prelim_grid_analysis/comparison_of_SL_products',
                   save_name='baltic_test2_'+name2+'_correlations_full_timeframe')


    
def make_flag_file(name='100_rem_res_new_flag_grids_test9'):    
    name=name
    test_params=[['ssh_std','count_cut']]
    threshold=[[1,250]]
    howt=[['relative','absolute']]*5
    sign=[['lt','lt']]*5

    #test_params=[['count_cut']]
    #threshold=[[150]]
    #howt=[['absolute']]*5
    #sign=[['lt']]*5
    test_params=[['ssh_std']]
    threshold=[[1]]
    howt=[['relative','absolute']]*5
    sign=[['lt','lt']]*5

    test_params=[['ssh_std']]
    threshold=[[90]]
    howt=[['relative_perc','absolute']]*5
    sign=[['lt','lt']]*5    
    
    
    #test_params=[['ssh_std','count_cut'],['ssh_std','count_cut'],['ssh_std','count_cut'],['ssh_std','count_cut'],#  ['ssh_std','count_cut'],['ssh_std','count_cut'],['ssh_std','count_cut'],['ssh_std','count_cut'],['ssh_std','count_cut'],#['ssh_std','count_cut']]#,['ssh_std_prod','count_cut'],['ssh_std_prod','count_cut']]#,['ssh_std_prod','count_cut'],['ssh_std_prod','count_cut']]
    #threshold=[[85,150],[85,250],[95,150],[95,250],[93,150],[93,250],[87,250],[87,150],[90,250],[1,250],[97.5,150],[97.5,250]]
    #howt=[['relative_perc','absolute']]*10
    #sign=[['lt','lt']]*10

    STATS=check_flags_grid_trends(name=name,return_it=False,
                                test_params=test_params,threshold=threshold,
                                outlier_out=False,save=True,remap=False,x_abs=15,mod2='detrend',mod1=2,how='no',
                                set2_name='AVISO',howt=howt,
                                sign=sign,mode='flag_file')

    bltc=sl(STATS*1)
    latnew=bltc.flat(kind='lat')
    lonnew=bltc.flat(kind='lon')
    data=bltc.flat(kind='values')
    time=STATS.time.values
    var='grid_flag'
    #print(latnew.shape)
    ds = xr.Dataset({var: (['time','x'],  data)},
                         coords={'time': (['time'], time),'lon': (['x'], lonnew),
                        'lat': (['x'], latnew)}) 


    lonnew2=[9.70732524]
    latnew2=[53.56735176]
    data2=(np.empty([len(time),1])*0).astype(int)
    ds2=xr.Dataset({var: (['time','x'],  data2)},
                         coords={'time': (['time'], time),'lon': (['x'], lonnew2),
                        'lat': (['x'], latnew2)}) 
    flag_file=xr.concat([ds,ds2],dim='x')
    flag_file.attrs=STATS.attrs
    flag_file.to_netcdf('/nfs/DGFI36/altimetry/Baltic_SEAL_internal/flag_files/flag_file_'+name+test_params[0][0]+'_'+str(threshold[0][0])+'.nc')
    return flag_file

    ###



    
    

def load_testfiles():
    test_file_dir='/home/oelsmann/Julius/Scripts/sealeveltools/sealeveltools/tests/files/'
    xrat=xr.open_dataset(test_file_dir+'xra.nc')
    xrt=xr.open_dataset(test_file_dir+'xr.nc')
    pdt=pd.read_csv(test_file_dir+'pd')
    flt=1.1
    return [xrat,xrt,pdt,flt]

def test_sl(func):
    """
    tests freshly installed functions
    
    """
    typen=load_testfiles()
    
    for typ in typen:
        #print(typ)
        method_out = getattr(sl(typ), func)
    print('Method '+ func+' works')

def reshape_irreg_data(dataset,data,latnew,lonnew,index=[],time_=True):
    
    self=sl(dataset)
    if self.var=='':
        var='new_var'
    else:
        var=self.var

    
    if index==[]:
        if time_:
            ds = xr.Dataset({var: (['time','x'],  data)},
                                 coords={'lon': (['x'], lonnew),
                                'lat': (['x'], latnew),
                                'time':self.data.time.values}) 
        else:
            ds = xr.Dataset({var: (['x'],  data)},
                                 coords={'lon': (['x'], lonnew),
                                'lat': (['x'], latnew)}) 
    else: # add matching index for zoi
        if time_:
            ds = xr.Dataset({var: (['time','x'],  data)},
                                 coords={'lon': (['x'], lonnew),
                                'lat': (['x'], latnew),'idx': (['x'], index),
                                'time':self.data.time.values}) 
        else:
            ds = xr.Dataset({var: (['x'],  data)},
                                 coords={'lon': (['x'], lonnew),
                                'lat': (['x'], latnew),'idx': (['x'], index)})         

    ds.attrs==self.data.attrs
    return ds[var]  



def compare_north_sea_corrs(name='north_sea_gridded_merged_grids02_05',
                            detrend=False,save=True,limit=150,max_lon=16.):

    add=''+str(limit)+'_lon_'+str(max_lon)

   
    

    for second_data in ['AVISO_north_sea','SLcci_merged_north_sea']:
        psmsl=load_tg()
        psl=psmsl
        psl=xr.where((psl.lon>-2),psl,np.nan).dropna(dim='x',how='all')
        psl=psl.transpose('time','x')
        psl=psl['sla']

        AVISO=load_sat(name=second_data)
        name=name
        
        if name == 'north_gridded_merged_north_sea_gridded_merged_grids02_05_ssh_std_90_count_cut_240detrendno2_flagged.nc':
            dirrr='/home/oelsmann/Julius/Scripts/north_sea/eval/flag_analysis/'
            baltic=xr.open_dataset(dirrr+name)['ssh']
        else:
            baltic=load_sat(name=name)['ssh']
        start=pd.DataFrame([baltic.time[0].values,AVISO.time[0].values]).max()
        ende=pd.DataFrame([baltic.time[-1].values,AVISO.time[-1].values]).min()
        baltic=baltic.loc[dict(time=slice(start[0], ende[0]))]
        AVISO=AVISO.loc[dict(time=slice(start[0], ende[0]))]['sla']
        psl=psl.loc[dict(time=slice(start[0], ende[0]))]
        psl=psl.where(psl.lon<max_lon).dropna(dim='x',how='all')
        baltic,psl=sl(baltic).couple(psl)
        if detrend:
            baltic=sl(baltic).detrend().data
            AVISO=sl(AVISO).detrend().data
            psl=sl(psl).detrend().data
            add=add+'detr'



        baltic_cor=sl(baltic).cor(sl(psl),limit=limit) # enlarge scatter size
        if second_data=='SLcci_merged_north_sea':
        #aviso_cor=sl(AVISO).cor(sl(psl))
            aviso_cor=sl(AVISO.shift({'time':1})).cor(sl(psl),limit=limit)
        else:
            aviso_cor=sl(AVISO).cor(sl(psl))

        dat=baltic_cor.data
        dat.attrs['long_name']='DGFI-ALT cor. TG (median: '+str(np.round(dat.median().values,2))+') '

        baltic_cor.data=dat
        dat2=aviso_cor.data
        if second_data=='SLcci_merged_north_sea':
            actname='SLCCI'
        else:
            actname='AVISO'        

        dat2.attrs['long_name']=actname+ ' cor. TG (median: '+str(np.round(dat2.median().values,2))+') ' 
        aviso_cor.data=dat2



        plt_xr_map([[baltic_cor.data],[aviso_cor.data]],var=['no_var','no_var'],extend=[-4,13,50,60],
                       ranges=[[0.5,1.],[0.5,1.]],msize=[200.,200.],cmap='rainbow',edgecolors=['k','k'],
                       save=save,save_dir='/home/oelsmann/Julius/Scripts/north_sea/plots/correlations',
                       save_name=name+'DGFI_ALT_vs_'+second_data+'_correlations'+add)



    
    

        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                           