import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.cm as mpl_cm
import os
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
import iris
import iris.quickplot as qplt

def plt_xr_map(ds,var='all_',time=[0],ranges=[[0,0]],units=[''],edgecolors=[None],
               domain='auto',save=False,save_dir='',save_name='',
               msize=[0.3],extend=[9999],cmap=mpl_cm.get_cmap('brewer_RdBu_11'),
               land=True,portrait=True,save_args={'format':'png'},
               cbar_args={'joint':False},wspace=0.0, hspace=0.0,
               aspect_ratio=1.6,rivers=False,f_y=1.,f_x=1.,fit_dims=True
               ,labels=True,stitle='',zorder_on=False,cbar_pos='vertical',cb_size=0.6,
               return_fig=False,**kwargs):

    """
    plot map of xr at timestep
    
    

    Parameters
    
    ---------
    ds:               Dataset or list of Datasets or list of lists of Datasets
                      if Dataset: 
                          number of plots = number of selected variables or time-steps
                      if list of Datasets (or lists of Datasets): 
                          overlay plots of Datasets in list
                          plot number is len(list) 
                      to make vectorplot input data as list containing 
                      dataset with u and v variables and set var='vector'    
                          
    var:              list of variables in data
                      if 'all_' plot all variables
    time:             list of time-steps to plot
    ranges:           list of [vmin,vmax] values of plotted data-range
                      in case var: vector ranges[0][0] sets vector lenght (eg ranges = [[vector_lenght,0]])
                      see plt.quiver vector_lenght = scale factor
    units:            list of units
    extend:           lon, lat domain [lonmin,lonmax,latmin,latmax]
    portrait:         in case we have 2x3 imgages show like 2(x) and 3(y)
    save:             bool (save figure)
    save_dir:         save-directory
    land:             plot land or not
    portrait:         if True thend in case of 2x3 plots plot as 2 (x)  x  3(y), or vice versa
    hspace,
    wspace:           control whitespace between subplots
    aspect_ratio:     set aspect ratio
    cbar_args:        joint-. set one colorbar for all plots
    rivers:           plot rivers
    stitle:           set main plot title
    f_y:              y-ratio of figure
    f_x:              x-ratio of figure
    labels:           add a,b,c) ... labels
    zorder_on:        bool, define zorder of land and coastline plots
        
    **kwargs:
    
    """
    global labels_on,cb_pos,cb_siz,zorder_oni
    cb_siz=cb_size
    labels_on=labels
    cb_pos=cbar_pos
    zorder_oni=zorder_on
    proj=ccrs.PlateCarree()
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
                                    edgecolor='face',
                                    facecolor='lightgrey')#'#f7f7f7')#cfeature.COLORS['land'])
    plt.ioff()
    # to be automised (loop over axis)
    
    if var=='all_':
        
        var=list(ds.keys())
        if 'time_bnds' in var:
            var.remove('time_bnds')
    plot_number,var,date_in,ranges,units,msizes,time_setter,edgecolors,overlay,multi_overlay,ax_index,sub_plot_index=arrays_options(ds,var,time,ranges,units,msize,edgecolors)

    function='scatter'
    xp,yp,nump=fit_2D_dimension(plot_number,overlay=overlay,
                                multi_overlay=multi_overlay,ax_index=ax_index,fit_dims=fit_dims)
    if not portrait:
       xp,yp = yp,xp 

    print('plot ',var)
    print('shape: ',xp,'x',yp)
    
    lbls=['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)','k)','l)','m)','n)','o)','p)']

    if multi_overlay:
        ind=[x - 1 for x in ax_index]
        lbls=np.asarray(lbls)[ind]
    size=7
        

    if xp == 3 and yp ==3:
        f_y=0.7
        f_x=1.4
        print('scale')
    if not fit_dims:
        fig =plt.figure(figsize=(yp*f_x*size, xp*size*f_y))
    else:
        fig =plt.figure(figsize=(xp*f_x*size, xp*size*f_y))        

    title=''
    
    if cbar_args['joint']:
        """
        one colorbar for all plots
        """
        axes_class = (GeoAxes,
                      dict(map_projection=proj))
        axgr = AxesGrid(fig, 111, axes_class=axes_class,
                        nrows_ncols=(xp,yp),
                        axes_pad=1.,
                        cbar_location='bottom',
                        cbar_mode='single',
                        cbar_pad=0.1,
                        cbar_size='1.%',
                        label_mode='')  # note the empty label_mode        

        for i, ax in enumerate(axgr):
            i=i+1
            date_ini=pd.Timestamp(date_in[i-1])
            vari=var[i-1]
            cbarb=True
            ii=i
            lat,lon,data,ds_f,function,long_name=make_map_data(ds,vari,date_ini,function)            
            if cbar_args['joint']:
                cbarb=False

            if 'long_name' in ds_f.attrs:
                title=ds_f.long_name        
            else:
                title=vari   
            if 'units' in ds_f.attrs:
                units[i-1]=ds_f.units   

            year=str(date_ini.month)+'-'+str(date_ini.year)
            #ax=plt.subplot(xp,yp,ii,projection=proj)    
            ax,im=plt_xr_map_ax(ax,fig,data,lat,lon,land_50m,title,year,time_setter,ranges[i-1],
                             units[i-1],extend,edgecolor=edgecolors[i-1],aspect_ratio=aspect_ratio,function=function,color_map=cmap,msize=msizes[i-1],cbarb=cbarb,land=land,cbar_args=cbar_args,im_out=True,rivers=rivers,**kwargs)
            ax.text(0.5, 1.12, title,horizontalalignment='center',fontsize=15,transform = ax.transAxes)

        axgr.cbar_axes[0].colorbar(im)
        #fig.colorbar(im,cax=axgr.cbar_axes[0],orientation='horizontal',shrink=0.6,fraction=0.5)

        
    #if cbar_args['joint']:
        #cbar_ax = fig.add_axes([0.27, 0.0, 0.5, 0.05])
        
        #fig.colorbar(ax, cax=cbar_ax)    
    #    cax = divider.append_axes('bottom', size='5%', pad=0.05)
    #    ax,im=plt_xr_map_ax(ax,fig,data,lat,lon,land_50m,title,year,time_setter,ranges[i-1],
    #                     units[i-1],extend,edgecolor=edgecolors[i-1],function=function,color_map=cmap,msize=msizes[i-1],cbarb=cbarb,land=land,cbar_args=cbar_args,im_out=True)

    #    fig.colorbar(im,cax=cbar_ax,orientation='horizontal')

    else:
        main_plot_index=0
       
        for i in range(1,plot_number+1):
            date_ini=pd.Timestamp(date_in[i-1])
            vari=var[i-1]
            cbarb=True
            
            if str(type(ds))=="<class 'list'>": # overlay plots
                overlay=True
                if multi_overlay:
                    print('multi')
                    ii=ax_index[i-1]  # e.g. ax_index = [1,1,1,2,2,3,4]
                                      # e.g. sub_plot_index = [0,1,2,0,1,0,0]
                    lat,lon,data,ds_f,function,long_name=make_map_data(ds[ii-1][sub_plot_index[i-1]],vari,date_ini,function)

                    if ax_index[i] == ax_index[i-1] and var[i] is not 'vector':
                        cbarb = False
                        title=''
                        main_plot_index=main_plot_index-1
                    else:
       
                        if 'long_name' in ds_f.attrs:
                            print(ds_f.long_name)
                            title=ds_f.long_name        
                        else:
                            title=vari   
                        if 'units' in ds_f.attrs:
                            units[i-1]=ds_f.units   
                        if vari=='vector':
                            title=long_name
                else:            
                    ii=1
                    lat,lon,data,ds_f,function,long_name=make_map_data(ds[i-1],vari,date_ini,function)
                        

                    if i<plot_number and 'vector' not in var:
                        cbarb=False
                        main_plot_index=main_plot_index-1
                    else:
                        if 'long_name' in ds_f.attrs:
                            title=title+' '+ds_f.long_name        
                        else:
                            title=vari   
                        if vari=='vector':
                            title=long_name
                        if 'units' in ds_f.attrs:
                            units[i-1]=ds_f.units   

            else:
                ii=i

                lat,lon,data,ds_f,function,long_name=make_map_data(ds,vari,date_ini,function)            
                if cbar_args['joint']:
                    cbarb=False

                if 'long_name' in ds_f.attrs:
                    title=ds_f.long_name        
                else:
                    title=vari   
                if 'units' in ds_f.attrs:
                    units[i-1]=ds_f.units   
            if main_plot_index <0:
                main_plot_index=0
            year=str(date_ini.month)+'-'+str(date_ini.year)
            ax=plt.subplot(xp,yp,ii,projection=proj)    
            ax=plt_xr_map_ax(ax,fig,data,lat,lon,land_50m,title,year,time_setter,ranges[i-1],
                             units[i-1],extend,lbls[main_plot_index],aspect_ratio=aspect_ratio,
                             edgecolor=edgecolors[i-1],function=function,
                             color_map=cmap,msize=msizes[i-1],cbarb=cbarb,land=land,
                             cbar_args=cbar_args,rivers=rivers,**kwargs)
            main_plot_index=main_plot_index+1
            
        #plt.tight_layout()

    if xp > 2 or yp > 2: 
        plt.subplots_adjust(wspace=wspace, hspace=hspace) 
    else:
        plt.subplots_adjust(wspace=0.07, hspace=0.03)
    
    #if holdon:
    #    hold(True)
    #plt.tight_layout()
    
    if stitle is not '':
        fig.suptitle(stitle, fontsize=16)
        
    
    
    if save_dir=='':
        save_dir = os.path.dirname(os.path.realpath('__file__'))
        save_name=var[0]
    
    if save:
        save_name=save_name+'.'+save_args['format']
        plt.savefig(save_dir+'/'+save_name,**save_args,bbox_inches='tight')
    if return_fig:
        return fig
    else:
        plt.show()
    
    

def plt_xr_map_ax(ax,fig,data,lat,lon,land_50m,title,year,time_setter,range_in,unit,extend,lbl,
                  function='scatter',msize=0.3,edgecolor=None,color_map='',cbarb=True,
                  land=True,aspect_ratio=1.6,cbar_args={'joint':False},im_out=False,rivers=False,**kwargs):
    """
    ax-object for xr_map plot
    """
    if function=='scatter' or zorder_oni:
        zol=-1
        zoc=0
    elif ~zorder_oni:
        zol=4
        zoc=5       

    if im_out:
        cbarb=True
    if land:
        ax.add_feature(land_50m,zorder=zol)
        ax.coastlines('10m',zorder=zoc)
        
    #if 'rivers' in kwargs:
    
    if rivers:
        rivers_50m = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', '10m')
        ax.add_feature(rivers_50m, facecolor='None', edgecolor='grey')    
    if 'lakes' in kwargs:
        ax.add_feature(cfeature.LAKES)
        
    ax.set_aspect(aspect=aspect_ratio)
    minv,maxv = color_range(data,range_in,option='std')

    if function=='scatter':
        
        if cbarb:
            im = ax.scatter(lon,lat,c=data,s=msize,edgecolor=edgecolor,
                            vmin=minv,vmax=maxv,cmap=color_map,transform=ccrs.Geodetic(),zorder=0)
        else:
            ax.scatter(lon,lat,c=data,s=msize,edgecolor=edgecolor,
                       vmin=minv,vmax=maxv,cmap=color_map,transform=ccrs.Geodetic(),zorder=0)
            
           
    elif function =='contour':
        

            if cbarb:

                im=ax.pcolormesh(lon,lat,data,vmin=minv,vmax=maxv,cmap=color_map,transform=ccrs.PlateCarree(),zorder=0)    #cmap=color_map,
            else:
                ax.pcolormesh(lon,lat,data,vmin=minv,vmax=maxv,cmap=color_map,transform=ccrs.PlateCarree(),zorder=0)
    elif function =='vector':
        
            u=data[0]
            v=data[1]
            #crs = ccrs.RotatedPole(pole_longitude=177.5, pole_latitude=37.5)
            #lon, lat, u, v, vector_crs = sample_data()
            lon, lat, u, v=re_sample_data(lon, lat, u, v,factor=10)
            print(range_in[0])
            ax.quiver(lon, lat, u, v, scale_units='inches',scale=range_in[0],transform=ccrs.PlateCarree())
            
            cbarb=False

    
            
            
    if extend[0] != 9999:
        ax.set_extent(extend, crs=ccrs.PlateCarree())
        xticks,yticks=plt_make_ticks(extend[2:])   
    else:
        xticks,yticks=plt_make_ticks(lat) 
        
    if function is not 'vector':
        gl=ax.gridlines(draw_labels=True,xlocs=xticks, ylocs=yticks, alpha=0.25)
        gl=gl_props(gl,0.15)
    #ax.set_title(title)
    
    if land == False:
        ax.coastlines('10m')
        
    if cbarb and im_out==False:
        #clb = fig.colorbar(im, ax=ax, orientation=cb_pos,shrink=0.6,aspect=15.,extend='both')
        clb = fig.colorbar(im, ax=ax, orientation=cb_pos,shrink=cb_siz,aspect=15.,extend='both')
        clb.mappable.set_clim(minv,maxv)
        clb.ax.set_title(unit,size=11)    
    plt.text(0.5, 1.12, title,
     horizontalalignment='center',
     fontsize=15,
     transform = ax.transAxes)
    if time_setter:
        plt.text(0.8, 0.15, year,
             horizontalalignment='center',
             fontsize=15,
             transform = ax.transAxes)    
    #cbar=fig.colorbar(im,cax=ax, orientation='vertical',shrink=0.5)   
    #cbar.ax.set_xlabel('RMS [mm/year]')    
    if labels_on and cbarb:
        
        ax.text(-0.15, 1.12, lbl,
             horizontalalignment='left',
             fontsize=15,
             transform = ax.transAxes)
    if im_out:
        return ax,im
    else:
        return ax       



def re_sample_data(lon, lat, u, v,factor=6):

    return lon[::factor,::factor], lat[::factor,::factor], u[::factor,::factor], v[::factor,::factor]
    
    
def plt_make_ticks(lat):
    
    latrange=np.abs(np.max(lat)-np.min(lat))
    
    res=int((latrange*0.4))
    
    if res < 0.1:
        res=1.
    
    xticks=np.arange(-180,180,res)
    yticks=np.arange(-90,90,res)
    
    return xticks,yticks
    
def get_divider(num):
    dividers=[]
    if num > 1:
       # check for factors
        for i in range(2,num):
            if (num % i) == 0:
                dividers.append(i)
    else:
        dividers.append(1)

    return np.array(dividers)

def color_range(data,range_in,option='std'):
    """
    calc colorrange
    """
    if range_in[1]==0:
        if option == 'std':
            std=np.nanquantile(data, .25)        
            med=np.nanmedian(data)
            std=med-std
            minv=med-3*std
            maxv=med+3*std

        elif option =='minmax':
            minv=np.nanmin(data)
            maxv=np.nanmax(data)
    else:
        minv=range_in[0]
        maxv=range_in[1]   
    return minv,maxv
    
    

def fit_2D_dimension(num,overlay=False,multi_overlay=False,ax_index=[],fit_dims=True):
    """
    aranges subplots 
    
    """
    if fit_dims:
        if overlay:
            return 1,1,2

        elif multi_overlay:
                num=np.max(ax_index)-1

                return fit_2D_dimension(num,overlay=False,multi_overlay=False,ax_index=[])

        else:

            dividers=get_divider(num)
            if len(dividers)>0:
                x=dividers[np.argmin(np.abs((dividers-np.sqrt(num))))]
                y=int(num/x)
            else:
                num=num+1
                x,y,num = fit_2D_dimension(num)
            return y,x,num
    else:
        x=num
        y=1        
        return y,x,num

def gl_props(gl,scaler):
    gl.xlabel_style = {'size': 80*scaler}
    gl.ylabel_style = {'size': 80*scaler}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabels_left = True
    gl.ylabels_right = False    
    return gl

def make_map_data(ds,vari,date_ini,function):
    """
    get data from dataset
    
    vector: True plot vector plot
    """
    
    function='scatter'
    long_name=''
    
    if vari=='vector':
        function='vector'
        ds_f_all=[]

        for varr in [[s for s in list(ds.keys()) if "u" in s][0],[s for s in list(ds.keys()) if "v" in s][0]]:
            # search for variables with u and v and get first
            ds_f=ds[varr] 

            if 'time' in ds.dims:

                lat=ds_f.loc[date_ini].lat.values.flatten()
                lon=ds_f.loc[date_ini].lon.values.flatten()
                data=ds_f.loc[date_ini].values       

            else:

                lat=ds_f.lat.values.flatten()
                lon=ds_f.lon.values.flatten()
                data=ds_f.values          
            ds_f_all.append(data)
        
        if len(lat) != len(lon):

            function='vector'
            lon, lat = np.meshgrid(lon, lat)
        data=ds_f_all
        if 'long_name' in ds_f.attrs:

            long_name=ds_f.attrs['long_name']
    else:
        

        if vari=='no_var':
            ds_f=ds
        else:
            ds_f=ds[vari] 

        if 'time' in ds.dims:

            lat=ds_f.loc[date_ini].lat.values.flatten()
            lon=ds_f.loc[date_ini].lon.values.flatten()
            data=ds_f.loc[date_ini].values       

        else:

            lat=ds_f.lat.values.flatten()
            lon=ds_f.lon.values.flatten()
            data=ds_f.values  

        if len(lat) != len(lon):

            function='contour'
            lon, lat = np.meshgrid(lon, lat)
        else:
            data=data.flatten()

    return lat,lon,data,ds_f,function,long_name


def adjust_array_len(arrayin,plot_number):
    
    len_ar=len(arrayin)
    if len_ar < plot_number and len_ar > 1:
        arrayin = arrayin*int((plot_number/len_ar+1))
    elif len_ar < plot_number and len_ar == 1:
        arrayin = arrayin*plot_number
    return arrayin
    
    

def arrays_options(ds,var,time_in,ranges,units,msize,edgecolors):
    """
    prepares axes properties for all subplots
    
    """

    overlay=False
    multi_overlay=False
    time_setter=False
    ax_index=None
    sub_plot_index=None
    if str(type(var)) =="<class 'str'>":
        var=[var]
    if str(type(msize)) =="<class 'float'>" or str(type(msize)) =="<class 'int'>":
        msize=[msize]        

    if str(type(ds))=="<class 'list'>": # overlay plots
        if str(type(ds[0]))=="<class 'list'>": # multi_overlay plots
            multi_overlay=True
            count=0
            ii=[]
            jj=[]
            iii=1
            for i in ds:
                jjj=0
                for j in i:
                    jj.append(jjj)    
                    ii.append(iii)
                    count += 1 
                    jjj +=1
                iii += 1
            ii.append(iii)
            plot_number=count
            ax_index=ii
            sub_plot_index=jj
        else:
            plot_number=len(ds)
            overlay=True
            if time_in[0]==0 and 'time' in ds[0].dims:
                time_setter=False
                time_in=[ds[0].time[0].values]
    else:
        if time_in[0]==0 and 'time' in ds.dims:
            time_setter=False
            time_in=[ds.time[0].values]

        plot_number=len(var)*len(time_in)        

    ranges = adjust_array_len(ranges,plot_number)
    units = adjust_array_len(units,plot_number)  
    msizes = adjust_array_len(msize,plot_number) 
    edgecolors = adjust_array_len(edgecolors ,plot_number) 
    if len(var) == plot_number:  
        time_in = adjust_array_len(time_in,plot_number)          
    elif len(time_in) == plot_number:
        var = adjust_array_len(var,plot_number)           
    else:
        var = adjust_array_len(var,plot_number)   
        time_in = adjust_array_len(time_in,plot_number)    
        


    return plot_number,var,time_in,ranges,units,msizes,time_setter,edgecolors,overlay,multi_overlay,ax_index,sub_plot_index

    
    
    
