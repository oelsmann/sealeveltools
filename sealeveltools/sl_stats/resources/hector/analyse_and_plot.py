#!/usr/bin/env python3
# 
# Read all mom-files in the ./obs_files directory, analyse them, plot them
# and create power spectral density plots.
#
#  This script is part of Hector 1.7.2
#
#  Hector is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  any later version.
#
#  Hector is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with Hector.  If not, see <http://www.gnu.org/licenses/>
#===============================================================================

import sys
import math
import os
import re
import glob
import subprocess

#===============================================================================
# Subroutines
#===============================================================================

# Compute year, month and day from MJD
#---------------
def caldat(mjd):
#---------------
    """ Compute calendar date for given julian date.
    * 
    * Example: YEAR = 1970, MONTH = 1, DAY = 1, JD = 2440588. 
    * Reference: Fliegel, H. F. and van Flandern, T. C., Communications of 
    * the ACM, Vol. 11, No. 10 (October, 1968).
    *
    * http://www.usno.navy.mil/USNO/astronomical-applications/
    """
    jul = mjd + 2400001
    l = jul+68569;
    n = 4*l//146097;
    l = l-(146097*n+3)//4;
    i = 4000*(l+1)//1461001;
    l = l-1461*i//4+31;
    j = 80*l//2447;
    k = l-2447*j//80;
    l = j//11;
    j = j+2-12*l;
    i = 100*(n-49)+i+l;

    year  = i;
    month = j;
    day   = k;

    return [year,month,day]



# Make plot of Power-Spectrum
#-----------------------
def make_PSD_plot(name):
#-----------------------
    """ Make a power spectral density plot from the residuals 
    
    Arguments:
        name = name of station and filename without .mom extension
    """
    #--- create new gnuplot script file
    fp = open("plot_spectra.gpl", "w")
    fp.write("set terminal postscript enhanced size 4,4 color portrait" + \
                                                      " solid \"Helvetica\"\n")
    
    fp.write("set output './psd_figures/{0:s}_psd.eps'\n".format(name))
    fp.write("set border 3;\n")
    fp.write("set xlabel 'Frequency (cpy)' font 'Helvetica, 18';\n")
    fp.write("set ylabel 'Power (mm^2/cpy)' offset -1,0 font 'Helvetica, 18'\n")
    fp.write("set xtics nomirror;\n")
    fp.write("set xtics autofreq;\n")
    fp.write("set ytics nomirror;\n")
    fp.write("set ytics autofreq;\n")
    fp.write("set logscale xy;\n")
    fp.write("set nokey;\n")
    fp.write("set format y '10^{%T}';\n")
    fp.write("set format x '10^{%T}';\n")
    fp.write("set pointsize 1;\n")
    fp.write("set xrange[*:200];\n")
    fp.write("set yrange[*:*];\n")
    fp.write("s=31557600.0;\n")
    fp.write("set style line 1 lt 1 lw 3 pt 7 linecolor rgb \"#a6cee3\"\n")
    fp.write("set style line 2 lt 1 lw 3 pt 7 linecolor rgb \"red\"\n")
    fp.write("plot 'estimatespectrum.out' using ($1*s):($2/s) w p ls 1,\\\n")
    fp.write("     'modelspectrum.out'    using ($1*s):($2/s) w l ls 2\n")
    fp.close()

    #--- Call gnuplot
    try:
        subprocess.call(['gnuplot','plot_spectra.gpl'])
    except OSError:
        print('Something seems to have gone wrong with the powerspectrum plot')



#-------------------------------
def make_data_plot(name,epochs):
#-------------------------------
    """ Make a time series plot

    Parameters:
        name : station name
        epoch: list with dates of outliers
    """  
    #--- create new gnuplot script file
    fp = open("plot_data.gpl", "w")
    fp.write("set terminal postscript enhanced size 8,4.8 color portrait" + \
                                                        " solid 'Helvetica'\n")
    fp.write("set output './data_figures/{0:s}_data.eps'\n".format(name))
    fp.write("set border 3;\n")
    fp.write("set xlabel 'Years' font 'Helvetica, 18';\n")
    fp.write("set ylabel 'mm' offset -1,0 font 'Helvetica, 18';\n")
    fp.write("set xtics nomirror;\n")
    fp.write("set xtics autofreq;\n")
    fp.write("set ytics nomirror;\n")
    fp.write("set ytics autofreq;\n")
    fp.write("set nokey;\n")
    fp.write("set pointsize 0.4;\n")
    fp.write("set bar 0.5;\n")
    fp.write("set xrange[*:*];\n")
    fp.write("set yrange[*:*];\n")
    fp.write("set style line 1 lt 1 lw 3 pt 7 linecolor rgb '#a6cee3'\n")
    fp.write("set style line 2 lt 1 lw 3 pt 7 linecolor rgb 'red'\n")
    fp.write("set style line 3 lt 1 lw 3 pt 2 linecolor rgb 'black'\n")
    fp.write("plot './mom_files/{0:s}.mom' u".format(name) + \
                                " (($1-51544)/365.25+2000):2 w p ls 1,\\\n")
    fp.write("     './mom_files/{0:s}.mom' u".format(name) + \
                                " (($1-51544)/365.25+2000):3 w l ls 2")
    if len(epochs)>0:
        fp.write(",\\\n     './model_outliers.out' u " + \
				" (($1-51544)/365.25+2000):2 w p ls 3\n")
    else:
        fp.write("\n")

    #---- A plot of the residuals is also nice to have
    fp.write("\nset output './data_figures/{0:s}_res.eps'\n".format(name))
    fp.write("plot './mom_files/{0:s}.mom' u ".format(name) + \
                                " (($1-51544)/365.25+2000):($2-$3) w l ls 2\n")
    fp.close()

    #--- Call gnuplot
    os.system('gnuplot plot_data.gpl')
   
 

#===============================================================================
# Main program
#===============================================================================

#--- Constant
comp_name = {'0':'E', '1':'N', '2':'U'}
eps = 1.0e-5

#--- Read command line arguments
if len(sys.argv)==2:
    noisemodel = sys.argv[1]
    stations   = []
elif len(sys.argv)==3:
    noisemodel = sys.argv[1]
    stations   = [sys.argv[2]]
else:
    print('Correct usage: analyse_and_plot.py ' + \
			'GGMWN|PLWN|FNWN|RWFNWN|WN|AR1 [station]')
    sys.exit()
    

#--- Read station names in directory ./obs_files
if len(stations)==0:
    # fnames = glob.glob("./obs_files/*.mom") # old
    
    fnames = glob.glob("./obs_files/*series.mom") 
    
    #--- Did we find files?
    if len(fnames)==0:
        print('Could not find any mom-file in ./obs_files')
        sys.exit()

    #--- Extract station names
    for fname in sorted(fnames):
        m = re.search('/(\w+)\.mom',fname)
        if m:
            station = m.group(1)
            stations.append(station)
        else:
            print('Could not parse station name from: {0:s}'.format(fname))
            sys.exit()

#--- Do we know the noisemodel?
if noisemodel not in ['GGMWN','PLWN','FNWN','RWFNWN','WN','AR1']:
    print('unknown noise model: {0:s}'.format(noisemodel))
    sys.exit()

#-- Open files to store results
fp_res = open('analysis_results.csv','w')
fp_sea = open('seasonal.csv','w')
fp_noi = open('noise.csv','w')

#--- Store outliers in dictionary
outliers = {}

#--- Analyse station
for station in stations:

    #--- Get sampling period
    try:
        fp_mom = open("./obs_files/{0:s}.mom".format(station))
    except IOError:
        print("Could not open file ./obs_files/{0:s}.mom".format(station))
        sys.exit()

    first_line = fp_mom.readline()
    m = re.search('# sampling period (\d+\.?\d*)',first_line)
    if m:
        fs = 1.0/float(m.group(1))
        T = 1.0/(365.25*fs)
    else:
        print("./obs_files/{0:s}.mom does not have # sampling period!". \
								format(station))
        sys.exit()

    #--- Get 4 letter marker and component
    m = re.search('(\w+)_(\d)',station)
    if m:
        marker    = m.group(1)
        component = m.group(2)
    else:
        marker    = station
        component = None

    param = '{0:s} {1:s}'.format(station,noisemodel)
    output = subprocess.check_output('analyse_timeseries.py {0:s} fraction'. \
                                        	      format(param),shell=True)
    cols = output.decode().split()

    #--- parse output
    cos_annual        = cols[7]
    sin_annual        = cols[8]
    cos_annual_sigma  = cols[9]
    sin_annual_sigma  = cols[10]
    cos_hannual       = cols[11]
    sin_hannual       = cols[12]
    cos_hannual_sigma = cols[13]
    sin_hannual_sigma = cols[14]

    sigma             = cols[15]
    if noisemodel=='GGMWN':
        fraction_WN  = cols[16]
        fraction_GGM = cols[17]
        d            = cols[18]
        phi          = cols[19]
    elif noisemodel=='PLWN':
        fraction_WN  = cols[16]
        fraction_PL  = cols[17]
        d            = cols[18]
        sigma_WN     = math.sqrt(float(fraction_WN))*float(sigma)
        sigma_PL     = math.sqrt(float(fraction_PL))*float(sigma)/ \
						   math.pow(T,0.5*float(d))
    elif noisemodel=='FNWN':
        fraction_WN  = cols[16]
        fraction_FN  = cols[17]
    elif noisemodel=='RWFNWN':
        fraction_WN  = cols[16]
        fraction_FN  = cols[17]
        fraction_RW  = cols[18]
    elif noisemodel=='AR1':
        fraction_AR1 = cols[16]
        phi          = cols[17]

    #--- Read fitted model into memory
    with open('./mom_files/{0:s}.mom'.format(station),'r') as fp_dummy:
        lines = fp_dummy.readlines()
    mjd_model = []
    x_model   = []
    for line in lines:
        if not line.startswith('#'):
            cols_model = line.split();
            mjd_model.append(float(cols_model[0]))
            x_model.append(cols_model[2])

    #--- Read found outliers into epochs array
    epochs = []
    fp_dummy2 = open('model_outliers.out','w')
    with open('outliers.out') as fp_dummy:
        for line in fp_dummy:
            epoch = float(line.strip())
            i = 0
            while i<len(mjd_model) and epoch>mjd_model[i]:
                i = i+1
            if i==len(mjd_model):
                i = i-1
            else:
                if i>0 and (epoch-mjd_model[i-1])<(mjd_model[i]-epoch):
                   i = i-1
            fp_dummy2.write('{0:f} {1:s}\n'.format(epoch,x_model[i]))
            mjd  = int(math.floor(epoch))
            [year,month,day] = caldat(mjd)
            date = '{0:02d}/{1:02d}/{2:4d}'.format(day,month,year)
            epochs.append(date)
    fp_dummy2.close()

    #--- Remember which outliers were found
    if marker not in outliers.keys():
        outliers[marker] = epochs
    else:
        for epoch in epochs:
            if epoch not in outliers[marker]:
                outliers[marker].append(epoch)

    #--- Does the data_figures directory exists?
    if not os.path.exists('./data_figures'):
        os.mkdir('./data_figures')

    #--- Does the psd_figures directory exists?
    if not os.path.exists('./psd_figures'):
        os.mkdir('./psd_figures')

    #--- Create control file to estimate power spectrum of residuals
    fp = open("estimatespectrum.ctl", "w")
    fp.write("DataFile            {0:s}.mom\n".format(station))
    fp.write("DataDirectory       ./mom_files\n")
    fp.write("interpolate         yes\n")
    fp.write("NoiseModels         {0:s}\n".format(noisemodel))
    fp.write("ScaleFactor         1.0\n")
    fp.write("PhysicalUnit        mm\n")
    fp.close()

    #--- Run estimatespectrum
    output = subprocess.check_output('estimatespectrum 4',shell=True) 
    estimatespectrum_cols = output.decode().split()
    freq0 = estimatespectrum_cols[-5]
    freq1 = estimatespectrum_cols[-3]

    fp = open("modelspectrum.ctl","w")
    fp.write("DataFile            {0:s}.mom\n".format(station))
    fp.write("DataDirectory       ./mom_files\n")
    fp.write("ScaleFactor         1.0\n")
    fp.write("PhysicalUnit        mm\n")
    if noisemodel=='GGMWN':
        fp.write("NoiseModels         GGMWN\n")
    elif noisemodel=='PLWN':
        fp.write("NoiseModels         Powerlaw White\n")
    elif noisemodel=='FNWN':
        fp.write("NoiseModels         Flicker White\n")
    elif noisemodel=='RWFNWN':
        fp.write("NoiseModels         RandomWalk Flicker White\n")
    elif noisemodel=='WN':
        fp.write("NoiseModels         White\n")
    elif noisemodel=='AR1':
        fp.write("NoiseModels         ARMA\n")
        fp.write("AR_p                1\n")
        fp.write("MA_q                0\n")
    fp.write("TimeNoiseStart      1000\n")
    fp.close()

    fp = open("modelspectrum.txt","w")
    fp.write("{0:s}\n{1:f}\n".format(sigma,24.0/fs))
    if noisemodel=='GGMWN':
        fp.write("{0:s}\n{1:s}\n{2:s}\n{3:s}\n". \
					format(fraction_GGM,fraction_WN,d,phi))
    elif noisemodel=='PLWN':
        fp.write("{0:s}\n{1:s}\n{2:s}\n".format(fraction_PL,fraction_WN,d))
    elif noisemodel=='FNWN':
        fp.write("{0:s}\n{1:s}\n".format(fraction_FN,fraction_WN))
    elif noisemodel=='RWFNWN':
        fp.write("{0:s}\n{1:s}\n{2:s}\n".format(fraction_RW,fraction_FN,\
								  fraction_WN))
    elif noisemodel=='WN':
        fp.write("1\n")
    elif noisemodel=='AR1':
        fp.write("{0:s}\n{1:s}\n".format(fraction_AR1,phi))
    fp.write("2\n{0:s} {1:s}\n".format(freq0,freq1))
    fp.close()

    #--- Make modelled psd line
    os.system('modelspectrum < modelspectrum.txt > /dev/null')

    #--- Make plot of power-spectrum and data
    make_PSD_plot(station)

    #--- Make time series plot
    make_data_plot(station,epochs)

    fp_res.write('{0:s}, '.format(station))
    for i in range(0,6):
        fp_res.write('{0:s}, '.format(cols[i]))
    fp_res.write('{0:s}\n'.format(cols[6]))

    fp_sea.write('{0:s}, '.format(station))
    for i in range(7,14):
        fp_sea.write('{0:s}, '.format(cols[i]))
    fp_sea.write('{0:s}\n'.format(cols[14]))

    #--- Save noise properties
    if component==None:
        fp_noi.write('{0:s}, '.format(marker))
    else:
        fp_noi.write('{0:s}, {1:s}, '.format(marker,comp_name[component]))
    if noisemodel=='PLWN':
        fp_noi.write('{0:f}, {1:f}, {2:f}\n'.format(sigma_WN,sigma_PL,\
								  2*float(d)))
    else:    
        for i in range(15,len(cols)-1):
            fp_noi.write('{0:s}, '.format(cols[i]))
        fp_noi.write('{0:s}\n'.format(cols[len(cols)-1]))

fp_res.close()
fp_sea.close()
fp_noi.close()

#--- Finally, save outliers to file
fp_out = open('outliers.csv','w')
for marker in outliers.keys():
    fp_out.write('{0:s}'.format(marker))
    for epoch in outliers[marker]:
        fp_out.write(', {0:s}'.format(epoch))
    fp_out.write('\n')
fp_out.close()
