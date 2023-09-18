#!/usr/bin/env python3
# 
# Apply wiener filter to estimate the varying seasonal signal
#
# INPUT
# -----
# Station name : example CASC
# sigma_a      : std of random variation of annual amplitude
# sigma_sa     : std of random variation of semi-annual amplitude
# phi          : coefficient AR(1) process
#
# OUTPUT
# ------
# file with estimated seasonal signal
# file with observations - seasonal signal
#
# Machiel Bos, 30/4/2018, Coimbra
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
import os
import math
import numpy as np
import subprocess

#===============================================================================
# Global constants
#===============================================================================

pi  = 4.0*math.atan(1.0)
EPS = 1.0e-8


#===============================================================================
# Functions
#===============================================================================

#-----------------------------------------
def model_PSD_S(omega,phi,sigma_s,omega0):
#-----------------------------------------
    """ Computes the power spectral density of the periodic signal.

    Parameters
    ----------
    omega :
        normalised angular velocity [rad]
    phi :
        coefficients of AR(1) process
    sigma_s :
        standard deviation of white noise that drives AR(1) [mm]
    omega0 :
        normalised angular velocity of periodic signal [rad]

    Returns
    -------
    float
        power spectral density [mm^2/rad]
    """ 
    return 2.0*(pow(sigma_s,2.0)/pi) * \
		(1.0/(1.0-2.0*phi*math.cos(omega+omega0)+pow(phi,2.0)) + \
                 1.0/(1.0-2.0*phi*math.cos(omega-omega0)+pow(phi,2.0)))



#---------------------------------------------
def model_PSD_W(omega,kappa,sigma_pl,sigma_w):
#---------------------------------------------
    """ Computes the power spectral density of the noise.

    Parameters
    ----------
    omega :
        normalised angular velocity [rad]
    kappa :
        spectral index
    sigma_pl :
        standard deviation of the power-law noise [mm]
    sigma_w :
        standard deviation of the white noise [mm]

    Returns
    -------
    float
        power spectral density [mm^2/rad]
    """ 
    if omega<1.0e-6:
        return 9.9e99
    else:
        return (1.0/pi) * (pow(sigma_pl,2.0)/ \
		pow(2.0*math.sin(0.5*omega),-kappa) + pow(sigma_w,2.0));



#-----------------------------------------------------------------
def wienerfilter(n,x,kappa,sigma_pl,sigma_w,sigma_a,sigma_sa,phi):
#-----------------------------------------------------------------
    """ Applies the Wiener Filter to the residuals in vector r

    Parameters
    ----------
    n :
        number of observations
    x :
        vector with residuals
    kappa :
        spectral index
    sigma_pl :
        standard deviation of the power-law noise [mm]
    sigma_w :
        standard deviation of the white noise [mm]
    sigma_a :
        standard deviation of white noise that drives annual AR(1) [mm]
    sigma_sa :
        standard deviation of white noise that drives semi-annual AR(1) [mm]
    phi :
        coefficients of AR(1) process

    Returns
    -------
    float
        vector with estimated varying seasonal signal (s_r) [mm]
    """
    #--- angular velocity of annual signal
    omega0 = 2*pi/365.25

    #--- Compute FFT of observed time series (column vector)
    xfft = np.fft.rfft(x.T,n).T

    for i in range(0,int(n/2)+1):

        #--- frequency in cylcles per day
        omega = 2*pi*i/n

        #--- Compute scaling of FFT 
        S  = model_PSD_S(omega,phi,sigma_a,omega0)     # annual signal
        S += model_PSD_S(omega,phi,sigma_sa,2*omega0)  # semi-annual signal
        W  = model_PSD_W(omega,kappa,sigma_pl,sigma_w) # noise 
        H  = S/(S+W)                                   # optimal filter

        #--- apply optimal filter
        xfft[i] *= H
 
    #--- Convert scaled FFT back to time domain (column vector)
    return np.squeeze(np.fft.irfft(xfft.T,n).T)



#===============================================================================
# Main program
#===============================================================================

#--- Constant
eps = 1.0e-6


#--- Read command line arguments
if len(sys.argv)!=5:
    print('Correct usage: apply_WF.py station_name sigma_a sigma_sa phi')
    sys.exit()
else:
    station_name = sys.argv[1]
    sigma_a      = float(sys.argv[2])
    sigma_sa     = float(sys.argv[3])
    phi          = float(sys.argv[4])


#--- Analyse mom file in directory ./obs_files
output = subprocess.check_output('analyse_timeseries.py {0:s} PLWN'. \
                                                 format(station_name),shell=True)

#--- Check output before further parsing
if output.decode().startswith('Cannot')==True:
    print(output)
    sys.exit()

#--- split columns
cols = output.decode().split()

#--- parse output
cos_annual        = float(cols[7])
sin_annual        = float(cols[8])
cos_annual_sigma  = float(cols[9])
sin_annual_sigma  = float(cols[10])
cos_hannual       = float(cols[11])
sin_hannual       = float(cols[12])
cos_hannual_sigma = float(cols[13])
sin_hannual_sigma = float(cols[14])
sigma_w           = float(cols[16])
sigma_pl          = float(cols[17])
d                 = float(cols[18])
                                                   
print('sigma_pl={0:f}'.format(sigma_pl))
print('sigma_w ={0:f}'.format(sigma_w))
print('d={0:f}'.format(d))

#--- Read file
header = ""
DeltaT = None
flag   = []   # True for observation, False for data gap
t      = []   # time in MJD
x      = []   # Observations
x_hat  = []   # Fitted model
mjd0   = None
with open('./mom_files/{0:s}.mom'.format(station_name),'r') as fp:
    line = fp.readline()
    #--- Read header & offsets
    while line[0]=="#":
        #--- add line to header
        header += line

        #--- Does it contain informatioon about the sampling period?
        if line[0:17]=='# sampling period':
            cols = line.split()
            DeltaT = float(cols[3])

        #--- Look at next line  
        line = fp.readline()
 
    #--- Remove last newline
    header = header.rstrip()
 
    #--- Read data
    while len(line)>0:
        cols = line.split()
        mjd  = float(cols[0])
        obs  = float(cols[1])
        mod  = float(cols[2])

        #--- Store the residual if it's the first or next in line
        if mjd0==None or abs(mjd-mjd0-DeltaT)<eps:
            t.append(float(mjd))
            x.append(float(obs))
            x_hat.append(float(mod))
            flag.append(True)
        else:
            #--- Fill missing data with zeros
            while mjd0+DeltaT+eps<mjd:
                mjd0 += DeltaT
                t.append(float(mjd0))
                x.append(float(0.0))
                x_hat.append(float(0.0))
                flag.append(False)
            t.append(float(mjd))
            x.append(float(obs))
            x_hat.append(float(mod))
            flag.append(True)
        
        #--- Shift mjd0 to current epoch (i.e. mjd)      
        mjd0 = mjd

        #--- Read next line
        line = fp.readline()

#--- Determine the number of observations (including data gaps)
n = len(t)

#--- Create residuals
shape = (n, 1)
r = np.zeros(shape)
for i in range(0,n):
    r[i] = x[i] - x_hat[i]

#--- Convert sigma_pl to my unit which is without (Delta T)^(-kappa/4)
kappa = -2.0*float(d)
if DeltaT==None:
    print('Assuming default sampling period of 1 day')
    DeltaT=1.0
sigma_pl *= pow(DeltaT/365.25,-kappa/4.0)


#--- Define Design matrix for seasonal signal (annual + semi-annual)
shape = (n, 4)
H = np.zeros(shape)
for i in range(0,n):
    H[i,0] = math.cos(2*pi*(t[i]-51544.0)/365.25);
    H[i,1] = math.sin(2*pi*(t[i]-51544.0)/365.25);
    H[i,2] = math.cos(4*pi*(t[i]-51544.0)/365.25);
    H[i,3] = math.sin(4*pi*(t[i]-51544.0)/365.25);

#--- Fill amplitudes of seasonal signal
shape = (4, 1)
theta = np.zeros(shape)
theta[0] = cos_annual
theta[1] = sin_annual
theta[2] = cos_hannual
theta[3] = sin_hannual

#--- Remember constant seasonal signal (annual + semi-annual)
s_c = np.squeeze(np.asarray(H.dot(theta)))

#for i in range(0,n):
#    print('{0:f},  {1:f},  {2:f}'.format(t[i],r[i,0],s_c[i]))

#--- Apply filter
s_r = wienerfilter(n,r,kappa,sigma_pl,sigma_w,sigma_a,sigma_sa,phi)



#----------------------
#--- Save results -----
#----------------------

#--- Does the sea_files directory exists?
if not os.path.exists('./sea_files'):
    os.mkdir('./sea_files/')

#--- Does the fil_files directory exists?
if not os.path.exists('./fil_files'):
    os.mkdir('./fil_files/')

fp_fil = open('./fil_files/{0:s}.mom'.format(station_name),'w')
fp_sea = open('./sea_files/{0:s}.mom'.format(station_name),'w')
fp_mom = open('./mom_files/{0:s}_WF.mom'.format(station_name),'w')

#--- Copy header
fp_fil.write('{0:s}\n'.format(header))
fp_sea.write('{0:s}\n'.format(header))
fp_mom.write('{0:s}\n'.format(header))

#--- Save filtered observations and estimated varying seasonal
for i in range(0,n):
    if flag[i]==True:
        fp_fil.write('{0:10.1f} {1:9.5f}\n'.format(t[i],x[i]-(s_c[i]+s_r[i])))
        fp_sea.write('{0:10.1f} {1:9.5f}\n'.format(t[i],s_c[i]+s_r[i]))
        fp_mom.write('{0:10.1f} {1:9.5f} {2:9.5f}\n'.format(t[i],x[i],x_hat[i]+s_r[i]))
fp_fil.close()
fp_sea.close()
fp_mom.close()
