#!/usr/bin/env python3
# 
# For given mom-filename and noise model combination, analyse the time series.
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
import re
import numpy as np
#===============================================================================
# Subroutines
#===============================================================================


#----------------------------------------
def extract_results(noisemodel,fraction):
#----------------------------------------
    """ Extract estimated noise parameters from estimatetrend.out.

    Args:
        noisemodel = GGWN|PLWN|FNWN|RWFNWN|WN|AR1
        fraction   = boolean (do we need noise fractions or amplitudes?)
    Returns:
        return list of estimated parameters
    """

    #--- Define variables we need to send back
    Sa_cos = Sa_cos_sigma = Sa_sin = Sa_sin_sigma = N = LogL = None
    Ssa_cos = Ssa_cos_sigma = Ssa_sin = Ssa_sin_sigma = None
    noise = ['None']*5
    GGM_section = PL_section = FN_section = WN_section = RW_section = False
    AR1_section = False
    trend = trend_error = mjd = aic = bic = bic_c = None

    #--- Read each line from file and see if it contains a label we need
    with open("estimatetrend.out",'r') as fp:
        for line in fp:
            #--- Number of observations
            m = re.match("^Number of observations:\s*(\d+)",line)
            if m:
                N = m.group(1)
            #--- logL
            m = re.match("^min log\(L\)=\s*(-?\d+\.?\d*)",line)
            if m:
                LogL = m.group(1)
            #--- AIC
            m = re.match("^AIC       =\s*(-?\d+\.?\d*)",line)
            if m:
                aic = m.group(1)
            #--- BIC
            m = re.match("^BIC       =\s*(-?\d+\.?\d*)",line)
            if m:
                bic = m.group(1)
            #--- BIC_c
            m = re.match("^BIC_c     =\s*(-?\d+\.?\d*)",line)
            if m:
                bic_c = m.group(1)
            #--- Driving noise
            m = re.match("^STD of the driving noise:\s*(\d+\.\d+)",line)
            if m:
                std = m.group(1)

            #--- trend
            m = re.match("^trend:\s+(-?\d+\.?\d*) \+\/\- (\d+\.?\d*)",line)
            if m:
                trend       = m.group(1)
                trend_error = m.group(2)
            #--- Sa_cos
            m = re.match( \
		 "^cos yearly :\s*(-?\d+\.?\d*)\s+\+\/-\s+(\d+\.?\d*)",line)
            if m:
                Sa_cos       = m.group(1)
                Sa_cos_sigma = m.group(2)
            #--- Sa_sin
            m = re.match( \
		 "^sin yearly :\s*(-?\d+\.?\d*)\s+\+\/-\s+(\d+\.?\d*)",line)
            if m:
                Sa_sin       = m.group(1)
                Sa_sin_sigma = m.group(2)
            #--- Ssa_cos
            m = re.match( \
		 "^cos hyearly :\s*(-?\d+\.?\d*)\s+\+\/-\s+(\d+\.?\d*)",line)
            if m:
                Ssa_cos       = m.group(1)
                Ssa_cos_sigma = m.group(2)
            #--- Ssa_sin
            m = re.match( \
		 "^sin hyearly :\s*(-?\d+\.?\d*)\s+\+\/-\s+(\d+\.?\d*)",line)
            if m:
                Ssa_sin       = m.group(1)
                Ssa_sin_sigma = m.group(2)

            #=== non-White noise parameters ====
            #--- GMM + WN
            if noisemodel=='GGMWN':
                m = re.match('^GGM:',line)
                if m:
                    GGM_section = True
                if GGM_section==True:
                    if fraction==True:
                        m = re.match('fraction  =\s*(\d+\.\d+)',line)
                        if m:
                            noise[1] = m.group(1)
                    else:
                        m = re.match('sigma     =\s*(\d+\.?\d*)',line)
                        if m:
                            noise[1] = m.group(1)
                    m = re.match('d         =\s*(-?\d+\.?\d*)',line)
                    if m:
                        noise[2] = m.group(1)
                    m = re.match('1-phi     =\s*(\d+\.?\d*e?[-+]?\d*)',line)
                    if m:
                        noise[3] = m.group(1)
                        GGM_section=False

            #--- PL + WN 
            elif noisemodel=='PLWN':
                m = re.match('^GGM:',line)
                if m:
                    PL_section = True
                if PL_section==True:
                    if fraction==True:
                        m = re.match('fraction  =\s*(\d+\.\d+)',line)
                        if m:
                            noise[1] = m.group(1)
                    else:
                        m = re.match('sigma     =\s*(\d+\.?\d*)',line)
                        if m:
                            noise[1] = m.group(1)
                    m = re.match('d         =\s*(-?\d+\.?\d*)',line)
                    if m:
                        noise[2] = m.group(1)
                        PL_section=False

            #--- FN + WN 
            elif noisemodel=='FNWN':
                m = re.match('^FlickerGGM:',line)
                if m:
                    FN_section = True
                if FN_section==True:
                    if fraction==True:
                        m = re.match('fraction  =\s*(\d+\.\d+)',line)
                        if m:
                            noise[1] = m.group(1)
                            FN_section = False
                    else:
                        m = re.match('sigma     =\s*(\d+\.?\d*)',line)
                        if m:
                            noise[1] = m.group(1)
                            FN_section = False

            #--- RW + FN + WN 
            elif noisemodel=='RWFNWN':
                m = re.match('^FlickerGGM:',line)
                if m:
                    FN_section = True
                if FN_section==True:
                    if fraction==True:
                        m = re.match('fraction  =\s*(\d+\.\d+)',line)
                        if m:
                            noise[1] = m.group(1)
                            FN_section = False
                    else:
                        m = re.match('sigma     =\s*(\d+\.?\d*)',line)
                        if m:
                            noise[1] = m.group(1)
                            FN_section = False
                m = re.match('^RandomWalkGGM:',line)
                if m:
                    RW_section = True
                if RW_section==True:
                    if fraction==True:
                        m = re.match('fraction  =\s*(\d+\.\d+)',line)
                        if m:
                            noise[2] = m.group(1)
                            RW_section = False
                    else:
                        m = re.match('sigma     =\s*(\d+\.?\d*)',line)
                        if m:
                            noise[2] = m.group(1)
                            RW_section = False

            #=== White noise parameters ====
            m = re.match('^White:',line)
            if m:
                WN_section = True
            if WN_section==True:
                if fraction==True:
                    m = re.match('fraction  =\s*(\d+\.\d+)',line)
                    if m:
                        noise[0] = m.group(1)
                        WN_section = False
                else:
                    m = re.match('sigma     =\s*(\d+\.?\d*)',line)
                    if m:
                        noise[0] = m.group(1)
                        WN_section = False

            #=== White noise parameters ====
            m = re.match('^ARMA:',line)
            if m:
                AR1_section = True
            if AR1_section==True:
                if fraction==True:
                    m = re.match('fraction  =\s*(\d+\.\d+)',line)
                    if m:
                        noise[0] = m.group(1)
                else:
                    m = re.match('sigma     =\s*(\d+\.?\d*)',line)
                    if m:
                        noise[0] = m.group(1)
                m = re.match('AR\[1\]\s+=\s*(-?\d+\.?\d*)',line)
                if m:
                    noise[1] = m.group(1)
                    AR1_section=False

    #--- Construct output
    output = [trend,trend_error,N,LogL,aic,bic,bic_c,Sa_cos,Sa_sin, \
               Sa_cos_sigma, Sa_sin_sigma, Ssa_cos, Ssa_sin, Ssa_cos_sigma, \
				    			   Ssa_sin_sigma,std]
    i=0 
    while noise[i]!='None' and i<5:
        output.append(noise[i])
        i += 1

    return output



#-------------------------------------------
def create_removeoutliers_ctl_file(station):
#-------------------------------------------
    """ Create ctl file for removeoutlier.

    Args:
        station : station name (including _0, _1 or _2) of the mom-file
    """

    #--- Create control.txt file for removeoutliers
    fp = open("removeoutliers.ctl", "w")
    fp.write("DataFile            {0:s}.mom\n".format(station))
    fp.write("DataDirectory         ./obs_files\n")
    fp.write("interpolate           no\n")
    fp.write("OutputFile            ./pre_files/{0:s}.mom\n".format(station))
    fp.write("seasonalsignal        yes\n")
    fp.write("halfseasonalsignal    yes\n")
    fp.write(estimateoffsets)
    fp.write("estimatepostseismic   no\n")
    fp.write("estimateslowslipevent no\n")
    fp.write("ScaleFactor           1.0\n")
    fp.write("PhysicalUnit          mm\n")
    fp.write("IQ_factor             3\n")
    fp.close()



#------------------------------------------------------
def create_estimatetrend_ctl_file (station,noisemodel):
#------------------------------------------------------
    """ Create ctl file for findoffset.

    Args:
        station : station name (including _0, _1 or _2) of the mom-file
        noisemodel (string): GGMWN, PLWN, FNWN, RWFNWN or WN
    """

    #--- Create control.txt file for EstimateTrend
    fp = open("estimatetrend.ctl", "w")
    fp.write("DataFile              {0:s}.mom\n".format(station))
    fp.write("DataDirectory         ./pre_files\n")
    fp.write("OutputFile            ./mom_files/{0:s}.mom\n".format(station))
    fp.write("interpolate           no\n")
    fp.write("PhysicalUnit          mm\n")
    fp.write("ScaleFactor           1.0\n")
    if noisemodel == 'GGMWN':
        fp.write("NoiseModels           GGM White\n")
    elif noisemodel == 'FNWN':
        fp.write("NoiseModels           FlickerGGM White\n")
        fp.write("GGM_1mphi             6.9e-06\n")
    elif noisemodel == 'PLWN':
        fp.write("NoiseModels           GGM White\n")
        fp.write("GGM_1mphi             6.9e-06\n")
    elif noisemodel == 'RWFNWN':
        fp.write("NoiseModels           RandomWalkGGM FlickerGGM White\n")
        fp.write("GGM_1mphi             6.9e-06\n")
    elif noisemodel == 'WN':
        fp.write("NoiseModels           White\n")
    elif noisemodel == 'AR1':
        fp.write("NoiseModels           ARMA\n")
        fp.write("AR_p                  1\n")
        fp.write("MA_q                  0\n")
    elif noisemodel == 'ARFIMA':
        fp.write("NoiseModels           ARFIMA\n")
        fp.write("AR_p                  1\n")
        fp.write("MA_q                  0\n")     
    elif noisemodel == 'AR1WN':
        fp.write("NoiseModels           ARMA White\n")
        fp.write("AR_p                  1\n")
        fp.write("MA_q                  0\n")
    elif noisemodel == 'ARFIMAWN':
        fp.write("NoiseModels           ARFIMA White\n")
        fp.write("AR_p                  1\n")
        fp.write("MA_q                  0\n")            
        
    
    else:
        print("Unknown noise model: {0:s}".format(noisemodel))
        sys.exit()
    fp.write("seasonalsignal        yes\n")
    fp.write("halfseasonalsignal    yes\n")
    fp.write(estimateoffsets)
    fp.write("estimatepostseismic   no\n")
    fp.write("estimateslowslipevent no\n")
    fp.write("ScaleFactor           1.0\n")
    fp.write("PhysicalUnit          mm\n")
    fp.close()



#===============================================================================
# Main program
#===============================================================================


#--- Read command line arguments
    
def analyse_timeseries_function(station,noisemodel,fraction=True,force_name=False):    
    
    """
    MODEL: GGMWN|PLWN|FNWN|RWFNWN|WN|AR1 + AR1WN|ARFIMA|ARFIMAWN
    
    if (len(sys.argv)<3 or len(sys.argv)>4) or \
       (len(sys.argv)==4 and sys.argv[3]!='fraction'):
        print('Correct usage: analyse_timeseries.py station_name ' + \
    			'GGMWN|PLWN|FNWN|RWFNWN|WN|AR1 [fraction]')
        sys.exit()
    else:
        station    = sys.argv[1]
        noisemodel = sys.argv[2]
        if (len(sys.argv)==4 and sys.argv[3]=='fraction'):
            fraction=True
        else:
            fraction=False
    """
    
    global estimateoffsets
    
    # standard - do not estimate offsets
    estimateoffsets="estimateoffsets       no\n"
    #estimateoffsets="estimateoffsets       no\n"
    
    
    #--- Check if the file exists 
    if os.path.isfile("./obs_files/{0:s}.mom".format(station))==False:
         print("Cannot find {0:s}.mom file in obs_files directory".format(station))
         sys.exit()
    
    #--- Does the mom-directory exists?
    if not os.path.exists('./pre_files'):
         os.makedirs('./pre_files')
    
    #--- Does the mom-directory exists?
    if not os.path.exists('./mom_files'):
         os.makedirs('./mom_files')
    
    #--- Remove outliers    
    create_removeoutliers_ctl_file(station)
    os.system("removeoutliers > removeoutliers.out")
    
    #--- Run estimatetrend
    create_estimatetrend_ctl_file(station,noisemodel)
    os.system("estimatetrend > estimatetrend.out")
    
    #--- Extract the results from estimatetrend.out
    output = extract_results(noisemodel,fraction)
    for i in range(0,len(output)-1):
        sys.stdout.write('{0:s} '.format(output[i]))
    #print('{0:s}'.format(output[len(output)-1]))
    
    #output = [trend,trend_error,N,LogL,aic,bic,bic_c,Sa_cos,Sa_sin, \
    #           Sa_cos_sigma, Sa_sin_sigma, Ssa_cos, Ssa_sin, Ssa_cos_sigma, \
	#			    			   Ssa_sin_sigma,std]
    
    return output

