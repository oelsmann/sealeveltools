#!/usr/bin/env python3
# 
# For given station name, find in the time series (1 or 3 components) offsets.
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
import time
import math

#===============================================================================
# Subroutines
#===============================================================================


#---------------------
def extract_results():
#---------------------
    """ Extract estimated noise parameters from findoffset.txt.

    Returns:
        return list of estimated parameters
    """

    #--- Define variables we need to send back
    trend = trend_error = mjd = bic_c = None

    #--- Read each line from file and see if it contains a label we need
    with open("findoffset.txt",'r') as fp:
        for line in fp:
            #--- This is the first BIC_c mentioned in the output and 
            #    corresponds to the situation before adding a new offset
            m = re.match("^BIC_c\s+=\s*(-?\d+\.?\d*)",line)
            if m:
                bic_c = float(m.group(1))
            m = re.match("^FindOffset MJD\s+=\s*(\d+\.?\d*)",line)
            if m:
                mjd = float(m.group(1))
            m = re.match("^trend:\s+(-?\d+\.?\d*) \+\/\- (\d+\.?\d*)",line)
            if m:
                trend       = float(m.group(1))
                trend_error = float(m.group(2))

    #--- Construct output
    output = [trend, trend_error, mjd, bic_c]

    return output



#----------------------------------------
def create_removeoutliers_ctl_file(comp):
#----------------------------------------
    """ Create ctl file for removeoutlier.

    Args:
        comp (integer): 0=East, 1=North and 2=Up
    """

    #--- Create control.txt file for removeoutliers
    fp = open("removeoutliers.ctl", "w")
    fp.write("DataFile            dummy_raw.mom\n")
    fp.write("DataDirectory       ./\n")
    fp.write("interpolate         no\n")
    fp.write("OutputFile          dummy{0:d}_0.mom\n".format(comp))
    fp.write("seasonalsignal      yes\n")
    fp.write("halfseasonalsignal  yes\n")
    fp.write("estimateoffsets     yes\n")
    fp.write("ScaleFactor         1.0\n")
    fp.write("PhysicalUnit        mm\n")
    fp.write("IQ_factor           3\n")
    fp.close()



#----------------------------------------------------------------
def create_findoffset_ctl_file (comp,i,noisemodel,extra_penalty):
#----------------------------------------------------------------
    """ Create ctl file for findoffset.

    Args:
        comp (integer): 0=East, 1=North and 2=Up
        i (integer): number of iteration
        noisemodel (string): PLWN, FNWN, RWFNWN or WN
    """

    #--- Create control.txt file for EstimateTrend
    fp = open("findoffset.ctl", "w")
    fp.write("DataFile            dummy{0:d}_{1:d}.mom\n".format(comp,i))
    fp.write("OutputFile          output.mom\n")
    fp.write("DataDirectory       ./\n")
    fp.write("interpolate         no\n")
    fp.write("PhysicalUnit        mm\n")
    fp.write("ScaleFactor         1.0\n")
    if noisemodel == 'FNWN':
        fp.write("NoiseModels         FlickerGGM White\n")
    elif noisemodel == 'PLWN':
        fp.write("NoiseModels         GGM White\n")
    elif noisemodel == 'RWFNWN':
        fp.write("NoiseModels         RandomWalkGGM FlickerGGM White\n")
    elif noisemodel == 'WN':
        fp.write("NoiseModels         White\n")
    else:
        print("Unknown noise model: {0:s}".format(noisemodel))
        sys.exit()
    fp.write("seasonalsignal      yes\n")
    fp.write("halfseasonalsignal  yes\n")
    fp.write("estimateoffsets     yes\n")
    fp.write("ScaleFactor         1.0\n")
    fp.write("PhysicalUnit        mm\n")
    if comp<2:
        fp.write("beta_size           5.0\n")
    else:
        fp.write("beta_size           10.0\n")
    fp.write("beta_spacing        8.5\n")
    fp.write("offsets_3D          yes\n")
    fp.write("GGM_1mphi           6.9e-06\n")
    fp.write("BIC_c_ExtraPenalty  {0:f}\n".format(extra_penalty))
    fp.close()



#-----------------------------------------
def add_offsets_to_header(comp,i,offsets):
#-----------------------------------------
    """ Add newly found offsets to header.

    Args:
        comp (integer): 0=East, 1=North and 2=Up
        i (integer): number of iteration
        noisemodel (string): PLWN, FNWN, RWFNWN or WN
    """
 
    fp_out = open("dummy{0:d}_{1:d}.mom".format(comp,i),"w")
    header = 1
    with open("dummy{0:d}_0.mom".format(comp),'r') as fp_in:
        for line in fp_in:
            if header==1 and line.startswith('#')==False:
                for mjd in offsets:
                    if mjd>1.0:
                        fp_out.write("# offset {0:f}\n".format(mjd))
                header = 0
            fp_out.write(line)
    
    fp_out.close



#------------------------
def find_minimum(n_comp):
#------------------------
    """ Sum BICc for three components and pick epoch (MJD) with minimum.

    Args:
        n_comp - 1 or 3 components

    Returns:
        list with [mjd,value] at minimum
    """

    mjds = []
    values = []
    for i in range(0,n_comp):
        j = 0
        with open('findoffset_{0:1d}.out'.format(i),'r') as fp:
            for line in fp:
                cols = line.split()
                mjd  = float(cols[0])
                value= float(cols[1])
                if i==0:
                    mjds.append(mjd)
                    values.append(value)
                else:
                    if math.fabs(mjd-mjds[j])>1.0e-6:   
                        print('Huh? MJDs are not equal {0:f} - {1:f}'. \
							format(mjd,mjds[j]))
                    else:
                        values[j] = values[j] + value
                        j         = j+1

    j_min = values.index(min(values))
    print('--> mjd={0:f},  value={1:f}'.format(mjds[j_min],values[j_min]))

    return mjds[j_min]



#-----------------------
def make_equal_length():
#-----------------------
    """ Read the three dummy0/1/2_0.mom files and make them equal length.
    """

    #--- Define arrays
    MJD    = [None]*3
    obs    = [None]*3
    header = [None]*3
    fp     = [None]*3
    n      = [0]*3
    index  = [0]*3

    #--- Read files into memory
    for i in range(0,3):
        MJD[i]    = []
        obs[i]    = []
        header[i] = []

        fp[i] = open("dummy{0:1d}_0.mom".format(i),"r")
        lines = fp[i].readlines()
        fp[i].close()
        for line in lines:
            if line.startswith('#')==False:
                columns = line.split()
                MJD[i].append(columns[0])
                obs[i].append(columns[1])
            else:
                header[i].append(line)

    #--- length of each time series, open output file and write header
    for i in range(0,3):
        n[i]  = len(MJD[i])
        fp[i] = open("dummy{0:1d}_0.mom".format(i),"w")
        for line in header[i]:
            fp[i].write(line)

    #--- Only copy lines into new file if MJD is present in 3 components
    while index[0]<n[0] and index[1]<n[1] and index[2]<n[2]:
        if MJD[0][index[0]]==MJD[1][index[1]] and \
					   MJD[0][index[0]]==MJD[2][index[2]]:
            for i in range(0,3):
                fp[i].write("{0:s}  {1:s}\n". \
				     format(MJD[i][index[i]],obs[i][index[i]]))
                index[i] = index[i] + 1
        elif MJD[0][index[0]]<MJD[1][index[1]] or \
					     MJD[0][index[0]]<MJD[2][index[2]]:
            index[0] = index[0] + 1
        elif MJD[1][index[1]]<MJD[0][index[0]] or \
					     MJD[1][index[1]]<MJD[2][index[2]]:
            index[1] = index[1] + 1
        elif MJD[2][index[2]]<MJD[0][index[0]] or \
					     MJD[2][index[2]]<MJD[0][index[0]]:
            index[2] = index[2] + 1
        else:
            print("This should not happen...")
            sys.exit()
    
    #--- Close the new files     
    for i in range(0,3):
        fp[i].close()



#===============================================================================
# Main program
#===============================================================================


#--- Read command line arguments
if len(sys.argv)<3 or len(sys.argv)>5:
    print('Correct usage: find_offset.py station_name PLWN|FNWN|RWFNWN|WN [3D]' +
							' [penalty]')
    sys.exit()
else:
    station    = sys.argv[1]
    noisemodel = sys.argv[2]
    if len(sys.argv)==4:
        if sys.argv[3]=='3D':
            n_comp = 3
        else:
            n_comp = 1
            extra_penalty = float(sys.argv[3])
    elif len(sys.argv)==5:
        if sys.argv[3]=='3D':
            n_comp = 3
        else:
            print('Only accept 3D as 4th argument if 5th one is given as well')
            sys.exit()
        extra_penalty = float(sys.argv[4])
    else:
        n_comp        = 1
        extra_penalty = 8.0


#--- Just for fun, also note down how long everything takes
start = time.time()

#--- Create empty arrays
bic_c   = []
offsets = []
misfit  = []

#--- Check if file for the 1 or 3 components exist and remove outliers
for comp in range(0,n_comp):

    #--- Construct filename
    if n_comp==1:
        name = station
    else:
        name = '{0:s}_{1:d}'.format(station,comp)

    #--- check file existence
    if os.path.isfile("./raw_files/{0:s}.mom".format(name))==False:
         print("Cannot find {0:s}.mom file in raw_files directory".
								format(name))
         sys.exit()
    

    #--- Copy file to dummy_raw.mom and run removeoutliers over it
    os.system("cp -f ./raw_files/{0:s}.mom dummy_raw.mom".format(name))
    create_removeoutliers_ctl_file(comp)
    status = os.system("removeoutliers")


#--- Make equal lengths if 3 components are used at the same time
if n_comp==3:
    make_equal_length()

#--- First test with no offset
offsets.append(0.0)
bic_c_0 = 0.0
for comp in range(0,n_comp):
    create_findoffset_ctl_file(comp,0,noisemodel,extra_penalty)
    os.system("findoffset > findoffset.txt")
    output = extract_results()
    print("MJD={0:f}, trend={1:f}, BIC_c={2:f}".\
					format(output[2],output[0],output[3]))
    #--- Add BIC_c value (associated before new jump is found) to total
    bic_c_0 = bic_c_0 + output[3]

    #--- Save results
    os.system('mv findoffset.out findoffset_{0:1d}.out'.format(comp))

#--- For the case there are no offsets, use listed BIC_c
print("For first round (no new offsets added) BIC_c: {0:f}".format(bic_c_0))
bic_c.append(bic_c_0)

#--- Next offset location
mjd_min = find_minimum(n_comp)
offsets.append(mjd_min)


i=0
bic_c_old=bic_c[0]
#--- Now test for 1 to 8 breaks 
while i<8:

    #--- Add offsets to header
    i = i+1
    misfit_row = []
    bic_c_0 = 0.0
    for comp in range(0,n_comp):
        add_offsets_to_header(comp,i,offsets)

        #--- Look at the effect of new offset
        create_findoffset_ctl_file(comp,i,noisemodel,extra_penalty)
        os.system("findoffset > findoffset.txt")
        output = extract_results()
        print("MJD={0:f}, trend={1:f}, BIC_c={2:f}".\
					format(output[2],output[0],output[3]))
        #--- Add BIC_c value (associated before new jump is found) to total
        bic_c_0 = bic_c_0 + output[3]

        #--- Save results
        os.system('mv findoffset.out findoffset_{0:1d}.out'.format(comp))

    #--- Save misfits for offset i
    print("For offsets {0:1d} BIC_c: {1:f}".format(i,bic_c_0))
    bic_c.append(bic_c_0)

    #--- Prepare next offset location and already save it, together with BIC_c
    mjd_min = find_minimum(n_comp)
    offsets.append(mjd_min)

    #--- Should we stop
    if bic_c[i]>=bic_c_old:
        break

    #--- Else prepare next round
    bic_c_old = bic_c[i]


#--- Remember number of offsets estimated
n=i

#--- Show results
for i in range(0,n+1):
    print("{0:1d} MJD={1:10.1f} BIC_c={2:10.2f}". \
						format(i,offsets[i],bic_c[i]))

#--- Save found breaks to file
fp = open("findoffset_BIC_c.dat","w")
for i in range(0,bic_c.index(min(bic_c))+1):
    fp.write("{0:10.1f} {1:11.3f}\n".format(offsets[i],bic_c[i]))
fp.close()

#--- Does the obs_files directory exists?
if not os.path.exists('./obs_files'):
    os.mkdir('./obs_files')

#--- Save time series with offsets in header
k = bic_c.index(min(bic_c))
for comp in range(0,n_comp):
    if n_comp==1:
        os.system("cp -f dummy{0:1d}_{1:1d}.mom obs_files/{2:s}.mom". \
		         				 format(comp,k,name))
    else:
        os.system("cp -f dummy{0:1d}_{1:1d}.mom obs_files/{2:s}". \
		         format(comp,k,station) + "_{0:1d}.mom".format(comp))

#--- Finally, show computation time
finish = time.time()
dif = finish - start
print("Computation time in seconds: {0:f}".format(dif)) 

#--- Clean up dummy files
os.system('rm -f dummy*.mom')
