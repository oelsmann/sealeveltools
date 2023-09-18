#!/usr/bin/env python3
# 
# Convert neu-files available at SOPAC and JPL to my mom-format. The neu-files 
# are stored in the ./ori_files directory. The mom-files will be stored in 
# the ./raw_files directory.
#
# NOTE:
# -----
# 1) Daily solutions are assumed. If not, update sampling period.
# 2) The East, North and Up displacements are counted relative to the first
#    observation which is set to zero.
#
# Machiel Bos, 26/10/2018, Santa Clara
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
import glob
import os
import re
import math

#===============================================================================
# Subroutines
#===============================================================================


#--------------------------
def get_MJD (yearfraction):
#--------------------------
    """ Compute MJD from yearfraction
    :param yearfraction: time is given as a year fraction
    :returns mjd
    """
    #--- Compute MJD at beginning of year
    # http://scienceworld.wolfram.com/astronomy/JulianDate.html
    y   = int(math.floor(yearfraction))  # year
    m   = 1;                             # month
    d   = 1;                             # day
    mjd = 367*y - int(7*(y+int((m+9)/12))/4) + int(275*m/9) + d + \
							1721014 - 2400001
                                     
    #--- Add the days
    if y%4==0:
        mjd += 366.0*(yearfraction-y)
    else:
        mjd += 365.0*(yearfraction-y)

    return mjd


#===============================================================================
# Main program
#===============================================================================

#--- Read all filenames in ./ori_files
fnames = glob.glob("./ori_files/*.neu")
fnames.sort()

#--- Did we find some neu-files?
if len(fnames)==0:
    print("Did not found any neu-files in the ./ori_files directory")
    sys.exit()

#--- Does the raw_files exists?
if not os.path.exists('./raw_files'):
    os.mkdir('./raw_files')

#--- Save file pointers in array
fp_out = [None]*3

#--- For each station, convert rad/lon/lat to mom format
for fname in fnames:
    m = re.search("\/(\w+)\.neu",fname)
    if m:
        station = m.group(1)
    else:
        print("Cannot figure out filename of: {0}".format(fname))
        sys.exit()

    print("{0:s}".format(station))

    #--- Open the three files for output writing (E,N and Up)
    for i in range(0,3):
        #--- Open output file
        fp_out[i] = open("./raw_files/{0:s}_{1:d}.mom".format(station,i),"w")

        #--- Write header
        fp_out[i].write("# sampling period 1.0\n")

    first_value = True
    with open('./ori_files/{0:s}.neu'.format(station)) as fp_in:
        for line in fp_in:
            if not line.startswith('#'):
                cols = line.split();
                yearfraction = float(cols[0])
                mjd  = get_MJD(yearfraction)
                n    = float(cols[1])
                e    = float(cols[2])
                u    = float(cols[3])

                #--- Save first position
                if first_value==True:
                    first_value=False
                    e0 = e
                    n0 = n
                    u0 = u

                #--- Subtract first position
                e -= e0
                n -= n0
                u -= u0

                #--- Write relative position to file (m -> mm)
                fp_out[0].write("{0:8.1f} {1:8.2f}\n".format(mjd,1000.0*e))
                fp_out[1].write("{0:8.1f} {1:8.2f}\n".format(mjd,1000.0*n))
                fp_out[2].write("{0:8.1f} {1:8.2f}\n".format(mjd,1000.0*u))


    #--- Close mom files
    fp_out[0].close()
    fp_out[1].close()
    fp_out[2].close()
