#!/usr/bin/env python3
# 
# Convert sol-files to my mom-format. The sol-files are stored in the
# ./ori_files directory. The mom-files will be stored in the ./raw_files
# directory.
#
# NOTE:
# -----
# 1) Daily solutions are assumed. If not, update sampling period.
# 2) The East, North and Up displacements are counted relative to the first
#    observation which is set to zero.
# 3) Observations with an associated error larger than 1 m are skipped.
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



#--------------------------------
def cartesian_to_geodetic(x,y,z):
#--------------------------------
    """ Convert Cartesian coordinates XYZ to the geodetic frame
    :param X Cartesian coordinate
    :param Y Cartesian coordinate
    :param Z Cartesian coordinate
    :returns [E,N,U]
    """

    #--- WGS84 constants
    a = 6378137.0
    f = 1.0/298.257223563

    #--- derived constants
    b = a - f*a
    e = math.sqrt(a*a-b*b)/a

    lamda = math.atan2(y,x)
    p     = math.sqrt(x*x + y*y)
    h_old = 0.0
    theta = math.atan2(z,p*(1.0-e*e)) # first guess with h=0 metres
    cs    = math.cos(theta)
    sn    = math.sin(theta)
    n     = a*a/math.sqrt(math.pow(a*cs,2.0) + math.pow(b*sn,2.0))
    h     = p/cs - n
    while abs(h-h_old)>1.0e-6:
        h_old = h
        theta = math.atan2(z,p*(1.0-e*e*n/(n+h)))
        cs    = math.cos(theta)
        sn    = math.sin(theta)
        n     = a*a/math.sqrt(math.pow(a*cs,2.0) + math.pow(b*sn,2.0))
        h     = p/cs - n

    return [lamda,theta,h]



#===============================================================================
# Main program
#===============================================================================

#--- Constant
deg = 45.0/math.atan(1.0)

#--- Read all filenames in ./ori_files
fnames = glob.glob("./ori_files/*.sol")
fnames.sort()

#--- Did we find some sol-files?
if len(fnames)==0:
    print("Did not found any sol-files in the ./ori_files directory")
    sys.exit()

#--- Does the raw_files exists?
if not os.path.exists('./raw_files'):
    os.mkdir('./raw_files')

#--- Save file pointers in array
fp_out = [None]*3

#--- convert each sol-file
for fname in fnames:

    #--- Extract station name
    m = re.search('/(\w+)\.sol',fname)
    if m:
       station = m.group(1)
    else:
        print("Cannot figure out filename of: {0}".format(fname))
        sys.exit()

    print("{0:s}".format(station))

    for i in range(0,3):
        #--- Open output file
        fp_out[i] = open("./raw_files/{0:s}_{1:d}.mom".format(station,i),"w")

        #--- Write header
        fp_out[i].write("# sampling period 1.0\n")

    first_value = True
    with open('./ori_files/{0:s}.sol'.format(station)) as fp_in:
        for line in fp_in:
            cols = line.split();
            yearfraction = float(cols[1])
            mjd     = get_MJD(yearfraction)
            x       = float(cols[2])
            y       = float(cols[3])
            z       = float(cols[4])
            sigma_x = float(cols[5])
            sigma_y = float(cols[6])
            sigma_z = float(cols[7])
            if x==0.0 and y==0.0 and z==0.0:
                print("zero entree at {0:f}".format(yearfraction))
            elif sigma_x>1.0 or sigma_y>1.0 or sigma_z>1.0:
                print("large error at {0:f}".format(yearfraction))
            else:
                [lamda,theta,h] = cartesian_to_geodetic(x,y,z)
                lon = deg*lamda
                lat = deg*theta
                if first_value==True:
                    first_value = False
                    cl = math.cos(lamda)
                    sl = math.sin(lamda)
                    ct = math.cos(theta)
                    st = math.sin(theta)
                    x0 = x
                    y0 = y
                    z0 = z
                    for i in range(0,3):
                        fp_out[i].write("{0:10.1f} {1:12.3f}\n".format(mjd,0.0))
                else:
                    x -= x0
                    y -= y0
                    z -= z0

                    #--- rotate and convert metres to millimetres
                    e = 1000.0*(-sl*x    + cl*y          )
                    n = 1000.0*(-cl*st*x - st*sl*y + ct*z)
                    u = 1000.0*(   cl*ct*x + ct*sl*y + st*z)

                    fp_out[0].write("{0:10.1f} {1:12.3f}\n".format(mjd,e))
                    fp_out[1].write("{0:10.1f} {1:12.3f}\n".format(mjd,n))
                    fp_out[2].write("{0:10.1f} {1:12.3f}\n".format(mjd,u))


    fp_out[0].close()
    fp_out[1].close()
    fp_out[2].close()
