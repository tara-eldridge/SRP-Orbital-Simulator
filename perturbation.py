import numpy as np
from control import pause
# This file contains the functions pertaining to the perturbations that
# can act on a satellite (or spacecraft) orbit around the Earth.

#---------------------------------------------------------------------------------
# Perturbation from other bodies in the system, often called 3rd-body perturbation

# Input data
#   GMd - gravitational parameter of the disturbing body [km^3/s^2]
#   rsc_vec - spaceraft's geocentric state vector [km]
#   rd_vec - geocentric state vector of the disturbing body [km]
# Output data
#   a3rb - acceleration due to the force caused by the disturbing body [km/s]
def AC3b (rsc_vec, rd_vec, GMd):

  # Magnitude of the position vector of the disturbing body
  rd = np.sqrt(rd_vec[0]**2 + rd_vec[1]**2 + rd_vec[2]**2)

  # relative position vector spacecraft -> disturber
  rho_vec = rd_vec - rsc_vec
  rho = np.sqrt(rho_vec[0]**2 + rho_vec[1]**2 + rho_vec[2]**2)

  # Acceleration due to the disturbing body
  a3rb = np.zeros(3)
  a3rb[0] = GMd*(rho_vec[0]/rho**3 - rd_vec[0]/rd**3)
  a3rb[1] = GMd*(rho_vec[1]/rho**3 - rd_vec[1]/rd**3)
  a3rb[2] = GMd*(rho_vec[2]/rho**3 - rd_vec[2]/rd**3)

  return a3rb

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
# Function to calculate the acceleration due to the direct solar radiation
# pressure (SRP) with shadow effect.
#
# Author:  Diogo Merguizo Sanchez
#          The University of Oklhoma
#          dmsanchez@ou.edu
#          2023
# 
# Input data: - satellite's position vector (km)
#             - Sun's position vector (km)
#             - satellite's area-to-mass ratio (m^2/kg)
#             - satellite's reflectivity coefficient (1.0 <= ref_co <= 2.0)
#             - Reflectivity coefficient (n.d.)
#             - Central body's equatorial or effecive radius
# Output data: - SRP acceleration vector (km/s^2)
#
# References: Beutler, 2004; Alsemo & Pardini, 2007; Delsate & Compère, 2012
#             Montenbruck & Gill, 2001

def SRPacc(xsat, xsun, AtoM, Cr, Re):

  PI = np.pi

  AU = 149597870.700 # km (From the 2023 Astronomical Almanac)
  Psun  = 4.56316e-6 # N/m^2 # Solar radiation pressure constant (at 1 au)
  reqsun = 695700.0 # km # Sun's volumetric mean radius - JPL Horizons rev. 2013

  SRP0 = Psun * AtoM * Cr * 1e-3 # km/s^2 (be sure to have the right units)

  Rsat = np.sqrt(xsat[0]**2 + xsat[1]**2 + xsat[2]**2)

  # Relative position spacecraft ---> Sun
  xsatsun = xsun - xsat
  Rsatsun = np.sqrt(xsatsun[0]**2 + xsatsun[1]**2 + xsatsun[2]**2)


  # --------------Shadow effect coefficient calculation (Montenbruck & Gill, 2001)-------------------

  # Apparent radius of the Sun
  a = np.arcsin(reqsun/Rsatsun)

  # Apparent radius of the planet
  b = np.arcsin(Re/Rsat)

  # Apparent separation of the centers of both bodies
  dotprod = - np.dot(xsat, xsatsun)
  c = np.arccos(dotprod/(Rsat*Rsatsun))

  if (a + b) <= c:
    nu = 1.0
  elif (c < (a - b)) or (c < (b - a)):
    nu = 0.0
  else:
    x = (c**2 + a**2 - b**2)/(2.0*c)

    y = np.sqrt(a**2 - x**2)

    # Occulted area
    Area = a**2 * np.arccos(x/a) + b**2 * np.arccos((c - x)/b) - c*y

    nu = 1.0 - Area/(PI*a**2)

  if np.isnan(nu):
    print("PROBLEM in SRP: nu is NaN.")
    exit()


  # Acceleration due to the SRP

  factor = (AU/Rsatsun)**2

  SRP = np.zeros(3)
  
  SRP =  - nu * SRP0 * factor * xsatsun/Rsatsun

  return SRP

'''
To be included in the main program:
#----------------------### Solar Radiation Pressure ###-------------------------
AtoM = 0.02 # m^2/kg
Cr = 1.5 # reflectivity coefficient (1.0 <= ref_co <= 2.0)
#-------------------------------------------------------------------------------
'''
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Gravitational potentitial of Mercury expanded in spherical harmonics.
#
# Author:  Diogo Merguizo Sanchez
#          The University of Oklhoma
#          dmsanchez@ou.edu
#          2023
# 
# Common library
# import numpy as np


#-------------------------------------------------------------------------------
# Function sidereal_time
'''
This function calculates the sidereal time of Mercury, i.e., the rotation angle
of Mercury given a spin rate and the time

Input: t - time (s)
       spin - sidereal rotation rate of the planet (rad/s)
       srtime0 - initial sidereal angle (rad)
Return: srtime - sidereal time of the planet (rad)
'''
def sidereal_time (t, spin, srtime0):
  pi2 = 2.0*np.pi

  srtime = srtime0 + spin*t
  srtime = srtime%pi2
  if srtime < 0:
    srtime = srtime + pi2
  
  return srtime
#-------------------------------------------------------------------------------
# Function Legendre
'''
This function calculates the Set the Normalized Legendre Functions and 
Associated Functions
Method: Standard Forward Column Method (Holmes & Featherstone, 2002)

Input: n, m - current order and degree of the Legendre functions
       nmax, mmax - maximum order and degree of the Legendre functions
       st, ct - sine and cosine of the planet-centric colatitude
       P, Pl - Current Normalized Legendre Functions and Associated
               Functions, and their derivatives

Return: P[n][m], Pl[n][m] - Updated Normalized Legendre Functions and Associated
                            Functions, and their derivatives
'''
def Legendre (n, m, st, ct, P, Pl):

  dn = float(n)
  dm = float(m)

  if n < 1:
    print("Legendre: error: n < 1")
    exit()
  
  if n == m:
    q0 = (2.0*dm + 1.0)/(2.0*dm)
    P[m][m] = st*np.sqrt(q0)*P[m - 1][m - 1]
    Pl[m][m] = dm*ct/st*P[m][m]
  
  if n > m:
    q1 = (2.0*dn - 1.0)*(2.0*dn + 1.0)/((dn - dm)*(dn + dm))
    q2 = (2.0*dn + 1.0)*(dn + dm - 1.0)*(dn - dm - 1.0)/((dn - dm)*(dn + dm)*(2.0*dn - 3.0))
    anm = np.sqrt(q1)
    bnm = np.sqrt(q2)
    P[n][m] = anm*ct*P[n - 1][m] - bnm*P[n - 2][m]
    fnm = (2.0*dn + 1.0)/anm
    Pl[n][m] = dn*ct/st*P[n][m] - fnm*P[n - 1][m]/st

  return P[n][m], Pl[n][m]
#-------------------------------------------------------------------------------
# Function EGM20082 
'''
  This function calculates the vector acceleration due to the gravity field of
the Earth from Earth's spherical harmonics EGM2008.

Global: C(n,m) and S(n,m) - spherical harmonics coefficients (from EGM2008.in)

Input: t - time (s)
       nmax - maximum degree of the spherical harmonics coefficients
       mmax - maximum order of the spherical harmonics coefficients
       xi - position vector relative to the inertial system (km)
       GM - gravitational parameter (km^3/s^2)
       Re - mean equatorial radius of the planet
       spin - spin rate of the planet (rad/s)
       srtime0 - sidereal time of the planet (rad)

Return: ACG - acceleration vetor (km/s) due to Earth's potential relative to
              the inertial system

Reference for EGM2008:
https://earth-info.nga.mil

'''

def EGM2008 (nmax, mmax, xi, C, S, t, GM, Re, spin, srtime0):

  Rm = Re

  MM = nmax + 1 # maximum degree and order to allocate variables' size

  # Allocating the size of the variables
  P = np.zeros((MM, MM))
  Pl = np.zeros((MM, MM))
  sml = np.zeros(MM)
  cml = np.zeros(MM)
  x = np.zeros(3)

  # The equations for the potential model are written in a reference frame
  # centered in the planet and fixed in the planet
  # (PCPF - Planet Centered, Planet Fixed)

  mst = sidereal_time (t, spin, srtime0)
  # print(f"Sidereal angle = {np.rad2deg(mst)}")
  smst = np.sin(mst)
  cmst = np.cos(mst)

  # Inertial frame ---> PCPF:
  x[0] =   xi[0]*cmst + xi[1]*smst
  x[1] = - xi[0]*smst + xi[1]*cmst
  x[2] =   xi[2]
  r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)

  # Auxiliary variables
  ct = x[2]/r  # cos(theta), theta = planet-centric colatitude
  st = np.sqrt(1.0 - ct*ct) # sin(theta)
  lamb = np.arctan2(x[1], x[0]) # planet-centric longitude
  sl = np.sin(lamb)
  cl = np.cos(lamb)
  GMoR = GM/r
  q = Rm/r

  # Set the Normalized Legendre Functions and Associated Functions, 
  # and their derivatives
  P[0][0] = 1.0
  P[1][0] = 1.73205080756888*ct
  P[1][1] = 1.73205080756888*st
  Pl[0][0] = 0.0
  Pl[1][0] = - P[1][1]
  Pl[1][1] =   P[1][0]

  for m in range(0, nmax + 1):
    if m < 2:
      mi = 2
    if m >= 2:
      mi = m
    for n in range(mi, nmax + 1):
      P[n][m], Pl[n][m] = Legendre(n, m, st, ct, P, Pl)
      # print(f"P({n},{m}) = {P[n][m]}")

  # Set the initial values of sml(m) = sin(m*lamb) and cml(m) = cos(m*lamb)
  sml[0] = 0.0
  sml[1] = sl

  cml[0] = 1.0
  cml[1] = cl

  # The potential is first written in spherical coordinates

  Vl = 0.0
  Vt = 0.0
  Vr = 0.0

  for m in range (0, mmax + 1):
    XLC = 0.0
    XLS = 0.0
    XTC = 0.0
    XTS = 0.0
    XRC = 0.0
    XRS = 0.0

    if m < 2:
      mi = 2
    if m >= 2:
      mi = m   
    for n in range (mi, nmax + 1):
      dn = float(n)
      dm = float(m)
      qn = q**n

      XRC = XRC + (dn + 1.0)*qn*C[n][m]*P[n][m]
      XRS = XRS + (dn + 1.0)*qn*S[n][m]*P[n][m]

      XTC = XTC + qn*C[n][m]*Pl[n][m]
      XTS = XTS + qn*S[n][m]*Pl[n][m]

      XLC = XLC + qn*C[n][m]*P[n][m]
      XLS = XLS + qn*S[n][m]*P[n][m]

    if m > 1:
      cml[m] = cml[m-1]*cl - sml[m-1]*sl
      sml[m] = sml[m-1]*cl + cml[m-1]*sl

    Vr = Vr +    (cml[m]*XRC + sml[m]*XRS)
    Vt = Vt +    (cml[m]*XTC + sml[m]*XTS)
    Vl = Vl + dm*(sml[m]*XLC - cml[m]*XLS)

  Vr = - GMoR/r*Vr
  Vt =   GMoR*Vt
  Vl = - GMoR*Vl

  # Spherical coordinates to Cartesian (Earth-Centered, Earth-Fixed)
  ac_MCMF = np.zeros(3)
  ac_MCMF[0] = st*cl*Vr + ct*cl*Vt/r - sl*Vl/(r*st)
  ac_MCMF[1] = st*sl*Vr + ct*sl*Vt/r + cl*Vl/(r*st)
  ac_MCMF[2] =    ct*Vr -    st*Vt/r

  # Earth-Centered, Earth-Fixed --> inertial frame
  ACG = np.zeros(3)
  ACG[0] = ac_MCMF[0]*cmst - ac_MCMF[1]*smst
  ACG[1] = ac_MCMF[0]*smst + ac_MCMF[1]*cmst
  ACG[2] = ac_MCMF[2]

  return ACG
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Perturbation due to the oblateness only

# Input data
#   J2 - Earth's oblateness coefficient
#   Re - Earth's mean equatorial radius [km]
#   GM - Earth's gravitational parameter [km^3/s^2]
#   xi - body's geocentric position vector [km]
# Output data
# ACJ2 - acceleration due to the oblateness of the Earth [km/s^2]

def J2acc (GM, J2, Re, xi):

  r = np.sqrt(xi[0]**2 + xi[1]**2 + xi[2]**2)
  x = xi[0]
  y = xi[1]
  z = xi[2]

  fJ2 = 1.5*J2*GM*Re**2/r**4

  ACJ2 = np.zeros(3)
  ACJ2[0] = fJ2*(x/r)*(5.0*z**2/r**2 - 1.0)
  ACJ2[1] = fJ2*(y/r)*(5.0*z**2/r**2 - 1.0)
  ACJ2[2] = fJ2*(z/r)*(5.0*z**2/r**2 - 3.0)

  return ACJ2

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Perturbation due to the atmospheric drag
# Model: Basic static spherical exponential atmosphere
# References:
# Exponential variation of the atmosphere:
# King-Hele, D. G., "Satellite Orbits in an Atmosphere: Theory and Applications".
#                    Blackie and Son Ltd, 1987.
# Static model of the atmosphere:
# Jachia, L. G., "Thermospheric Temperature, Density, and Composition: New Models".
#                 Smithsonian Astrophys. Obs. Spec. Rpt. 375, 1977.
# Components of the velocity with respect to the atmosphere:
# Vallado, D. A., "Fundamentals of Astrodynamics and Applications". Springer, 2001.
# Input data:
#   Xsat - satellite's state vector [km, km/s]
#   Cd - drag coefficient [non-dimensional]
#   AtoM - area-to-mass ratio [m^2/kg]
#   spin - Earth's spin rate [rad/s]
# Output data:
#   ATMacc - acceleration due to the atmospheric drag [km/s^2]
#
def atm_drag (Xsat, Cd, AtoM, spin):
  # Import the atmospheric density model (USSA76)
  from USSA76 import rho_atm

  xsat = Xsat[0:3]
  vsat = Xsat[3:6]

  rsat = np.sqrt(xsat[0]**2 + xsat[1]**2 + xsat[2]**2)
  altitude  = rsat - 6378.1366 # km
  
  # Calculate the atmospheric density as a function of the altitude
  rho = rho_atm(altitude)*1.e9 # kg/km^3

  # Velocity relative to the atmosphere
  vrel_vec = np.zeros(3)
  vrel_vec[0] = vsat[0] + spin*xsat[1]
  vrel_vec[1] = vsat[1] - spin*xsat[0]
  vrel_vec[2] = vsat[2]
  # Magnitude of the relative velocity
  vrel = np.sqrt(vrel_vec[0]**2 + vrel_vec[1]**2 + vrel_vec[2]**2)

  # Adjusting the units of the area-to-mass ratio
  AtoM = AtoM*1.e-6 # km^2/kg

  f_atm = - 0.5*rho*Cd*AtoM*vrel # kg/km^3 * km^2/kg * km/s = 1/s

  # print (rho, Cd, AtoM, vrel, f_atm)
  
  ATMacc = np.zeros(3)
  ATMacc[0] = f_atm*vrel_vec[0] # 1/s * km/s = km/s^2
  ATMacc[1] = f_atm*vrel_vec[1]
  ATMacc[2] = f_atm*vrel_vec[2]
  
  # print(f"Z = {altitude} km, rho = {rho*1.e-9} kg/m^3, ATMacc = {ATMacc} (km/s^2)")

  return ATMacc