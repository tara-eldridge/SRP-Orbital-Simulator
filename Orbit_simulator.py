# Orbit simulator
#
# In this program, a satellite orbit around the Earth is simulated considering
# the Earth as the central body and the perturbations from the Sun, the Moon,
# the solar radiation pressure, and the gravitational effects of the
# Earth's spherical harmonics. The satellite is outfitted with a spherical solar sail which has
# an adjustable area-to-mass ratio.
#
# Authors: Diogo Merguizo Sanchez
#          The University of Oklahoma
#          2025
#          
#          Tara J. Eldridge
#          The University of Oklahoma
#          2025
#
# Input - initial orbital elements: a, initial semi-major axis (km)
#                                   e, initial eccentricity
#                                   i, initial inclination (deg)
#                                   omega, initial argument of perigee (deg)
#                                   Omega, initial RAAN (deg)
#                                   Ma, initial mean anomaly (deg)
#                                   JD0, initial Julian date for the natural bodies' initial conditions
#                                   tu, time unit: 0: seconds, 1: minutes, 2: hours, 3: days, 4: years
#                                   tf, final time (tu)
#                                   t_step, time step (tu)
#
# Input - initial satellite parameters: AtoM0, initial and minimum area-to-mass ratio (m^2/kg)
#                                       AtoMmax, maximum area-to-mass ratio (m^2/kg)
#                                       infmethod: method of inflation: 0: none, 1: binary, 2: linear
#                                       defmethod: method of deflation: 0: none, 1: binary, 2: linear
#                                       inflateratio: fraction of total period to inflate over (1/#)
#                                       deflateratio: fraction of total period to deflate over (1/#)
# 
# Output - satellite parameters propagated over time (AtoM, cosine)
#
# Libraries
import numpy as np
import timeit
from scipy.integrate import solve_ivp
import os
import matplotlib.pyplot as plt
# Self-made libraries
from control import pause

# Import the constants from the file "major_bodies_parameters.py"
from major_bodies_parameters import constants

# Import major bodies' physical parameters
from major_bodies_parameters import sun
from major_bodies_parameters import earth
from major_bodies_parameters import moon

# Import the major bodies' initial conditions
from major_bodies import load_mb

# Import the function to convert orbital elements to Cartesian coordinates
from functions import orb2xyz

# Import the function to convert Cartesian coordinates to orbital elements
from functions import xyz2orb

# Import the function to change the reference plane (ecliptic to equatorial - if needed)
from functions import ecl2equ

# Import the function to calculate the GST at a given epoch
from functions import gst0

# Import function with the perturbations acting on the satellite
from perturbation import SRPacc
from perturbation import AC3b
from perturbation import EGM2008
from perturbation import atm_drag

# Import funcion to calculate the Earth's J2 (oblateness) perturbation on the Moon
from perturbation import J2acc
########################################################################
########################################################################
########################################################################
#----------------------### Initial conditions ###-----------------------

fd = 'Results'

figurelabel = f'Trial'    #Label to put on all figure legends

# Satellite's initial conditions

#Balloon Inflation/Deflation Methods
inf_method = 2 # 0: no inflation, 1: binary inflation, 2: linear inflation
def_method = 2 # 0: no deflation, 1: binary deflation, 2: linear deflation

#Fraction of the total period to deflate/inflate over (1/#)
inflateratio = 32
deflateratio = 32

#Number of steps over which linear inflation and deflation occurs
Num = 1000

# Aray to hold inflation and deflation times
tstarinf = np.zeros(Num+1)
tstardef = np.zeros(Num+1)

# Arrays to track the area-to-mass ratio of the satellite, time within the integrator, and cosine values
atomp = []
innertime = []
cosines = []
costhetabefore = 0
 
central_body = 'earth' # central body
frame = 'earth' # Options: Earth's mean equator [earth] or ecliptic [ecliptic] reference frame

a = 22000.0 #300 + earth().Re #8378.1366 #initial semi-major axis (km)
e = 0.05 # e, initial eccentricity
i = 0.001 # i, initial inclination (deg)
w = 0.0 # w, initial argument of perigee (deg)
OM = 0.0 # OM, initial RAAN (deg)
Ma = 0.0 # Ma, initial mean anomaly (deg)
epoch = '2025-01-01' # initial epoch (yyyy-mm-dd) for the natural bodies' initial conditions
tu  = 4 # tu, time unit: 0: seconds, 1: minutes, 2: hours, 3: days, 4: years
tf = 1 #2.12*60.0 # tf, final time (tu)
t_step = 10e-5 # t_step, time step (tu)

# Check if the perigee radius is greater than the Earth's radius
if a*(1.0 - e) < earth().Re:
    print("The perigee radius is less than the Earth's radius.")
    print("Please, change the initial conditions.")
    exit()

# Satellite properties for SRP calculations
AtoM0 = 0.04      # in m^2/kg, Minimum (Balloon is fully deflated)
AtoMmax = 1.0     # in m^2/kg, Maximum (Balloon is fully inflated)
AtoMD = AtoM0     #Initially, the balloon is fully deflated
deltaAtoM = (AtoMmax - AtoM0) / float(Num)
mass = 100.0 # kg
Cr = 1.88 # reflectivity coefficient (1.0 <= Cr <= 2.0)

# Satellite properties for atmospheric drag calculation
Cd = 2.2 # drag coefficient

# Earth's spherical harmonics (gravity model): using nmax = 2, mmax = 0 for oblateness
nmax = 2
mmax = 0

### Perturbations ###
# Flags to turn on/off the perturbations
# Solar radiation pressure
k_SRP = 1 # 1: on, 0: off
# Moon's perturbation
k_moon = 1 # 1: on, 0: off
# Sun's perturbation
k_sun = 1 # 1: on, 0: off
# Earth's spherical harmonics
k_EGM2008 = 1 # 1: on, 0: off
# Atmospheric drag
k_atm_drag = 1 # 1: on, 0: off

########################################################################
################## Control output and data analysis ####################
# Flags to plot the results (plots will be automatically saved)
plot_sv = 0 # plot the state vector: 0: no, 1: yes
plot_sv_3d = 0 # plot the state vector in 3D: 0: no, 1: yes
plot_sv_sun_3d = 0 # plot the state vector of the Sun in 3D: 0: no, 1: yes
plot_rp_ra = 0 # plot the perigee and apogee altitudes vs. time: 0: no, 1: yes
plot_sma = 1 # plot the semi-major axis vs. time: 0: no, 1: yes
plot_ecc = 1 # plot the eccentricity vs. time: 0: no, 1: yes
plot_period = 0 # plot the period of the orbit vs. time: 0: no, 1: yes
plot_cos = 0  #plot the cosine of theta of the orbit vs. time: 0: no, 1: yes
plot_atom = 0 # plot the area to mass ratio of the orbit vs. time: 0: no, 1: yes
plot_atom_cos = 0 #plot the area to mass ratio vs the cosine of theta: 0: no, 1: yes
#
# Flag to save satellite's initial conditions and code runtime in a text file: 0: no, 1: yes
save_initial = 1
# Flags to save the results
save_mb_sv = 1 # save major bodies' state vector: 0: no, 1: yes

# Save satellite's orbital elements
save_sat_oe = 1 # save satellite's orbital elements: 0: no, 1: yes

save_sat_atom_and_cos = 1 # save satellite's area-to-mass ratio and cosine data: 0: no, 1: yes
#

########################################################################
########################################################################
########################################################################
if tu == 0:
  tu_conv = 1.0
  unit = 'seconds'
elif tu == 1:
  tu_conv = 60.0
  unit = 'minutes'
elif tu == 2:
  tu_conv = 3600.0
  unit = 'hours'
elif tu == 3:
  tu_conv = 86400.0
  unit = 'days'
elif tu == 4:
  tu_conv = 86400.0*365.25
  unit = 'years'

tf = tf*tu_conv
dt = t_step*tu_conv

tu_conv

nt = round(tf/dt)
tspan = np.arange(0.0, tf + dt, dt)

# Load the satellite's initial orbital elements
oe_sat = np.zeros(7)
oe_sat[0] = a
oe_sat[1] = e
oe_sat[2] = np.deg2rad(i)
oe_sat[3] = np.deg2rad(w)
oe_sat[4] = np.deg2rad(OM)
oe_sat[5] = np.deg2rad(Ma)
# oe_sat[6] = period  #this is defined later

# Create a folder to store the results
os.system(f"mkdir -p {fd}")

# Save the satellite's initial conditions in a text file
if save_initial == 1:
  file2write=open(f'{fd}/Initial_Conditions.txt','w')
  file2write.write(f'Inflation method: {inf_method}\nDeflation Method: {def_method}\n\
  Inflation: 1/{inflateratio} of period\nDeflation: 1/{deflateratio} of period\n\
  Number of steps: {Num}\nSemi-major axis: {a} km\nEccentricity: {e}\n\
  Inclination: {i} degrees\nTime unit: {unit}\nFinal time: {tf/tu_conv} {unit}\nCr: {Cr}\n\
  Minimum area-to-mass ratio: {AtoM0} m^2/kg\nMaximum area-to-mass ratio: {AtoMmax} m^2/kg\n\
  Mass: {mass} kg')
  file2write.close()

########################################################################

# -------------------- Constants --------------------
const = constants()
G = const.G # gravitational constant [km^3/s^2]
au = const.au # astronomical unit [km]

#----------------------### Major bodies parameters ###-------------------------
# Load the major bodies' physical parameters
sun = sun()
earth = earth()
moon = moon()

# Major bodies' gravitational parameters
GM = np.zeros(3)
GM[0] = sun.GM # Sun's gravitational parameter [km^3/s^2]
GM[1] = earth.GM # Earth's gravitational parameter [km^3/s^2]
GM[2] = moon.GM # Moon's gravitational parameter [km^3/s^2]

# Earth Oblateness (for the J2 perturbation in the Moon)
J2 = earth.J2 # Earth's J2 coefficient

# Load the Earth's spherical harmonics coefficients
# The coefficients are stored in the file "EGM2008_upto50_TideFree.in"
# Full normalized coefficients up to degree and order 50
input = np.loadtxt('EGM2008_upto50_TideFree.in', skiprows=1)

C = np.zeros((nmax + 1, mmax + 1))
S = np.zeros((nmax + 1, mmax + 1))

# print(f"len(input[:,0]) = {len(input[:,0])}")

for i in range(0, len(input[:,0]) + 1):
    degree = input[i,0].astype('int')
    order = input[i,1].astype('int')
    C[degree, order] = input[i, 2]
    S[degree, order] = input[i, 3]
    # print(f"C({degree},{order}) = {input[i,2]:.14e}; S({degree},{order}) = {input[i,3]:.14e}")
    if degree == nmax and order == mmax:
        break

# Test if the variables were correctly loaded
# print(f"\nTest if the variables were correctly loaded:")
# for n in range(2, nmax + 1):
#     for m in range(0, nmax + 1):
#       if m <= n:
#         # print(f"C({n},{m}) = {C[n, m]:20.14e} | S({n},{m}) = {S[n, m]:20.14e}")
#         print("C({0},{1}) = {2:20.14e}, S({0},{1}) = {3:20.14e}".format(n, m, C[n, m], S[n, m]))
#---------------------------------------------------------------------------------------
# Convert the satellite's initial orbital elements to Cartesian coordinates
init_sc = orb2xyz(earth.GM, oe_sat)

period = 2.0*np.pi*np.sqrt(a**3/earth.GM)

print(f"\nSatellite's initial conditions wiht respect to the Earth:")
print(f"---------------------------------------------------------")
print(f"Initial semi-major axis (a): {oe_sat[0]:.5e} km")
print(f"Initial eccentricity (e): {oe_sat[1]:.5e}")
print(f"Initial inclination (i): {np.rad2deg(oe_sat[2]):.3f} deg")
print(f"Initial argument of perigee (w): {np.rad2deg(oe_sat[3]):.2f} deg")
print(f"Initial RAAN (OM): {np.rad2deg(oe_sat[4]):.2f} deg")
print(f"Initial mean anomaly (Ma): {np.rad2deg(oe_sat[5]):.2f} deg")
print(f"Initial position vector (x, y, z): {init_sc[0]:.5e}, {init_sc[1]:.5e}, {init_sc[2]:.5e} (km)")
print(f"Initial velocity vector (vx, vy, vz): {init_sc[3]:.5e}, {init_sc[4]:.5e}, {init_sc[5]:.5e} (km/s)")
print(f"Satellite's period: {period/3600:.2f} hours")
print(f"\nSatellite's physical properties:")
print(f"Area-to-mass ratio (A/m): {AtoM0:.5e} m^2/kg")
print(f"Reflectivity coefficient (Cr): {Cr}")
print(f"Drag coefficient (Cd): {Cd}")
print(f"---------------------------------------------------------\n")

# Load the major bodies' initial conditions (mb) and the inital Julian date (jd) at the epoch
mb, jd0 = load_mb(frame, epoch)

# Load the initial GST at the epoch
GST0 = np.deg2rad(gst0(jd0))

print(f"Initial Julian date: {jd0}")
print(f"Initial GST: {np.rad2deg(GST0)} deg")

# Major bodies' initial conditions
# Sun
Xb_sun = mb[0, :]
# Earth
Xb_earth = mb[1, :]
# Moon
Xb_moon = mb[2, :]
# Convert the Sun's initial conditions to orbital elements
mu = GM[0] + GM[1]
oe_sun = xyz2orb(mu, -Xb_sun[0:3], -Xb_sun[3:6])
# Earth is the central body, so the orbital elements are not defined
# Convert the Moon's initial conditions to orbital elements
mu = GM[1] + GM[2]
oe_moon = xyz2orb(mu, Xb_moon[0:3], Xb_moon[3:6])

print(f"\nMajor bodies' initial conditions:")
print(f"---------------------------------")
print(f"Sun:")
print(f"Position vector: {Xb_sun[0]:.5e}, {Xb_sun[1]:.5e}, {Xb_sun[2]:.5e} (km)")
print(f"R_sun = {np.linalg.norm(Xb_sun[0:2]):.8e} km")
print(f"Velocity vector: {Xb_sun[3]:.5e}, {Xb_sun[4]:.5e}, {Xb_sun[5]:.5e} (km/s)")
print(f"V_sun = {np.linalg.norm(Xb_sun[3:5]):.8e} km/s")
print(f"Orbital elements from state vector (Earth viewed from the Sun):")
print(f"a = {(oe_sun[0]):.8e} km")
print(f"e = {oe_sun[1]:.8e}")
print(f"i = {np.rad2deg(oe_sun[2]):.2f} deg")
print(f"w = {np.rad2deg(oe_sun[3]):.2f} deg")
print(f"Omega = {np.rad2deg(oe_sun[4]):.2f} deg")
print(f"Ma = {np.rad2deg(oe_sun[5]):.2f} deg")

print(f"\nMoon:")
print(f"Position vector: {Xb_moon[0]:.5e}, {Xb_moon[1]:.5e}, {Xb_moon[2]:.5e} (km)")
print(f"R_moon = {np.linalg.norm(Xb_moon[0:2]):.8e} km")
print(f"Velocity vector: {Xb_moon[3]:.5e}, {Xb_moon[4]:.5e}, {Xb_moon[5]:.5e} (km/s)")
print(f"V_moon = {np.linalg.norm(Xb_moon[3:5]):.8e} km/s")
print(f"Orbital elements from state vector:")
print(f"a = {(oe_moon[0]):.8e} km")
print(f"e = {oe_moon[1]:.8e}")
print(f"i = {np.rad2deg(oe_moon[2]):.2f} deg")
print(f"w = {np.rad2deg(oe_moon[3]):.2f} deg")
print(f"Omega = {np.rad2deg(oe_moon[4]):.2f} deg")
print(f"Ma = {np.rad2deg(oe_moon[5]):.2f} deg")
print(f"---------------------------------\n")

# Load the initial state vectors of all bodies into a single array (Xb)
# This array is used to initialize the integration process

# Size of the array Xb
neq = 6*3 # 6 elements for each of the 3 bodies (Sun, Moon, and satellite)
Xb_init = np.zeros(neq)

# Sun
Xb_init[0:6] = Xb_sun
# Moon
Xb_init[6:12] = Xb_moon
# Satellite
Xb_init[12:18] = init_sc

#-------------------------------------------------------------------------------
#-------------#### Event Functions to control the integration ####--------------
critical = 0
def Reentry(t, f):
    r = np.sqrt(f[12]**2 + f[13]**2 + f[14]**2)
    alt = r - earth.Re
    if alt <= 100.0:
      print(f"Reentry: t = {t/tu_conv} tu, r = {r} km")

    return alt - 100.0
Reentry.direction = -1
Reentry.terminal = True
#-------------------------------------------------------------------------------
#-----------------#### Function with derivatives (model) ####-------------------
def Derivs(t, f):
    
    Re = earth.Re

    x_sun = np.zeros(6)
    x_sun[0] = f[0]
    x_sun[1] = f[1]
    x_sun[2] = f[2]
    x_sun[3] = f[3]
    x_sun[4] = f[4]
    x_sun[5] = f[5]
    r_sun = np.sqrt(x_sun[0]**2 + x_sun[1]**2 + x_sun[2]**2)
    r_sun_vector = np.array([x_sun[0], x_sun[1], x_sun[2]])

    x_moon = np.zeros(6)
    x_moon[0] = f[6]
    x_moon[1] = f[7]
    x_moon[2] = f[8]
    x_moon[3] = f[9]
    x_moon[4] = f[10]
    x_moon[5] = f[11]
    r_moon = np.sqrt(x_moon[0]**2 + x_moon[1]**2 + x_moon[2]**2)

    x_sat = np.zeros(6)
    x_sat[0] = f[12]
    x_sat[1] = f[13]
    x_sat[2] = f[14]
    x_sat[3] = f[15]
    x_sat[4] = f[16]
    x_sat[5] = f[17]
    r = np.sqrt(x_sat[0]**2 + x_sat[1]**2 + x_sat[2]**2)
    r_vector = np.array([x_sat[0], x_sat[1], x_sat[2]])

    #Vector pointing from Sun to satellite
    r_sunsat_vector = np.array(r_vector - r_sun_vector)
    r_sunsat = np.sqrt(r_sunsat_vector[0]**2 + r_sunsat_vector[1]**2 + r_sunsat_vector[2]**2)

    # OE = xyz2orb(earth.GM, x_sat[0:3], x_sat[3:6])
    # print (f"t = {t/tu_conv}, r = {r}, a = {OE[0]}, e = {OE[1]}, i = {np.rad2deg(OE[2])}")

    # Sun EOM
    dxsundt = x_sun[3]
    dysundt = x_sun[4]
    dzsundt = x_sun[5]

    mu_sun_earth = sun.GM + earth.GM
    ddxsundt = - mu_sun_earth*x_sun[0]/r_sun**3
    ddysundt = - mu_sun_earth*x_sun[1]/r_sun**3
    ddzsundt = - mu_sun_earth*x_sun[2]/r_sun**3

    # Moon EOM
    dxmoondt = x_moon[3]
    dymoondt = x_moon[4]
    dzmoondt = x_moon[5]

    # Sun's perturbation on the Moon
    AC3b_sun = AC3b(x_moon[0:3], x_sun[0:3], sun.GM)
    # Earth's J2 perturbation on the Moon
    ACJ2 = J2acc(earth.GM, earth.J2, earth.Re, x_moon[0:3])
    mu_moon_earth = moon.GM + earth.GM
    ddxmoondt = - mu_moon_earth*x_moon[0]/r_moon**3 + ACJ2[0] + AC3b_sun[0]
    ddymoondt = - mu_moon_earth*x_moon[1]/r_moon**3 + ACJ2[1] + AC3b_sun[1]
    ddzmoondt = - mu_moon_earth*x_moon[2]/r_moon**3 + ACJ2[2] + AC3b_sun[2]

    sat_vel_vector = np.array([x_sat[3], x_sat[4], x_sat[5]])
    sat_vel = np.sqrt(x_sat[3]**2 + x_sat[4]**2 + x_sat[5]**2)

    # Dot product between the satellite velocity and Sun-satelite vector
    SRPdot = np.dot(r_sunsat_vector, sat_vel_vector)
  
    #Check that vectors are real
    if np.isnan(r_sunsat):
        pause()
    if np.isnan(sat_vel):
        pause()

    costheta = SRPdot / (sat_vel * r_sunsat)

    global t_step
    global tu_conv

    global AtoMD
    global AtoMmax
    global costhetabefore

    #Arrays to hold calculated values
    global atomp
    global innertime
    global cosines

    #---------- Satellite is moving towards the Sun (when costheta <= 0) ----------
    # INFLATION PROCESS

    if inf_method == 1 and costheta <= 0:     #Binary inflation is selected
      AtoMD = AtoMmax

    if inf_method == 2:     #Linear inflation is selected
      if (costheta * costhetabefore) <= 0 and costhetabefore > 0:    #Satellite has switched directions
        muE = earth.GM
        satE = (sat_vel**2/2) - (muE/r)
        sata = -muE / (2*satE)
        sat_period = 2 * np.pi * np.sqrt(sata**3/muE)   #Instantaneous period at time of switch

        tchange = sat_period / inflateratio      #Total amount of time for inflation
        tstart = t
        tstarinf[0] = tstart
        d = 1
        for d in range (Num+1):      #An array of the times for the number of intervals Num
            tstarinf[d] = tstart + (d * tchange / Num)
            d += 1

      if costheta <= 0:    #satellite is traveling towards the Sun (against the SRP)
        l = 0
        for l in range (Num):
          if t >= tstarinf[l] and t < tstarinf[l+1]:
            h = l
            AtoMD = AtoM0 + (deltaAtoM*h)
            l+=1

        if t > tstarinf[Num]:   #Time for total inflation is reached (maximum AtoM)
          AtoMD = AtoMmax


    #---------- Satellite is moving taway from the Sun (when costheta > 0) ----------
    # DEFLATION PROCESS
    if def_method == 1 and costheta > 0:     #Binary deflation is selected
      AtoMD = AtoM0

    if def_method == 2:     #Linear deflation is selected
      if (costheta * costhetabefore) < 0 and costhetabefore <= 0:    #satellite has switched directions
        muE = earth.GM
        satE = (sat_vel**2/2) - (muE/r)
        sata = -muE / (2*satE)
        sat_period = 2 * np.pi * np.sqrt(sata**3/muE)

        tchange = sat_period / deflateratio      #Total amount of time for inflation
        tstart = t
        tstardef[0] = tstart
        d = 1
        for d in range (Num+1):      #An array of the times for the number of intervals Num
            tstardef[d] = tstart + (d * tchange / Num)
            d += 1

      if costheta > 0:  #satellite is traveling away from the Sun (with the SRP)
        l = 0
        for l in range (Num):
          if t >= tstardef[l] and t < tstardef[l+1]:
            h = l
            AtoMD = AtoMmax - (deltaAtoM*h)
            l+=1

        if t > tstardef[Num]:   #Time for total deflation is reached (minimum AtoM)
          AtoMD = AtoM0

    #Adjust AtoM to become the minimum or maximum if needed
    if AtoMD > AtoMmax:
      AtoMD = AtoMmax
    if AtoMD < AtoM0:     
      AtoMD = AtoM0

    #
    #Update values
    costhetabefore = costheta
    
    atomp.append(AtoMD)
    innertime.append(t)
    cosines.append(costheta)

    # Solar radiation pressure
    if k_SRP == 1:
      acc_SRP = SRPacc(x_sat[0:3], x_sun[0:3], AtoMD, Cr, Re)
    else:
      acc_SRP = np.zeros(3)

    # print(f"t = {t/tu_conv}, acc_SRP = {acc_SRP} (km/s^2)")
    # pause()

    # Perturbation from the Moon
    if k_moon == 1:
      acc_Moon = AC3b(x_sat[0:3], x_moon[0:3], moon.GM)
    else:
      acc_Moon = np.zeros(3)

    # Perturation from the Sun
    if k_sun == 1:
      acc_Sun = AC3b(x_sat[0:3], x_sun[0:3], sun.GM)
    else:
      acc_Sun = np.zeros(3)

    # print(f"t = {t/tu_conv}, acc_Sun/Moon = {acc_Sun + acc_Moon} (km/s^2)")
    # pause()

    # Atmospheric drag
    if r <= (900.0 + Re) and k_atm_drag == 1:
      acc_atm = atm_drag (x_sat, Cd, AtoMD, earth.spin)
      # print(f"t = {t/tu_conv}, acc_atm = {acc_atm} (km/s^2)")
    else:
      acc_atm = np.zeros(3)

    # Perturbation from the Earth's spherical harmonics
    if k_EGM2008 == 1:
      ACG = EGM2008 (nmax, mmax, x_sat[0:3], C, S, t, earth.GM, earth.Re, earth.spin, GST0)
    else:
      ACG = np.zeros(3)

    # print(f"t = {t/tu_conv}, ACG = {ACG} (km/s^2)")
    # pause()
 
    # Spacecraft EOM
    dxdt = x_sat[3]
    dydt = x_sat[4]
    dzdt = x_sat[5]

    mu = earth.GM
    ddxdt = - mu*x_sat[0]/r**3 + ACG[0] + acc_Moon[0] + acc_Sun[0] + acc_SRP[0] + acc_atm[0]
    ddydt = - mu*x_sat[1]/r**3 + ACG[1] + acc_Moon[1] + acc_Sun[1] + acc_SRP[1] + acc_atm[1]
    ddzdt = - mu*x_sat[2]/r**3 + ACG[2] + acc_Moon[2] + acc_Sun[2] + acc_SRP[2] + acc_atm[2]

    return [dxsundt, dysundt, dzsundt, ddxsundt, ddysundt, ddzsundt, \
            dxmoondt, dymoondt, dzmoondt, ddxmoondt, ddymoondt, ddzmoondt, \
            dxdt, dydt, dzdt, ddxdt, ddydt, ddzdt]
#-------------------------------------------------------------------------------
# Integration process
start = timeit.default_timer()

# Print the information about the integration process
print(f"\nIntegration process:")
print(f"TF = {tf/tu_conv:.2f} {unit}")
print(f"dt = {dt/tu_conv:.2e} {unit}")
print(f"nt = {nt}")

print('\nRunning...\n')

solution = solve_ivp (Derivs, [0.0, tf + dt], Xb_init, events=[Reentry], method='LSODA', \
                     t_eval=tspan, first_step =dt/100.0, rtol = 1.e-10, atol = 1.e-12)

state = solution.y
times = solution.t

atomp = np.array(atomp)
innertime = np.array(innertime)
cosine = np.array(cosines)

x_sun = state[0:6, :]
x_moon = state[6:12, :]
x_sat = state[12:18, :]

stop = timeit.default_timer()
runtime = stop - start
if runtime < 60.0:
    print(f"Runtime = {runtime:.2f} seconds.\n")
elif runtime >= 60.0 and  runtime < 3600.0:
    print(f"Runtime = {runtime/60.0:.2f} minutes.\n")
else:
    print(f"Runtime = {runtime/3600.0:.2f} hours.\n")

if save_initial == 1:
  file2write = open(f'{fd}/Initial_Conditions.txt', 'a')  # append mode
  file2write.write(f'\nRuntime = {runtime:.2f} secs = {runtime/60.0:.2f} mins = {runtime/3600.0:.2f} hours')
  file2write.close()

################################################################################
################################################################################
################################################################################
# Data analysis

n = len(times)
# print (f"nt = {nt}")
# print (f"n = {n}")
# print (f"x_sat error = {x_sat[0, -1] - x_sat[0, 0]}")
# print (f"y_sat error = {x_sat[1, -1] - x_sat[1, 0]}")
# print (f"z_sat error = {x_sat[2, -1] - x_sat[2, 0]}\n")

# Plot the satellite's orbit (x-y plane)
if plot_sv == 1:
  plt.plot(x_sat[0, :], x_sat[1, :], linewidth=0.5)
  # plt.plot(state[6, :], state[7, :])
  plt.show()

# Plot the satellite's orbit (3D)
if plot_sv_3d == 1:
  # Plot the satellite's orbit in 3D
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(x_sat[0, :], x_sat[1, :], x_sat[2, :], linewidth=0.5)
  plt.show()
  fig.savefig(f'{fd}/sat-sv-3D.png', dpi=300)

# Plot the sun's orbit (3D)
if plot_sv_sun_3d == 1:
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(x_sun[0, :], x_sun[1, :], x_sun[2, :], linewidth=0.5)
  plt.show()

# Save the state vectors of the satellite in a file
filename = f'{fd}/state_vectors_sat.dat'
with open(filename, 'w') as f:
    for j in range(n):
        f.write(f"{times[j]:<15.8e} {x_sat[0, j]:<15.8e} {x_sat[1, j]:<15.8e} {x_sat[2, j]:<15.8e} {x_sat[3, j]:<15.8e} {x_sat[4, j]:<15.8e} {x_sat[5, j]:<15.8e}\n")

# Save the state vectors of the major bodies in a file
if save_mb_sv == 1:
    filename1 = f'{fd}/state_vectors_sun.dat'
    filename2 = f'{fd}/state_vectors_moon.dat'
    with open(filename1, 'w') as f:
        for j in range(n):
            f.write(f"{times[j]:<15.8e} {x_sun[0, j]:<15.8e} {x_sun[1, j]:<15.8e} {x_sun[2, j]:<15.8e} {x_sun[3, j]:<15.8e} {x_sun[4, j]:<15.8e} {x_sun[5, j]:<15.8e}\n")
    with open(filename2, 'w') as f:
        for j in range(n):
            f.write(f"{times[j]:<15.8e} {x_moon[0, j]:<15.8e} {x_moon[1, j]:<15.8e} {x_moon[2, j]:<15.8e} {x_moon[3, j]:<15.8e} {x_moon[4, j]:<15.8e} {x_moon[5, j]:<15.8e}\n")
       
# exit()

# Convert the state vector to orbital elements
orb = []
rp = np.zeros(n)
ra = np.zeros(n)
for j in range(n):
    oe_sat = xyz2orb(earth.GM, x_sat[0:3, j], x_sat[3:6, j])
    orb.append(oe_sat)

    # Calculate the perigee and apogee altitudes to test the decay due to the atmospheric drag
    rp[j] = oe_sat[0]*(1.0 - oe_sat[1])  - earth.Re
    ra[j] = oe_sat[0]*(1.0 + oe_sat[1])  - earth.Re
orb = np.array(orb)

# Save the area-to-mass ratio and cosine of theta for the satellite in a file
if save_sat_atom_and_cos == 1:
  n2 = len(innertime)
  filename2 = f'{fd}/sat_atom_and_cos.dat'
  with open(filename2, 'w') as f2:
    for j in range(n2):
      f2.write(f"{innertime[j]/tu_conv:<15.8e} {atomp[j]:<15.8e}  {cosines[j]:<15.8e}\n") 

# Save the orbital elements of the satellite in a file
if save_sat_oe == 1:
  filename = f'{fd}/orbital_elements_sat.dat'
  with open(filename, 'w') as f:
    for j in range(n):
      f.write(f"{times[j]/tu_conv:<15.8e} {orb[j][0]:<15.8e} \
      {orb[j][1]:<15.8e} {np.rad2deg(orb[j][2]):<15.8e} \
      {np.rad2deg(orb[j][3]):<15.8e} {np.rad2deg(orb[j][4]):<15.8e} \
      {np.rad2deg(orb[j][5]):<15.8e} {orb[j][6]/tu_conv:<15.8e}\n")  

# # Set font to Times New Roman 12pt.
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# plt.rcParams['font.size'] = 12 # Example font size
# plt.rcParams['mathtext.default'] = 'regular' # For math text

# Plot semi-major axis vs. time
if plot_sma == 1:
  fig = plt.figure()
  time = np.array(times)
  time = time/tu_conv
  plt.plot(time, orb[:, 0], linewidth=0.75, label=''+figurelabel+'', color='#000000')
  plt.xlabel('Time ('+unit+')')
  plt.ylabel('Semi-Major Axis (km)')
  # plt.title('Semi-Major Axis vs. Time')
  # plt.legend(bbox_to_anchor=(1, 1),bbox_transform=fig.transFigure)
  # plt.grid(True)
  plt.show()
  fig.savefig(f'{fd}/a-vs-t.png', dpi=300)

# Plot eccentricity vs. time
if plot_ecc == 1:
  fig = plt.figure()
  plt.plot(time, orb[:, 1], linewidth=0.75, label=''+figurelabel+'', color='#000000')
  plt.xlabel('Time ('+unit+')')
  plt.ylabel('Eccentricity')
  # plt.title('Eccentricity vs. Time')
  # plt.legend(bbox_to_anchor=(1, 1),bbox_transform=fig.transFigure)
  # plt.grid(True)
  plt.show()
  fig.savefig(f'{fd}/e-vs-t.png', dpi=300)

# Plot perigee and apogee altitudes vs. time
if plot_rp_ra == 1:
  plt.plot(time, rp, label='Perigee', linewidth=0.5)
  plt.plot(time, ra, label='Apogee', linewidth=0.5)
  plt.xlabel('Time ('+unit+')')
  plt.ylabel('Altitude (km)')
  plt.title('Perigee and Apogee Altitudes vs. Time')
  plt.legend()
  plt.grid(True)
  plt.show()
  fig.savefig(f'{fd}/rp-and-ra-vs-t.png', dpi=300)

# Plot the period vs. time
if plot_period == 1:
  plt.plot(time, orb[:, 6], linewidth=0.5)
  plt.xlabel('Time ('+unit+')') 
  plt.ylabel('Period')
  plt.title('Period vs. Time')
  plt.grid(True)
  plt.show()
  fig.savefig(f'{fd}/period-vs-t.png', dpi=300)

# Plot the area-to-mass ratio vs. time
if plot_atom == 1:
  fig = plt.figure()
  plt.plot(innertime/tu_conv, atomp, linewidth=0.5, label=''+figurelabel+'')
  plt.xlabel('Time ('+unit+')') 
  plt.ylabel(f'Area-to-Mass Ratio ($m^{2}$/kg)')
  plt.title('Area-to-Mass Ratio vs. Time')
  # plt.legend(bbox_to_anchor=(1, 1),bbox_transform=fig.transFigure)
  # plt.grid(True)
  plt.show()
  fig.savefig(f'{fd}/atom-vs-t.png', dpi=300)

# Plot the cosine of theta vs. time
if plot_cos == 1:
  fig = plt.figure()
  plt.plot(innertime/tu_conv, cosines, linewidth=0.5, label=''+figurelabel+'')
  plt.xlabel('Time ('+unit+')') 
  plt.ylabel('Cosine of Theta')
  plt.title('Cosine of Theta vs. Time')
  # plt.legend(bbox_to_anchor=(1, 1),bbox_transform=fig.transFigure)
  plt.grid(True)
  plt.show()
  fig.savefig(f'{fd}/cos-vs-t.png', dpi=300)

# Plot the area-to-mass ratio vs. cosine of theta
if plot_atom_cos == 1:
  plt.plot(cosines, atomp, linewidth=0.5)
  plt.xlabel('Cosine of Theta') 
  plt.ylabel(f'Area-to-Mass Ratio ($m^{2}$/kg)')
  plt.title('Area-to-Mass Ratio vs. Cosine of Theta')
  plt.grid(True)
  plt.show() 
  fig.savefig(f'{fd}/atom-vs-cos.png', dpi=300)   

