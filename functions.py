# Useful functions for orbital mechanics
#
# Author:  Diogo Merguizo Sanchez
#          The University of Oklhoma
#          dmsanchez@ou.edu
#          2023
# 
import numpy as np
def xyz2orb (mu, r_vec, v_vec):
    pi2 = 2.0*np.pi

    r = np.linalg.norm(r_vec) # magnitude of the position vector
    # v = np.linalg.norm(v_vec) # magnitude of the velocity vector
    
    h_vec = np.cross(r_vec, v_vec) # vector angular momentum
    h = np.linalg.norm(h_vec) # magnitude of the angular momentum
#     print(f"h = {h} km^2/s")
    
    p = h**2/mu # semi latus rectum
#     print(f"p = {p} km")
    
    # Eccentricity vector
    e_vec = np.cross(v_vec, h_vec)/mu - r_vec/r
    
    e = np.linalg.norm(e_vec) # eccentricity
#     print(f"e = {e}")
    
    a = p/(1.0 - e**2) # semi-major axis
#     print(f"a = {a} km")
    
    i = np.arccos(h_vec[2]/h) # inclination
#     print(f"i = {np.rad2deg(i)} deg")
    
    # Defining the direction k and the Nodal vector
    k_dir = np.array([0.0, 0.0, 1.0])

    Nodal_vec = np.cross(k_dir, h_vec)

    Nodal = np.linalg.norm(Nodal_vec)

    # Longitude of the ascending node
    if Nodal_vec[1] >= 0.0:
        omega = np.arccos(Nodal_vec[0]/Nodal)
    else:
        omega = pi2 - np.arccos(Nodal_vec[0]/Nodal)
    omega = omega%pi2
    if omega < 0:
        omega = omega + pi2
#     print(f"\u03A9 = {np.rad2deg(omega)} deg")

    # Argument of the perigee
    if e_vec[2] >= 0.0:
        w = np.arccos(np.dot(Nodal_vec,e_vec)/(Nodal*e))
    else:
        w = pi2 - np.arccos(np.dot(Nodal_vec,e_vec)/(Nodal*e))
    w = w%pi2
    if w < 0:
        w = w + pi2
#     print(f"\u03C9 = {np.rad2deg(w)} deg") 
    
    # True anomaly
    if np.dot(r_vec,v_vec) >= 0.0:
        theta = np.arccos(np.dot(e_vec,r_vec)/(e*r))
    else:
        theta = pi2 - np.arccos(np.dot(e_vec,r_vec)/(e*r))
    theta = theta%pi2
    if theta < 0:
        theta = theta + pi2
#     print(f"\u03BD = {np.rad2deg(theta)} deg")

    # Eccentric anomaly (Curtis, 2020, pp. 146)
    p1  = np.sqrt((1 - e)/(1 + e))
    p2 = np.tan(theta/2.0)
    Ea = 2.0*np.arctan(p1*p2)
    Ea = Ea%pi2
    if Ea < 0:
        Ea = Ea + pi2

    # Mean anomaly (Kepler's equation)
    Me = Ea - e*np.sin(Ea)
    Me = Me%pi2
    if Me < 0:
        Me = Me + pi2

    #Period calculation
    per = pi2 * np.sqrt(a**3 / mu)

    # Return the orbital elements in a list
    oe = [a, e, i, w, omega, Me, per]
    
    return oe
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Function to convert orbital elements to the state vector

def orb2xyz(mu, oe):
    import math

    p = oe[0]*(1 - oe[1]**2)
    
    r = p/(1 + oe[1]*np.cos(oe[5]))
    # print(f"r = {r} km")
    
    # The orbit lies in a plane, thus we have a 3-dimension vector with the 3rd component equal to zero
    rf_vec = np.array([r*np.cos(oe[5]),r*np.sin(oe[5]),0])
    vf_vec = np.array([-np.sqrt(mu/p)*np.sin(oe[5]), np.sqrt(mu/p)*(oe[1] + np.cos(oe[5])), 0])
    # print(f"rf_vec = {rf_vec[0]}, {rf_vec[1]}, {rf_vec[2]}")
    # print(f"vf_vec = {vf_vec[0]}, {vf_vec[1]}, {vf_vec[2]}")
    
    # Rotation matrix
    R_11 = np.cos(oe[4])*np.cos(oe[3]) - np.sin(oe[4])*np.sin(oe[3])*np.cos(oe[2])
    R_12 =-np.cos(oe[4])*np.sin(oe[3]) - np.sin(oe[4])*np.cos(oe[3])*np.cos(oe[2])
    R_13 = np.sin(oe[4])*np.sin(oe[2])

    R_21 = np.sin(oe[4])*np.cos(oe[3]) + np.cos(oe[4])*np.sin(oe[3])*np.cos(oe[2])
    R_22 =-np.sin(oe[4])*np.sin(oe[3]) + np.cos(oe[4])*np.cos(oe[3])*np.cos(oe[2])
    R_23 =-np.cos(oe[4])*np.sin(oe[2])

    R_31 = np.sin(oe[3])*np.sin(oe[2])
    R_32 = np.cos(oe[3])*np.sin(oe[2])
    R_33 = np.cos(oe[2])

    # A matrix is a 3-dimensional array:
    rot_matrix = np.array([[R_11, R_12, R_13],[R_21, R_22, R_23],[R_31, R_32, R_33]])
    # print(f"rot_matrix = {rot_matrix}")
    
    r_vec_inertial = np.matmul(rot_matrix, rf_vec)
    # print(f"r_vec_inertial = {r_vec_inertial} (km)")
    
    v_vec_inertial = np.matmul(rot_matrix, vf_vec)
    # print(f"v_vec_inertial = {v_vec_inertial} (km/s)")
    
    # print(f"Check r_inertial (not required): r_inertial = {np.linalg.norm(r_vec_inertial)} km. It must have the same value of r.")
    
    state_vec = np.concatenate((r_vec_inertial, v_vec_inertial))
    
    return state_vec

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Tranformation from the Ecliptic plane to Planet Equatorial plane

def ecl2equ(eps, xecl):
    xequ = np.zeros(6)

    # Position
    xequ[0] = xecl[0]
    xequ[1] = xecl[1]*np.cos(eps) - xecl[2]*np.sin(eps)
    xequ[2] = xecl[1]*np.sin(eps) + xecl[2]*np.cos(eps)

    # Velocity
    xequ[3] = xecl[3]
    xequ[4] = xecl[4]*np.cos(eps) - xecl[5]*np.sin(eps)
    xequ[5] = xecl[4]*np.sin(eps) + xecl[5]*np.cos(eps)

    return xequ
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Newton's method to solve Kepler's equation
def newton_kepler(Me, e):

    pi = np.pi
    pi2 = 2*pi
    nmax = 1000 # maximum number of iterations

    # Initial guess for the eccentric anomaly
    if Me < pi:
        E_0 = Me + e/2
    elif Me > pi:
        E_0 = Me - e/2
    else:
        E_0 = Me

    for j in range(0, nmax):
        ratio = (E_0 - e*np.sin(E_0) - Me)/(1 - e*np.cos(E_0))

        print(f"iteration no. {j}, |ratio| = {np.abs(ratio)}")

        if np.abs(ratio) <= 1.0e-6:
            break

        E = E_0 - ratio

        E_0 = E #update the value of E_0
        
    return E_0
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Rotation matrix: perifocal frame to geocentric equatorial frame
def RxX (oe):
    # Rotation matrix
    R_11 = np.cos(oe[4])*np.cos(oe[3]) - np.sin(oe[4])*np.sin(oe[3])*np.cos(oe[2])
    R_12 =-np.cos(oe[4])*np.sin(oe[3]) - np.sin(oe[4])*np.cos(oe[3])*np.cos(oe[2])
    R_13 = np.sin(oe[4])*np.sin(oe[2])

    R_21 = np.sin(oe[4])*np.cos(oe[3]) + np.cos(oe[4])*np.sin(oe[3])*np.cos(oe[2])
    R_22 =-np.sin(oe[4])*np.sin(oe[3]) + np.cos(oe[4])*np.cos(oe[3])*np.cos(oe[2])
    R_23 =-np.cos(oe[4])*np.sin(oe[2])

    R_31 = np.sin(oe[3])*np.sin(oe[2])
    R_32 = np.cos(oe[3])*np.sin(oe[2])
    R_33 = np.cos(oe[2])

    return np.array([[R_11, R_12, R_13],[R_21, R_22, R_23],[R_31, R_32, R_33]])
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Function to calculate the Julian Date (JD) from the Gregorian calendar
def julian_date(year, month, day, hour, minute, second):
    # Julian Date (JD) at 0 hr UT
    JD = 367*year - int(7*(year + int((month + 9)/12))/4) + int(275*month/9) + day + 1721013.5

    # Fraction of the day
    frac_day = (second/60.0 + minute)/60.0 + hour
    JD = JD + frac_day

    return JD
#-------------------------------------------------------------------------------
# Function to calculate the initial Greenwich Sidereal Time (GST) at 0h UT
# Reference: Curtis, H. D. (2020). Orbital Mechanics for Engineering Students. Elsevier.
def gst0(epoch):
    # Julian centuries since J2000
    T0 = (epoch - 2451545.0)/36525.0

    # Greenwich Sidereal Time (GST) at epoch
    gst = 100.4606184 + 36000.77004*T0 + 0.000387933*T0**2 - T0**3/38710000.0

    gst = gst%360.0

    return gst
#-------------------------------------------------------------------------------