# In this file, we import the initial state vector for the major bodies in
# the system (Sun-Earth-Moon) from the JPL Horizons database.

def vector(body, frame, start_date):
# Input:
#     body - the body's name e.g. '301' (The Moon)
#     frame - the reference - 'ecliptic' or 'earth' (equatorial)
#     start_date - first date interval 'yyy-mm-dd'

  from astroquery.jplhorizons import Horizons
  from datetime import datetime, timedelta


  # Adding one day to the start date
  date_1 = datetime.strptime(start_date, '%Y-%m-%d')
  date_2 = date_1 + timedelta(days=1)
  end_date = str(date_2)[0:10]

  # Query the JPL Horizons database
  obj = Horizons(id=body, location='500@399',
                 epochs={'start':start_date, 'stop':end_date,
                         'step':'10d'})
  state_vec_mb = obj.vectors(refplane=frame)
  return state_vec_mb

def orbital_elements (body, frame, start_date):
# Input:
#     body - the body's name e.g. '301' (The Moon)
#     frame - the reference - 'ecliptic' or 'earth' (equatorial)
#     start_date - first date interval 'yyy-mm-dd'

  from astroquery.jplhorizons import Horizons
  from datetime import datetime, timedelta

  # Adding one day to the start date
  date_1 = datetime.strptime(start_date, '%Y-%m-%d')
  date_2 = date_1 + timedelta(days=1)
  end_date = str(date_2)[0:10]

  # Query the JPL Horizons database
  obj = Horizons(id=body, location='500@399',
                 epochs={'start':start_date, 'stop':end_date,
                         'step':'10d'})
  oe_mb = obj.elements(refplane=frame)
  return oe_mb

# Function to load the major bodies' initial conditions
# Input: frame - the reference frame - 'ecliptic' or 'earth'
#        start_date - the first date interval 'yyyy-mm-dd'
def load_mb (frame, start_date):
  import numpy as np
  from major_bodies_parameters import constants

  const = constants()

  au = const.au # km
  day = const.day # seconds

  print (f"\nEpoch: {start_date}")
  print (f"Reference frame: {frame.capitalize()} mean equator and equinox of J2000.0")

  # Load the major bodies' initial conditions
  vec_sun = vector('10', frame, start_date)
  vec_earth = vector('399', frame, start_date)
  vec_moon = vector('301', frame, start_date)

  jd = vec_sun['datetime_jd'][0]

  # Transfer the initial conditions to the state vector
  Xb = np.zeros((3, 6))
  
  # Sun
  Xb[0, 0] = vec_sun['x'][0]*au
  Xb[0, 1] = vec_sun['y'][0]*au
  Xb[0, 2] = vec_sun['z'][0]*au
  Xb[0, 3] = vec_sun['vx'][0]*au/day
  Xb[0, 4] = vec_sun['vy'][0]*au/day
  Xb[0, 5] = vec_sun['vz'][0]*au/day

  # Earth
  Xb[1, 0] = vec_earth['x'][0]*au
  Xb[1, 1] = vec_earth['y'][0]*au
  Xb[1, 2] = vec_earth['z'][0]*au
  Xb[1, 3] = vec_earth['vx'][0]*au/day
  Xb[1, 4] = vec_earth['vy'][0]*au/day
  Xb[1, 5] = vec_earth['vz'][0]*au/day

  # Moon
  Xb[2, 0] = vec_moon['x'][0]*au
  Xb[2, 1] = vec_moon['y'][0]*au
  Xb[2, 2] = vec_moon['z'][0]*au
  Xb[2, 3] = vec_moon['vx'][0]*au/day
  Xb[2, 4] = vec_moon['vy'][0]*au/day
  Xb[2, 5] = vec_moon['vz'][0]*au/day


  return Xb, jd
########################################################################
# Test the function
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import numpy as np
  from functions import xyz2orb
  import os

  au = 149597870.7 # km
  day = 86400 # seconds

  # Physical parameters of the major bodies
  Msun = 1.989e30 # kg
  Mearth = 5.972e24 # kg
  Mmoon = 7.34767309e22 # kg
  G = 6.67430e-20 # km^3/kg/s^2

  GMsun = G*Msun
  GMearth = G*Mearth
  GMmoon = G*Mmoon

  # Initial date (epoch - YYYY-MM-DD)
  epoch = '2025-01-01'

  # Earth
  body = '399'
  vec_earth = vector(body, 'earth', epoch)

  # Table header
  print (f"\nEpoch: {epoch}")
  print (vec_earth['datetime_jd'][0])

  print ("\n")
  print ("Earth (Central body)")
  X_earth = np.zeros(6)
  X_earth[0] = vec_earth ['x'][0]*au
  X_earth[1] = vec_earth ['y'][0]*au
  X_earth[2] = vec_earth ['z'][0]*au
  print (f"R_earth = {np.linalg.norm(X_earth[0:2])} km")
  X_earth[3] = vec_earth ['vx'][0]*au/day
  X_earth[4] = vec_earth ['vy'][0]*au/day
  X_earth[5] = vec_earth ['vz'][0]*au/day
  print (f"V_earth = {np.linalg.norm(X_earth[3:5])} km/s")
  print (f"X_earth = {X_earth}")

  # Moon
  body = '301'
  vec_moon = vector(body, 'earth', epoch)

  print ("\nMoon (State vector from JPL Horizons)")
  X_moon = np.zeros(6)
  X_moon[0] = vec_moon ['x'][0]*au
  X_moon[1] = vec_moon ['y'][0]*au
  X_moon[2] = vec_moon ['z'][0]*au
  print (f"R_moon = {np.linalg.norm(X_moon[0:2])} km")
  X_moon[3] = vec_moon ['vx'][0]*au/day
  X_moon[4] = vec_moon ['vy'][0]*au/day
  X_moon[5] = vec_moon ['vz'][0]*au/day
  print (f"V_moon = {np.linalg.norm(X_moon[3:5])} km/s")
  print (f"X_moon = {X_moon}")

  oe_moon  = np.zeros(6)
  mu = GMearth + GMmoon
  oe_moon = xyz2orb(mu, X_moon[0:3], X_moon[3:6])
  print ("\nMoon (Orbital elements from state vector)")
  print (f"a = {oe_moon[0]} km")
  print (f"e = {oe_moon[1]}")
  print (f"i = {np.rad2deg(oe_moon[2])} deg")
  print (f"w = {np.rad2deg(oe_moon[3])} deg")
  print (f"Omega = {np.rad2deg(oe_moon[4])} deg")
  print (f"Me = {np.rad2deg(oe_moon[5])} deg")

  jpl_oe_moon = orbital_elements(body, 'earth', epoch)
  print ("\nMoon from JPL Horizons")
  print (f"a = {jpl_oe_moon['a'][0]*au} km")
  print (f"e = {jpl_oe_moon['e'][0]}")
  print (f"i = {jpl_oe_moon['incl'][0]} deg")
  print (f"w = {jpl_oe_moon['w'][0]} deg")
  print (f"Omega = {jpl_oe_moon['Omega'][0]} deg")
  print (f"Me = {jpl_oe_moon['M'][0]} deg")

  # Sun
  print ("\nSun (State vector from JPL Horizons)")
  body = '10'
  vec_sun = vector(body, 'earth', epoch)

  # print (f"HERE: {jd}")

  X_sun = np.zeros(6)
  X_sun[0] = vec_sun ['x'][0]*au
  X_sun[1] = vec_sun ['y'][0]*au
  X_sun[2] = vec_sun ['z'][0]*au
  print (f"R_sun = {np.linalg.norm(X_sun[0:2])} km")
  X_sun[3] = vec_sun ['vx'][0]*au/day
  X_sun[4] = vec_sun ['vy'][0]*au/day
  X_sun[5] = vec_sun ['vz'][0]*au/day
  print (f"V_sun = {np.linalg.norm(X_sun[3:5])} km/s")
  print (f"X_sun = {X_sun}")

  oe_sun  = np.zeros(6)
  mu = GMsun + GMearth
  oe_sun = xyz2orb(mu, X_sun[0:3], X_sun[3:6])
  print ("\nSun (Orbital elements from state vector)")
  print (f"a = {oe_sun[0]} km")
  print (f"e = {oe_sun[1]}")
  print (f"i = {np.rad2deg(oe_sun[2])} deg")
  print (f"w = {np.rad2deg(oe_sun[3])} deg")
  print (f"Omega = {np.rad2deg(oe_sun[4])} deg")
  print (f"Me = {np.rad2deg(oe_sun[5])} deg")

  # Test the function load_mb
  Xb = load_mb('earth', epoch)
  print ("\nMajor bodies' initial conditions")
  # Sun
  print ("\nSun")
  print (f"Xb_sun = {Xb[0,:]}")
  # Earth
  print ("\nEarth")
  print (f"Xb_earth = {Xb[1,:]}")
  # Moon
  print ("\nMoon")
  print (f"Xb_moon = {Xb[2,:]}")





