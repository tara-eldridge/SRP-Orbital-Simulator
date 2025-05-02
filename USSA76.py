# Function to calculate the atmospheric density
# Model USSA1976
# MATLAB original code by: Howard D. Curtis, Orbital Mechanics for Engineering Students, 4th Revised Edition, Elsevier, 2020
# Translated to Python by: Diogo Merguizo Sanchez, The University of Oklahoma, 2024
#
# Inputs:
#        z: altitude[km]
# Outputs:
#        rho: atmospheric density[kg/m^3]

import numpy as np
def rho_atm (z):

  # Geometric altitudes (km)
  h = np.array([0, 25, 30, 40, 50, 60, 70, 80, 90, 100,
       110, 120, 130, 140, 150, 180, 200, 250,
       300, 350, 400, 450, 500, 600, 700, 800, 900, 1000])
  
  # Corresponding densities (kg/m^3) from USSSA76
  r = np.array([1.225, 4.008e-2, 1.841e-2, 3.996e-3, 1.027e-3, 3.097e-4, 8.283e-5,
                1.846e-5, 3.416e-6, 5.606e-7, 9.708e-8, 2.222e-8, 8.152e-9, 3.831e-9,
                2.076e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11, 7.014e-12, 2.803e-12,
                1.184e-12, 5.215e-13, 1.137e-13, 3.070e-14, 1.136e-14, 5.759e-15, 3.561e-15])
  
  # Scale heights (km)
  H = np.array([7.310, 6.427, 6.546, 7.360, 8.342, 7.583, 6.661,
                5.927, 5.533, 5.703, 6.782, 9.973, 13.243, 16.322,
                21.652, 27.974, 34.934, 43.342, 49.755, 54.513, 58.019,
                60.980, 65.654, 76.377, 100.587, 147.203, 208.020])
  
  # Handle the altitudes outsinde of the range
  if z > 1000.0:
    z = 1000.0
  elif z < 0.0:
    z = 0.0
  
  # Determine the interpolation interval
  for j in range(0, 27):
    if z >= h[j] and z <= h[j+1]:
      i = j
      break
  
  if z == 1000.0:
    i = 26

  # Exponential interpolation of the density
  rho = r[i]*np.exp(-(z - h[i])/H[i])

  return rho

# Test the function
if __name__ == '__main__':
  import matplotlib.pyplot as plt

  # Altitude range
  z = range(0, 101, 1)

  # Calculate the atmospheric density
  rho = np.zeros(len(z))
  for i in range(0, len(z)):
    rho[i] = rho_atm(z[i])
    print(f"z = {z[i]} km, rho = {rho[i]} kg/m^3")

  # Plot the atmospheric density
  plt.figure()
  plt.plot(z, rho, 'b')
  plt.xlabel('Altitude [km]')
  plt.ylabel('Atmospheric density [kg/m^3]')
  plt.title('Atmospheric density - USSA1976')
  plt.grid()
  plt.show()

