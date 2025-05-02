Version 04.30.2025

This study associated with this code proposes a dynamical model for the trajectory of a non-functional satellite with an inflatable spherical solar sail that acts as a de-orbiting device by harnessing Solar Radiation Pressure (SRP) and varying area-to-mass ratios. The model mimics real-world behaviors by using equations of motion that feature natural perturbations and SRP calculations using the cannonball model and shadow function. The time needed for the proposed balloon to push medium Earth orbit space debris back into the atmosphere can be calculated for various binary and linear inflation methods. The results indicate that for satellites in near-circular orbits with an initial semi-major axis of 22,000 km, the solar sail can be deployed to push them into atmospheric re-entry within about 34.96 or 50.14 years for initial orbital inclinations of 0.001 or 56.06 degrees, respectively. For a GPS satellite with a semi-major axis of 26,560 km, eccentricity of 0.001, and inclination of 55 degrees, de-orbiting occurs within 61.3602 years. There is a potential for the improvement of these times by using a larger balloon or more reflective material.

Orbit_simulator.py
- This file is the orbit simulator interface, where initial conditions (for the orbit and the satellite) and model outputs (such as plots and saved results) can be altered. Methods of inflation and deflation for the spherical balloon can also be changed in this file. Various perturbations can be turned on and off.

control.py
- This file contains small functions to control the code workflow.

EGM2008_upto50_TideFree.in
- This file stores Earth’s spherical harmonic coefficients (full and normalized) up to degree and order 50.

functions.py
- This file contains functions for:
	- Converting from the ecliptic to the Earth’s equatorial plane
	- Solving Kepler’s equation using Newton’s Method
	- Converting from the perifocal frame to the geocentric equatorial frame
	- Calculating the Julian Date from the Gregorian calendar
	- Calculating the initial Greenwich Sidereal Time (GST) at 0h UT

major_bodies_parameters.py
- This file contains parameters and constants used in the simulation for the Sun, Earth, and Moon.

major_bodies.py
- This file imports the initial state vector for the major bodies in the system (Sun-Earth-Moon) from the JPL Horizons database.

perturbation.py
- This file contains the functions pertaining to the perturbations that can act on a satellite (or spacecraft) orbiting the Earth:
	- Third-body perturbations
	- Solar Radiation Pressure (SRP) and resulting acceleration (including the shadow function)
	- Gravitational potential of Mercury expanded in spherical harmonics
	- Vector acceleration due to the gravitational field of the Earth from Earth's spherical harmonics EGM2008
	- Perturbations solely due to the Earth’s oblateness
	- Perturbations due to atmospheric drag

sandbox.py
- A testing file.

USSA76.py
- This file contains a function to calculate the atmospheric density.
