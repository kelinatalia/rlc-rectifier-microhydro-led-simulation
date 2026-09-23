# RLC Rectifier for a Micro Hydro Generator Powered LED Circuit

## Overview
This project designs and simulates an AC to DC rectifier circuit that powers 3 parallel green LEDs from a small micro hydro generator (5V peak, 20Hz). The circuit is simulated numerically in Python using the 4th order Runge-Kutta method (RK4), comparing 3 different filter configurations to find the best design.

## Steps
- Modeled the circuit: AC source, full wave bridge rectifier, LC filter, and 3 parallel LED branches with current limiting resistors
- Derived the governing equations (KVL and KCL) for the LC filter as coupled ODEs
- Solved the system numerically using RK4 with a small time step over 15 signal periods
- Compared 3 filter configurations: no filter, capacitor only, and LC filter
- Tested different current limiting resistor values to find the best trade-off between LED brightness and safety
- Plotted the AC source, rectified voltage, output voltage, LED current, and inductor current for each configuration

## Result
The LC filter (400 mH inductor, 22 uF capacitor) gives the best performance. It reduces output ripple by 78.9 percent compared to no filter, and by 54 percent compared to capacitor only. The peak LED current stays at 7.58 mA, well below the 20 mA safety limit. A 100 ohm current limiting resistor was chosen as the best balance between LED brightness and safety margin.

## Tech Stack
Python, NumPy, Matplotlib
