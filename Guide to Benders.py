# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 17:22:29 2025

@author: arsalann
"""

import numpy as np

# Example: Compute voltage sensitivity for IEEE 33-bus
def compute_voltage_sensitivity(line_data, num_buses=33):
    alpha = np.zeros((num_buses, num_buses))  # alpha[b, s]
    for b in range(num_buses):
        for s in range(num_buses):
            path = find_path(s, b, line_data)  # Implement path-finding (e.g., Dijkstra)
            alpha[b, s] = -sum(line_data[k][2] for k in path)  # Sum resistances
    return alpha

alpha = compute_voltage_sensitivity(line_data)  # Shape: (33, 33)

#######(2) Add Benders Cuts in Pyomo
 
def voltage_feasibility_cut(model, b):
    if voltage_at_bus[b] < 0.95:  # Check violation from your power flow
        return sum(alpha[b, s] * model.P_btot[s, t] for s in model.S for t in model.T) <= 0.95
    else:
        return pyo.Constraint.Skip

model.voltage_cut = pyo.Constraint(model.Buses, rule=voltage_feasibility_cut)



#(3) Update Master Problem Iteratively
 
while True:
    # Solve master problem
    solver.solve(model)
    P_btot_values = {s: pyo.value(model.P_btot[s, t]) for s in model.S for t in model.T}
    
    # Run custom power flow
    voltage_at_bus = custom_power_flow(P_btot_values)  # Your implementation
    
    # Check violations and add cuts
    if all(v >= 0.95 for v in voltage_at_bus.values()):
        break  # Converged
    else:
        model.voltage_cut.reconstruct()  # Update cuts
