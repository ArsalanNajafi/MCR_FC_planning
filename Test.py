# -*- coding: utf-8 -*-
"""
Created on Fri May  2 10:03:35 2025

@author: arsalann
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:59:40 2022

@author: Alireza
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from pyomo.environ import Suffix

#from utility import create_line_tuple


data = pyo.DataPortal()
data.load(filename='iee33_bus_data.dat')


# current_directory = os.path.dirname(__file__)
#current_directory = os.getcwd()
#parent_directory  = os.path.split(current_directory)[0] # Repeat as needed
#file_path         = os.path.join(current_directory, 'DLF.xls')

# Loading power network data
#str_data   = pd.read_excel(file_path, sheet_name='Strdata')
#load_data  = pd.read_excel(file_path, sheet_name='Loaddata')
#power_data = pd.read_excel(file_path, sheet_name='Sheet1')

# Constant parameters
#no_buses    = len(str_data)
#no_branches = len(str_data)

vb = 12.66
sb = 10
zb = vb**2/sb

model=pyo.ConcreteModel()

#model.BRANCH  = no_branches # We have |N| branches in radial system
#model.BUS     = no_buses # We have |N|+1 nodes in radial system but we dismiss root node
model.HORIZON = 1 

model.L     =  pyo.Set(initialize= data['L'])
model.LINES = pyo.Set(initialize= data['lineCon']) # Lines Set
model.N = pyo.Set(initialize= data['N']) # Bus Set
model.T = pyo.Set(initialize=[x+1 for x in range(model.HORIZON)])


model.P = pyo.Var(model.L, model.T) # Branch/Line Active power
model.Q = pyo.Var(model.L, model.T) # Branch/Line Reactive power

def Limit_Volt(model,n,t):
    return (0.95,1.05)
model.V = pyo.Var(model.N, model.T, bounds=Limit_Volt )
model.PS = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
model.QS = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)

model.abs_dev = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals, 
                       doc="Absolute voltage deviation from 1.0 p.u.")


# Network Characteristics
pu = 10
volt = 12.66
#bus_branch = create_line_tuple(str_data)

# Defining model az concrete one

line_buses = {l: (from_bus, to_bus) for (l, from_bus, to_bus) in data['lineCon']}

# Power Network characteristics

# Equation 6
def lines_active_power_rule(model, n, t):
    # First Sum: if there is line which connect bus n to bus m "Loads from BUS n to BUS m"
    # Second Sum: if there is line from any bus m to bus n,
    # Active load as input to node n minus sum of the output load from node n and resistance of the line must be zero
    return sum(model.P[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES) - sum(model.P[l,t]  for l in model.L for m in model.N if (l,n,m) in model.LINES) + 0.001*data['bus_Pd'][n]/sb - model.PS[n,t] == 0
    #return sum(model.P[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES) - sum(model.P[l,t] + str_data["R"][l-1]*model.I2[l,t] for l in model.L for m in model.N if (l,n,m) in model.LINES) == 0 #- load_data['P'][n]
model.lines_active_power_con = pyo.Constraint(model.N, model.T, rule= lines_active_power_rule)

# Equation 7
def lines_reactive_power_rule(model, n , t):
    return sum(model.Q[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES) - sum(model.Q[l,t]  for l in model.L for m in model.N if (l,n,m) in model.LINES) + 0.001*data['bus_Qd'][n]/sb - model.QS[n,t] == 0
    #return sum(model.Q[l,t] for l in model.L for m in model.N if (l,n,m) in model.LINES) - sum(model.Q[l,t] +  str_data['X'][l-1]*model.I2[l, t] for l in model.L for m in model.N if (l,n,m) in model.LINES)  == 0 #- load_data['Q'][n]
model.lines_reactive_power_con = pyo.Constraint(model.N,  model.T, rule=lines_reactive_power_rule)


# # Equation 8
def kirchhoff_voltage_rule(model, l, t):
     return sum(-model.V[n,t] - ((data['R'][l]/zb)*model.P[l,t] + (data['X'][l]/zb)*model.Q[l,t]) + model.V[m,t] for n in model.N for m in model.N if (l,n,m) in model.LINES) == 0
model.kirchhoff_voltage_con = pyo.Constraint(model.L, model.T, rule=kirchhoff_voltage_rule) 



def abs_dev_upper(model, n, t):
    """Absolute deviation: |V[n,t] - 1.0| <= abs_dev[n,t]"""
    return model.abs_dev[n,t] >= model.V[n,t] - 1.0

def abs_dev_lower(model, n, t):
    return model.abs_dev[n,t] >= -(model.V[n,t] - 1.0)
model.con_abs_upper = pyo.Constraint(model.N, model.T, rule=abs_dev_upper)
model.con_abs_lower = pyo.Constraint(model.N, model.T, rule=abs_dev_lower)

# generatos active power rule
def gen_active_power_rule(model, n, t):
    if n == 0:
        return pyo.Constraint.Skip
    return model.PS[n,t] == 0
model.gen_active_power_con = pyo.Constraint(model.N,  model.T, rule=gen_active_power_rule)


# generatos reactive power rule
def gen_reactive_power_rule(model, n, t):
    if n == 0:
        return pyo.Constraint.Skip
    return model.QS[n,t] == 0
model.gen_reactive_power_con = pyo.Constraint(model.N,  model.T, rule=gen_reactive_power_rule)

##def con_rule(model, n, t):
##    if n == 0:  # Apply constraint only to node 'n1'
##        return model.V[n, t] == 1.05
##    else:
##        return pyo.Constraint.Skip  # Skip for other nodes

##model.con = pyo.Constraint(model.N, model.T, rule=con_rule)

# For a specific node 'n1' and time t=1 (example)
model.con = pyo.Constraint(expr=model.V[0, 1] == 1.05)

#model.obj=pyo.Objective(expr=sum(model.P[l,t] for t in model.T for l in model.L), sense=pyo.minimize)

model.obj = pyo.Objective(
    expr=sum(model.abs_dev[n,t] for n in model.N for t in model.T),
    sense=pyo.minimize
)

"""
solve the model
"""    

start_time= time.time()

scenario_model="ChanceConstraint"
SOLVER_NAME="gurobi"
TIME_LIMIT=100000

solver=pyo.SolverFactory(SOLVER_NAME)

if SOLVER_NAME == 'cplex':
    solver.options['timelimit'] = TIME_LIMIT
elif SOLVER_NAME == 'glpk':         
    solver.options['tmlim'] = TIME_LIMIT
elif SOLVER_NAME == 'gurobi':           
    solver.options['TimeLimit'] = TIME_LIMIT

# solver.options['NonConvex'] = 2

model.dual = Suffix(direction=Suffix.IMPORT)
results = solver.solve(model)


with open('Power_Network_Constraints.txt', 'w') as f:
    f.write("Description of the power network constraints:\n")
    model.pprint(ostream=f)
    
    


print("Voltage in each branch:")
for l in model.L:
    print(pyo.value(model.V[l,1]))


print("Voltage in each branch:")
for n in model.N:
    print(pyo.value(model.PS[n,1]))

print("Voltage in each branch:")
for n in model.N:
    print(pyo.value(model.QS[n,1]))



volt = [pyo.value(model.V[l,1]) for l in model.L]
PS  = [pyo.value(model.PS[n,1]) for n in model.N]
QS  = [pyo.value(model.QS[n,1]) for n in model.N]
ActivePower = [pyo.value(model.P[l,1]) for l in model.L]
ReactivePower = [pyo.value(model.Q[l,1]) for l in model.L]

df = pd.DataFrame({"V":volt})

with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer,  sheet_name='Voltage')
    
df2 = pd.DataFrame({"P_Active":PS,
                   "Q_Reactive":QS})    
with pd.ExcelWriter('output2.xlsx') as writer:
    df2.to_excel(writer,  sheet_name='Optimal_Power')    
    
plt.figure(1)    

plt.plot(volt)
plt.title('Voltage over node')
plt.xlabel('Node')
plt.ylabel('Voltage (pu)')
plt.grid(True)
plt.show()


line_data = data['lineData']

def find_path(s, b, line_data):
    """Finds path from bus s to bus b in a radial network (returns line indices)."""
    path = []
    current = b
    while current != s:
        found = False
        for k, (from_bus, to_bus, r, x) in enumerate(line_data):
            if to_bus == current:
                path.append(k)
                current = from_bus
                found = True
                break
        if not found:
            raise ValueError(f"No path from bus {s} to bus {b}")
    return path

def compute_voltage_sensitivity(line_data, num_buses=33):
    """Computes voltage sensitivity matrix for a radial network."""
    alpha = np.zeros((num_buses + 1, num_buses + 1))  # +1 for 1-based indexing
    
    for b in range(1, num_buses + 1):
        for s in range(1, num_buses + 1):
            if s == b:
                alpha[b, s] = 0  # No self-sensitivity
                continue
                
            try:
                path_indices = find_path(s, b, line_data)
                total_r = sum(line_data[k][2] for k in path_indices)
                alpha[b, s] = -total_r  # Negative sign: voltage drops with increased load
            except ValueError:
                alpha[b, s] = 0  # No path exists
    
    return alpha

Alpha = compute_voltage_sensitivity(line_data, 33)

Path = find_path(1, 10, line_data)



volt = [pyo.value(model.V[l,t]) for l in model.L for t in model.T]
min_volt_per_node = {
     l: min(pyo.value(model.V[l, t]) for t in model.T)
     for l in model.L
}
    
    
nodes = list(min_volt_per_node.keys())
min_voltages = list(min_volt_per_node.values())
    
volt_per_node = {
     l: pyo.value(model.V[l, t]) for l in model.L for t in model.T
}

duals = {
    (l, t): model.dual[model.kirchhoff_voltage_con[l, t]]
    for l in model.L 
    for t in model.T
    if model.kirchhoff_voltage_con[l, t] in model.dual
}


duals2 = {
    (n, t): sum(
        duals[l, t] 
        for l in model.L 
        if (l, t) in duals and n in line_buses[l]
    )
    for n in model.N
    for t in model.T
}
