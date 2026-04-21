# -*- coding: utf-8 -*-
"""
Created on Wed May 14 14:13:19 2025

@author: arsalann
"""

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

def PowerFlow_Convex(ParkingDemand, Pattern, Price, SampPerH, Vmin):

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
    PenF = 1000
    parking_to_node = {
        1: 28,
        2: 17,
        3: 15
    }
    model=pyo.ConcreteModel()
    
    #model.BRANCH  = no_branches # We have |N| branches in radial system
    #model.BUS     = no_buses # We have |N|+1 nodes in radial system but we dismiss root node
    model.HORIZON = SampPerH*24  
    
    model.L     =  pyo.Set(initialize= data['L'])
    model.LINES = pyo.Set(initialize= data['lineCon']) # Lines Set
    model.N = pyo.Set(initialize= data['N']) # Bus Set
    model.T = pyo.Set(initialize=[x+1 for x in range(model.HORIZON)])
    
    
    model.P = pyo.Var(model.L, model.T) # Branch/Line Active power
    model.Q = pyo.Var(model.L, model.T) # Branch/Line Reactive power
    
    def Limit_Volt(model,n,t):
        return (0.6 ,1.05)
    model.V = pyo.Var(model.N, model.T, bounds=Limit_Volt )
    model.PS = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.QS = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.Isqr = pyo.Var(model.L, model.T, within = pyo.NonNegativeReals)
    model.Dev = pyo.Var(model.N, model.T, within= pyo.NonNegativeReals)

    # Network Characteristics
    pu = 10
    volt = 12.66
    #bus_branch = create_line_tuple(str_data)
    
    # Defining model az concrete one
    
    line_buses = {l: (from_bus, to_bus) for (l, from_bus, to_bus) in data['lineCon']}
    
    #ZL = np.zeros(len(model.L))
    #for idx, l in enumerate(model.L):  # Use enumerate to get integer indices
    #    ZL[idx] = data['R'][l]**2 + data['X'][l]**2  # Squaring R and X
    
    ZL = {l: data['R'][l]**2 + data['X'][l]**2 for l in model.L}
    
    # Power Network characteristics
    
    # Defining model az concrete one
    model.Activedemand = pyo.Param(model.N, model.T, initialize=0, mutable=True)
    model.Reactivedemand = pyo.Param(model.N, model.T, initialize=0, mutable=True)
    
    
    
    
    
    for node in model.N:
        for t in model.T:
            model.Reactivedemand[node, t] = data['bus_Qd'][node] * Pattern.iloc[t-1]# Power Network characteristics
            model.Activedemand[node, t] = data['bus_Pd'][node] * Pattern.iloc[t-1]# Power Network characteristics
    
    
    for t in model.T:
        for p, n in parking_to_node.items():
            # Add parking demand to node demand
            model.Activedemand[n, t] += ParkingDemand[p, t]
    
    
    
    
    
    # Equation 6
    def lines_active_power_rule(model, n, t):
        # First Sum: if there is line which connect bus n to bus m "Loads from BUS n to BUS m"
        # Second Sum: if there is line from any bus m to bus n,
        # Active load as input to node n minus sum of the output load from node n and resistance of the line must be zero
        return sum(model.P[l,t]+ (data['R'][l]/zb)*model.Isqr[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES) - sum(model.P[l,t]  for l in model.L for m in model.N if (l,n,m) in model.LINES) + 0.001*model.Activedemand[n, t]/sb - model.PS[n,t] == 0
        #return sum(model.P[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES) - sum(model.P[l,t] + str_data["R"][l-1]*model.I2[l,t] for l in model.L for m in model.N if (l,n,m) in model.LINES) == 0 #- load_data['P'][n]
    model.lines_active_power_con = pyo.Constraint(model.N, model.T, rule= lines_active_power_rule)
    
    # Equation 7
    def lines_reactive_power_rule(model, n , t):
        return sum(model.Q[l,t]+ (data['X'][l]/zb)*model.Isqr[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES) - sum(model.Q[l,t]  for l in model.L for m in model.N if (l,n,m) in model.LINES) + 0.001*model.Reactivedemand[n, t]/sb - model.QS[n,t] == 0
        #return sum(model.Q[l,t] for l in model.L for m in model.N if (l,n,m) in model.LINES) - sum(model.Q[l,t] +  str_data['X'][l-1]*model.I2[l, t] for l in model.L for m in model.N if (l,n,m) in model.LINES)  == 0 #- load_data['Q'][n]
    model.lines_reactive_power_con = pyo.Constraint(model.N,  model.T, rule=lines_reactive_power_rule)
    
    
    # # Equation 8
    def kirchhoff_voltage_rule(model, l, t):
         return sum(model.V[n,t] + 2*((data['R'][l]/zb)*model.P[l,t] + (data['X'][l]/zb)*model.Q[l,t]) + (ZL[l]/zb)*model.Isqr[l,t] - model.V[m,t] for n in model.N for m in model.N if (l,n,m) in model.LINES) == 0
    model.kirchhoff_voltage_con = pyo.Constraint(model.L, model.T, rule=kirchhoff_voltage_rule) 
    
    
    
    
    
    def ENL(model, l, t):
         return sum(model.V[m,t]*model.Isqr[l,t] for n in model.N for m in model.N if (l,m,n) in model.LINES)  >= model.P[l,t]**2 + model.Q[l,t]**2
    model.ENL_con = pyo.Constraint(model.L, model.T, rule=ENL) 
    
    
    
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
    
    def Voltage_Pen1(model, n, t):
        return model.V[n, t] >= Vmin - model.Dev[n,t]
    model.conV_dev1 = pyo.Constraint(model.N, model.T, rule = Voltage_Pen1)
    
    
    def Voltage_Pen2(model, n, t):
        return model.Dev[n,t] <= Vmin
    model.conV_dev2 = pyo.Constraint(model.N, model.T, rule = Voltage_Pen2)

#    model.con = pyo.Constraint(expr=model.V[0, 1] == 1.05)
    
    model.obj=pyo.Objective(expr=sum((data['R'][l]/zb)*model.Isqr[l,t] for t in model.T for l in model.L) + sum(model.Dev[n,t]*PenF for n in model.N for t in model.T) , sense=pyo.minimize)
    
    """
    solve the model
    """    
    
    start_time= time.time()
    
    scenario_model="ChanceConstraint"
    SOLVER_NAME="gurobi"
    TIME_LIMIT=100000
    
    
    solver=pyo.SolverFactory(SOLVER_NAME)
    
    solver.options['NonConvex'] = 2  # Critical for non-convex problems
    
    
    if SOLVER_NAME == 'cplex':
        solver.options['timelimit'] = TIME_LIMIT
    elif SOLVER_NAME == 'glpk':         
        solver.options['tmlim'] = TIME_LIMIT
    elif SOLVER_NAME == 'gurobi':           
        solver.options['TimeLimit'] = TIME_LIMIT
    
    # solver.options['NonConvex'] = 2
    model.dual = Suffix(direction=Suffix.IMPORT)
    results = solver.solve(model)
    
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
    min_volt_per_node = {
        l: min(pyo.value(model.V[l, t]) for t in model.T)
        for l in model.L
    }
    
    
    nodes = list(min_volt_per_node.keys())
    min_voltages = list(min_volt_per_node.values())
    
#    volt_per_node = {
#         l: pyo.value(model.V[l, t]) for l in model.L for t in model.T
#    }
    
    volt_per_node = [pyo.value(model.V[l, t]) for l in model.L for t in model.T]
    
    volt_per_node = np.zeros((len(model.L), len(model.T)))

# Fill with Pyomo values
    for i, l in enumerate(model.L):
        for j, t in enumerate(model.T):
            volt_per_node[i, j] = pyo.value(model.V[l, t])
    
    
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(nodes, min_voltages, color='red')
    plt.xlabel('Node (l)')
    plt.ylabel('Minimum Voltage')
    plt.title('Minimum Voltage per Node over Time Horizon')
    plt.xticks(rotation=90)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    
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
        
    plt.figure(2)    
    
    plt.plot(volt)
    plt.title('Voltage over node')
    plt.xlabel('Node')
    plt.ylabel('Voltage (pu)')
    plt.grid(True)
    plt.show()
    
    return min_voltages, volt_per_node, duals, duals2
   
    print(volt_per_node)    