# -*- coding: utf-8 -*-
"""
Created on Tue May 17 16:59:40 2022
@author: Alireza
Revised for SOCP with Line Losses
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from pyomo.environ import Suffix
from GlobalData import GlobalData

[parking_to_node, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

def PowerFlow(ParkingDemand, Pattern, Price):

    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')
    
    current_directory = os.path.dirname(__file__)
    current_directory = os.getcwd()
    file_path = os.path.join(current_directory, 'data.xlsx')
    EVdata = pd.read_excel(file_path, sheet_name='ParkingData_ver2')
    
    vb = 12.66
    sb = 10
    zb = vb**2/sb
    Vmin_sq = Vmin**2  # Squared Minimum Voltage (e.g., 0.9^2 = 0.81)
    Vmax_sq = 1.05**2  # Squared Maximum Voltage (e.g., 1.05^2 = 1.1025)
    PenF = 1
    
    model = pyo.ConcreteModel()
    
    # Sets
    model.HORIZON = SampPerH * 24 
    model.L     = pyo.Set(initialize= data['L'])
    model.LINES = pyo.Set(initialize= data['lineCon']) 
    model.N = pyo.Set(initialize= data['N']) 
    model.T = pyo.Set(initialize=[x+1 for x in range(model.HORIZON)])
    
    # Line Characteristics (Pre-calculate Z^2 for losses)
    line_buses = {l: (from_bus, to_bus) for (l, from_bus, to_bus) in data['lineCon']}
    Z_sq = {l: (data['R'][l]**2 + data['X'][l]**2) for l in model.L}

    # --- VARIABLES (SOCP FORMULATION) ---
    # P, Q: Branch Active/Reactive Power
    model.P = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)
    model.Q = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)
    
    # v_sq: Squared Voltage Magnitude (V^2)
    # l_sq: Squared Current Magnitude (I^2)
    model.v_sq = pyo.Var(model.N, model.T, bounds=(0.5**2, 1.2**2)) 
    model.l_sq = pyo.Var(model.L, model.T, within=pyo.NonNegativeReals)
    
    # PS, QS: Distributed Generation (0 in this case)
    model.PS = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.QS = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    
    # Dev, abs_dev: Penalty variables for optimization
    model.Dev = pyo.Var(model.N, model.T, within= pyo.NonNegativeReals)
    model.abs_dev = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    
    # Parameters
    model.Activedemand = pyo.Param(model.N, model.T, initialize=0, mutable=True)
    model.Reactivedemand = pyo.Param(model.N, model.T, initialize=0, mutable=True)
    
    # --- DATA LOADING ---
    for node in model.N:
        for t in model.T:
            model.Reactivedemand[node, t] = data['bus_Qd'][node] * Pattern.iloc[t-1]
            model.Activedemand[node, t] = data['bus_Pd'][node] * Pattern.iloc[t-1]
    
    # Add Parking Demand
    for t in model.T:
        for p, n in parking_to_node.items():
            model.Activedemand[n, t] += ParkingDemand[p, t]
    
    # --- CONSTRAINTS (SOCP DISTFLOW) ---
    
    # 1. Active Power Balance (INCLUDING LOSSES: R * I^2)
    def lines_active_power_rule(model, n, t):
        inflow = sum(model.P[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES)
        outflow_loss = sum(model.P[l,t] + data['R'][l]*model.l_sq[l,t] 
                           for l in model.L for m in model.N if (l,n,m) in model.LINES)
        # Equation: Sum(In) - Sum(Out + Losses) - Gen - Load = 0
        return inflow - outflow_loss - model.PS[n,t] - 0.001*model.Activedemand[n,t]/sb == 0
    model.lines_active_power_con = pyo.Constraint(model.N, model.T, rule=lines_active_power_rule)
    
    # 2. Reactive Power Balance (INCLUDING LOSSES: X * I^2)
    def lines_reactive_power_rule(model, n , t):
        inflow = sum(model.Q[l,t] for l in model.L for m in model.N if (l,m,n) in model.LINES)
        outflow_loss = sum(model.Q[l,t] + data['X'][l]*model.l_sq[l,t] 
                           for l in model.L for m in model.N if (l,n,m) in model.LINES)
        return inflow - outflow_loss - model.QS[n,t] - 0.001*model.Reactivedemand[n,t]/sb == 0
    model.lines_reactive_power_con = pyo.Constraint(model.N,  model.T, rule=lines_reactive_power_rule)
    
    # 3. Kirchhoff Voltage Law (SOCP Drop)
    # v_sq(from) - v_sq(to) = 2*(R*P + X*Q) + Z^2 * I_sq
    def kirchhoff_voltage_rule(model, l, t):
        from_bus = line_buses[l][0]
        to_bus   = line_buses[l][1]
        return model.v_sq[from_bus, t] - model.v_sq[to_bus, t] \
             == 2 * (data['R'][l]*model.P[l,t] + data['X'][l]*model.Q[l,t]) + Z_sq[l]*model.l_sq[l,t]
    model.kirchhoff_voltage_con = pyo.Constraint(model.L, model.T, rule=kirchhoff_voltage_con) 
    
    # 4. Second-Order Cone Constraint (Rotated Cone)
    # v_sq(to) * I_sq >= P^2 + Q^2
    # This relates Voltage, Current, and Power exactly.
    def SOC_constraint(model, l, t):
        to_bus = line_buses[l][1]
        # Gurobi handles this non-convex quadratic constraint (v*l >= P^2 + Q^2)
        return model.v_sq[to_bus, t] * model.l_sq[l,t] >= model.P[l,t]**2 + model.Q[l,t]**2
    model.SOC_con = pyo.Constraint(model.L, model.T, rule=SOC_constraint)
    
    # 5. Fix Slack Bus (Bus 0) to 1.05 p.u. Squared
    def fix_slack_voltage(model, t):
        return model.v_sq[0, t] == 1.05**2
    model.fix_slack_con = pyo.Constraint(model.T, rule=fix_slack_voltage)
    
    # 6. Generators (Only Slack Bus is source)
    def gen_active_power_rule(model, n, t):
        if n == 0: return pyo.Constraint.Skip
        return model.PS[n,t] == 0
    model.gen_active_power_con = pyo.Constraint(model.N,  model.T, rule=gen_active_power_rule)
    
    def gen_reactive_power_rule(model, n, t):
        if n == 0: return pyo.Constraint.Skip
        return model.QS[n,t] == 0
    model.gen_reactive_power_con = pyo.Constraint(model.N,  model.T, rule=gen_reactive_power_rule)
    
    # 7. Voltage Limits (Applied to Squared Voltage)
    # v_sq >= Vmin^2 - Dev (Soft constraint)
    def Voltage_Pen1(model, n, t):
        return model.v_sq[n, t] >= Vmin_sq - model.Dev[n,t]
    model.conV_dev1 = pyo.Constraint(model.N, model.T, rule = Voltage_Pen1)
    
    def Voltage_Pen2(model, n, t):
        return model.Dev[n,t] <= 0.1 # Max allowed violation in squared voltage
    model.conV_dev2 = pyo.Constraint(model.N, model.T, rule = Voltage_Pen2)
    
    # Objective: Minimize Deviation from 1.0 p.u.
    # We approximate |V - 1| with |V^2 - 1| because V is close to 1.
    # |V^2 - 1| = |(V-1)(V+1)| approx 2|V-1|.
    def abs_dev_upper(model, n, t):
        return model.abs_dev[n,t] >= model.v_sq[n,t] - 1.0
    def abs_dev_lower(model, n, t):
        return model.abs_dev[n,t] >= -(model.v_sq[n,t] - 1.0)
    model.con_abs_upper = pyo.Constraint(model.N, model.T, rule=abs_dev_upper)
    model.con_abs_lower = pyo.Constraint(model.N, model.T, rule=abs_dev_lower)

    model.obj = pyo.Objective(
        expr=sum(model.abs_dev[n,t] for n in model.N for t in model.T) +
        sum(model.Dev[n,t]*PenF for n in model.N for t in model.T),
        sense=pyo.minimize
    )

    # --- SOLVE ---
    start_time = time.time()
    
    SOLVER_NAME="gurobi"
    
    solver = pyo.SolverFactory(SOLVER_NAME)
    
    # IMPORTANT: Set NonConvex=2 for the QCQP (Quadratically Constrained Quadratic Program)
    # The constraint v*l >= P^2 + Q^2 is non-convex.
    solver.options = {
        'NonConvex': 2,
        'NumericFocus': 3,
        'ScaleFlag': 2, 
        'FeasibilityTol': 1e-6,
        'OptimalityTol': 1e-6,
        'TimeLimit': 100000
    }
    
    model.dual = Suffix(direction=Suffix.IMPORT)
    results = solver.solve(model)
    
    # --- EXTRACT RESULTS ---
    if results.solver.status == pyo.SolverStatus.ok and \
       results.solver.termination_condition == pyo.TerminationCondition.optimal:
        
        print("SOCP Solver Successful.")
        
        # Extract Duals
        duals = {
            (l, t): model.dual[model.kirchhoff_voltage_con[l, t]]
            for l in model.L for t in model.T
            if model.kirchhoff_voltage_con[l, t] in model.dual
        }
        
        # Calculate Total Losses for information
        total_losses_p = sum(data['R'][l] * pyo.value(model.l_sq[l,t]) for l in model.L for t in model.T)
        print(f"Total Active Power Losses (p.u.): {total_losses_p:.4f}")
        
        duals_balance_p = {
            (n,t): model.dual[model.lines_active_power_con[n,t]]
            for n in model.N for t in model.T
            if model.lines_active_power_con[n,t] in model.dual
        }
        
        # Prepare Voltage Data (Convert Squared V back to V for plotting)
        min_voltages_sq = {
            l: min(pyo.value(model.v_sq[l, t]) for t in model.T)
            for l in model.L
        }
        
        min_voltages = {k: np.sqrt(v) for k, v in min_voltages_sq.items()}
        nodes = list(min_voltages.keys())
        vals = list(min_voltages.values())
        
        volt_per_node_sq = np.zeros((len(model.L), len(model.T)))
        for i, l in enumerate(model.L):
            for j, t in enumerate(model.T):
                volt_per_node_sq[i, j] = pyo.value(model.v_sq[l, t])
        
        volt_per_node = np.sqrt(volt_per_node_sq) # Take sqrt for final output

        # Plotting
        plt.figure(figsize=(8, 4))
        plt.plot(nodes, vals, color='red', marker='o', linestyle='-')
        plt.axhline(y=Vmin, color='b', linestyle='--', label='Vmin Limit')
        plt.xlabel('Node')
        plt.ylabel('Voltage (p.u.)')
        plt.title('Minimum Voltage per Node (SOCP with Losses)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("MinVoltage_SOCP.png", dpi=600)
        plt.show(block=False)
        
        # Save to Excel
        df = pd.DataFrame({"Node": nodes, "Min_Voltage_pu": vals})
        with pd.ExcelWriter('output_SOCP.xlsx') as writer:
            df.to_excel(writer, sheet_name='Voltage')

        SPobj = pyo.value(model.obj)
        
        # Duals Dev (for Benders)
        duals_Dev = {
            (n, t): model.dual[model.conV_dev1[n, t]]
            for n in model.N for t in model.T
            if model.conV_dev1[n, t] in model.dual
        }
        duals_DevLow = { # Not strictly used in SOCP logic but kept for compatibility
            (n, t): model.dual[model.con_abs_lower[n, t]]
            for n in model.N for t in model.T
            if model.con_abs_lower[n, t] in model.dual
        }
        duals_DevUp = {
            (n, t): model.dual[model.con_abs_upper[n, t]]
            for n in model.N for t in model.T
            if model.con_abs_upper[n, t] in model.dual
        }

        return min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj

    else:
        print(f"SOCP Solver Failed: {results.solver.termination_condition}")
        return None, None, None, None, None, None, None