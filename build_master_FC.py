# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:01:57 2025

@author: arsalann
"""
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from MaxOverlap import max_overlaps_per_parking

from GlobalData import GlobalData



[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

    
   
####################################################################################################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
##########  ONLY FIXED CHARGER #################

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:58:11 2026

@author: arsalann
"""

# -*- coding: utf-8 -*-
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PowerFlow import PowerFlow
from DataCuration import DataCuration
from MaxOverlap import max_overlaps_per_parking
from GlobalData import GlobalData

[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

def build_masterOnlyFC(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price):
    [parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

    modelFC = pyo.ConcreteModel()

    EVdata = parking_data[parking_data['ParkingNo'] == s].reset_index(drop=True)
    max_overlaps = max_overlaps_per_parking(EVdata)
    
    print(f'Parking: {s}, Max overlaps: {max_overlaps[s]}, Total EVs: {len(EVdata)}')

    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')
    line_data = data['lineData']

    ATT = EVdata['AT']
    DTT = EVdata['DT']
    
    plt.subplot(2,1,1)
    plt.hist(ATT, bins=18, color='yellow', edgecolor='brown', label='Arrival time')
    plt.legend(); plt.grid(); plt.ylabel('Frequency'); plt.xlim(1,SampPerH*24)
    
    plt.subplot(2,1,2)
    plt.hist(DTT, bins=18, color='black', edgecolor='brown', label='Departure time')
    plt.xlim(1,SampPerH*24); plt.grid(); plt.legend(); plt.xlabel('Time sample'); plt.ylabel('Frequency')
    plt.savefig('Histogram.png', dpi=300); plt.show(block=False)

    PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
    
    # TIGHT BIG-M CALCULATION
    EVdata['MaxPowerNeeded'] = EVdata.apply(
        lambda row: max(0.001, (row['SOCout'] - row['SOCin']) * row['EVcap'] / max(1, (row['DT'] - row['AT']) / SampPerH)), 
        axis=1
    )

    modelFC.HORIZON = SampPerH * 24
    modelFC.nEV = len(EVdata) - 1
    
    # --- SETS ---
    # Removed: modelFC.J, modelFC.KK, modelFC.rob_ev_pairs, modelFC.y_indices
    modelFC.Nodes = pyo.Set(initialize=range(33))
    modelFC.T = pyo.Set(initialize=[x + 1 for x in range(modelFC.HORIZON)])
    modelFC.K = pyo.Set(initialize=[x + 1 for x in range(modelFC.nEV)])
    
    # --- VARIABLES ---
    
    # Changed: Replaced static 'grid_charge' with time-dependent 'is_charging' to allow smart charging optimization of Ns
    modelFC.is_charging = pyo.Var(modelFC.K, modelFC.T, within=pyo.Binary) # 1 if EV k is plugged in at time t
    modelFC.Ns = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(0, max_overlaps[s]+1))
    
    modelFC.P_btot = pyo.Var(modelFC.T, within=pyo.NonNegativeReals)
    modelFC.P_b_EV_grid = pyo.Var(modelFC.K, modelFC.T, within=pyo.NonNegativeReals)
    modelFC.Out1_P_b_EV = pyo.Var(modelFC.T, within=pyo.NonNegativeReals)
    modelFC.P_ch_EV = pyo.Var(modelFC.K, modelFC.T, within=pyo.NonNegativeReals)
    modelFC.SOC_EV = pyo.Var(modelFC.K, modelFC.T, within=pyo.NonNegativeReals)
    modelFC.PeakPower = pyo.Var(within=pyo.NonNegativeReals)
    
    # Benders Decomposition Variables
    modelFC.Alpha = pyo.Var(within=pyo.Reals)
    modelFC.AlphaDown = pyo.Param(initialize=-100, mutable=True)

    # --- CONSTRAINTS ---
    
    # 1. POWER BALANCE
    # Removed: robot_power
    def PowerPurchased(modelFC, t):
        ev_power = sum(modelFC.P_b_EV_grid[k, t] for k in modelFC.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        return modelFC.P_btot[t] == ev_power
    modelFC.ConPurchasedPower = pyo.Constraint(modelFC.T, rule=PowerPurchased) 

    def Out1_P_b_EV_rule(modelFC, t):
        ev_power = sum(modelFC.P_b_EV_grid[k, t] for k in modelFC.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        return modelFC.Out1_P_b_EV[t] == ev_power 
    modelFC.ConOut1_P_b_EV = pyo.Constraint(modelFC.T, rule=Out1_P_b_EV_rule)  

    def PowerPurchasedLimit(modelFC, t):
        return modelFC.P_btot[t] <= PgridMax
    modelFC.ConPowerPurchasedLimit = pyo.Constraint(modelFC.T, rule=PowerPurchasedLimit)

    # 2. FIXED CHARGER CONSTRAINTS
    
    # Link EV power to binary charging status
    def Link_Grid_Assign(modelFC, k, t):
        if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
            return modelFC.P_b_EV_grid[k, t] == 0
        return modelFC.P_b_EV_grid[k, t] <= EVdata['MaxPowerNeeded'][k] * modelFC.is_charging[k, t]
    modelFC.ConLinkGridAssign = pyo.Constraint(modelFC.K, modelFC.T, rule=Link_Grid_Assign)

    # Plug Availability
    # Removed: robots_charging
    def PlugAvailability(modelFC, t):
        evs_plugged = sum(modelFC.is_charging[k, t] for k in modelFC.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        return evs_plugged <= modelFC.Ns
    modelFC.ConPlugAvailability = pyo.Constraint(modelFC.T, rule=PlugAvailability)
    
    # Power Availability
    # Removed: rob_power
    def PowerAvailability(modelFC, t):
        ev_power = sum(modelFC.P_b_EV_grid[k, t] for k in modelFC.K)
        return ev_power <= modelFC.Ns * ChargerCap
    modelFC.ConPowerAvailability = pyo.Constraint(modelFC.T, rule=PowerAvailability)

    # 3. EV CHARGING & SOC
    # Removed: robot_contrib
    def Charging_toEV(modelFC, k, t):
        grid_contrib = modelFC.P_b_EV_grid[k, t]
        return modelFC.P_ch_EV[k,t] == grid_contrib
    modelFC.ConCharging_toEV = pyo.Constraint(modelFC.K, modelFC.T, rule=Charging_toEV)

    def SOC_EV_f1(modelFC, k, t):
        if t < EVdata['AT'][k]: return modelFC.SOC_EV[k, t] == 0
        elif t == EVdata['AT'][k]: return modelFC.SOC_EV[k, t] == EVdata['SOCin'][k] * EVdata['EVcap'][k]
        elif t <= EVdata['DT'][k]: return modelFC.SOC_EV[k, t] == modelFC.SOC_EV[k, t - 1] + (1 / SampPerH) * modelFC.P_ch_EV[k, t]
        return pyo.Constraint.Skip
    modelFC.ConSOC_EV_f1 = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_EV_f1)

    def SOC_EV_f2(modelFC, k, t):
        if t == EVdata['DT'][k]: return modelFC.SOC_EV[k, t] == EVdata['SOCout'][k] * EVdata['EVcap'][k]
        return pyo.Constraint.Skip
    modelFC.ConSOC_EV_f2 = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_EV_f2)

    def SOC_Limit(modelFC, k, t):
        return modelFC.SOC_EV[k, t] <= EVdata['EVcap'][k]   
    modelFC.ConSOC_Limit = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_Limit)

    # 4. PEAK POWER & BENDERS
    # Removed: All Robot SOC/Charging/Discharging constraints
    def PeakPowerConstraint(modelFC, t):
        return modelFC.PeakPower >= modelFC.P_btot[t]               
    modelFC.ConPeakPower = pyo.Constraint(modelFC.T, rule=PeakPowerConstraint)

    def AlphaFun(model):
        return modelFC.Alpha >= modelFC.AlphaDown
    modelFC.ConAlphaFun = pyo.Constraint(rule=AlphaFun)

    
    modelFC.ConNs = pyo.Constraint(rule = modelFC.Ns >= max_overlaps[s])

    # --- OBJECTIVE FUNCTION ---
    # Removed: PFV_Rob (Robot Investment Cost)
    modelFC.obj = pyo.Objective(
        expr=(1) * (PFV_Charger * modelFC.Ns * Ch_cost ) +
             (1/SampPerH) * sum(Price.iloc[t - 1] * 0.001 * modelFC.P_btot[t] for t in modelFC.T) +
             (1 / 30) * PeakPrice * modelFC.PeakPower +
             1 * modelFC.Alpha, 
        sense=pyo.minimize
    )

    # Benders Cut Placeholder
    modelFC.cuts = pyo.ConstraintList()
    
    return modelFC