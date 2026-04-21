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

def build_master(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price):
    [parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

    model = pyo.ConcreteModel()

    EVdata = parking_data[parking_data['ParkingNo'] == s].reset_index(drop=True)
    max_overlaps = max_overlaps_per_parking(EVdata)
    
    print(f'Parking: {s}, Max overlaps: {max_overlaps[s]}, Total EVs: {len(EVdata)}')
    model.nCharger = max_overlaps[s]

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
    plt.savefig('Histogram.png', dpi=300); plt.show()

    M = 50
    PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
    PFV_Rob = (1 / 365) * ((IR * (1 + IR) ** NYearRob) / (-1 + (1 + IR) ** NYearRob))
    
    # TIGHT BIG-M CALCULATION: Prevents fractional LP relaxation
    EVdata['MaxPowerNeeded'] = EVdata.apply(
        lambda row: max(0.001, (row['SOCout'] - row['SOCin']) * row['EVcap'] / max(1, (row['DT'] - row['AT']) / SampPerH)), 
        axis=1
    )

    model.HORIZON = SampPerH * 24
    model.nEV = len(EVdata) - 1
    model.nMESS = 12
    model.RobType = 1 * len(robotCC)
    
    model.Nodes = pyo.Set(initialize=range(33))
    model.T = pyo.Set(initialize=[x + 1 for x in range(model.HORIZON)])
    model.I = pyo.Set(initialize=[x + 1 for x in range(model.nCharger)])
    model.K = pyo.Set(initialize=[x + 1 for x in range(model.nEV)])
    model.J = pyo.Set(initialize=[x + 1 for x in range(model.nMESS)])
    model.KK = pyo.Set(initialize=[x + 1 for x in range(model.RobType)])
    model.rob_ev_pairs = pyo.Set(dimen=2, initialize=lambda m: [(k,j) for k in m.K for j in m.J])

    # --- VARIABLES ---
    model.z = pyo.Var(model.I, within=pyo.Binary)
    model.assign = pyo.Var(model.K, model.I, within=pyo.Binary)
    model.assignRobot = pyo.Var(model.K, model.J, within=pyo.Binary)
    model.u_rob = pyo.Var(model.J, model.T, within=pyo.Binary)
    model.u_rob_type = pyo.Var(model.J, model.KK, within=pyo.Binary)
    
    model.Ns = pyo.Var(within=pyo.NonNegativeIntegers)
    model.P_btot = pyo.Var(model.T, within=pyo.NonNegativeReals)
    
    # AGGREGATED POWER VARIABLES (Eliminated the 'i' dimension for massive speedup)
    model.P_b_EV_grid = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.Out1_P_b_EV = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.P_ch_rob_total = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    model.P_dch_rob = pyo.Var(model.rob_ev_pairs, model.T, within=pyo.NonNegativeReals)
    
    model.P_ch_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.SOC_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.SOC_rob = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    model.CapRobot = pyo.Var(model.J, model.KK, within=pyo.NonNegativeReals)
    model.PeakPower = pyo.Var(within=pyo.NonNegativeReals)
    
    # Benders Decomposition Variables
    model.Alpha = pyo.Var(within=pyo.Reals)
    model.AlphaDown = pyo.Param(initialize=-100, mutable=True)

    # Hardcoded charger initialization (from original)
    if s == 2:
        for i in model.I:
            if i <= 120: model.z[i].fix(1)
    else: 
        for i in model.I:
            if i <= 30: model.z[i].fix(1)

    # --- CRITICAL TIME SET FOR OVERLAPS ---
    critical_times = sorted(list(set(EVdata['AT']).union(set(EVdata['DT']))))
    model.CRITICAL_TIMES = pyo.Set(initialize=critical_times)

    # --- CONSTRAINTS ---
    
    # 1. POWER BALANCE
    def PowerPurchased(model, t):
        ev_power = sum(model.P_b_EV_grid[k, t] for k in model.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        robot_power = sum(model.P_ch_rob_total[j, t] for j in model.J)
        return model.P_btot[t] == ev_power + robot_power
    model.ConPurchasedPower = pyo.Constraint(model.T, rule=PowerPurchased) 

    def Out1_P_b_EV_rule(model, t):
        ev_power = sum(model.P_b_EV_grid[k, t] for k in model.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        return model.Out1_P_b_EV[t] == ev_power 
    model.ConOut1_P_b_EV = pyo.Constraint(model.T, rule=Out1_P_b_EV_rule)  

    def PowerPurchasedLimit(model, t):
        return model.P_btot[t] <= PgridMax
    model.ConPowerPurchasedLimit = pyo.Constraint(model.T, rule=PowerPurchasedLimit)

    # 2. CHARGER ASSIGNMENT & CAPACITY (Aggregate limits replace physical occupancies)
    def NoCharger(model, k, i):
        return model.assign[k, i] <= model.z[i]   
    model.ConNoCharger1 = pyo.Constraint(model.K, model.I, rule=NoCharger)

    def NoCharger2(model):
        return model.Ns == sum(model.z[i] for i in model.I)   
    model.ConNoCharger2 = pyo.Constraint(rule=NoCharger2)

    def ChargingOptions(model, k):
        return sum(model.assign[k, i] for i in model.I) + sum(model.assignRobot[k, j] for j in model.J) <= 1    
    model.ConChargingOptions = pyo.Constraint(model.K, rule=ChargingOptions)

    def SingleChargerAssignment(model, k):
        return sum(model.assign[k, i] for i in model.I) <= 1
    model.ConSingleAssign = pyo.Constraint(model.K, rule=SingleChargerAssignment)

    def critical_time_constraint(model, i, ct):
        active_at_ct = sum(model.assign[k, i] for k in model.K if EVdata['AT'][k] <= ct <= EVdata['DT'][k])
        return active_at_ct <= 1
    model.ConCriticalTimes = pyo.Constraint(model.I, model.CRITICAL_TIMES, rule=critical_time_constraint)

    # AGGREGATE PHYSICAL LIMITS (Replaces millions of x_rob and occupancy variables)
    def PlugAvailability(model, t):
        evs_plugged = sum(model.assign[k, i] for k in model.K if EVdata['AT'][k] <= t <= EVdata['DT'][k] for i in model.I)
        robots_charging = sum(model.u_rob[j, t] for j in model.J)
        return evs_plugged + robots_charging <= model.Ns
    model.ConPlugAvailability = pyo.Constraint(model.T, rule=PlugAvailability)

    def PowerAvailability(model, t):
        ev_power = sum(model.P_b_EV_grid[k, t] for k in model.K)
        rob_power = sum(model.P_ch_rob_total[j, t] for j in model.J)
        return ev_power + rob_power <= model.Ns * ChargerCap
    model.ConPowerAvailability = pyo.Constraint(model.T, rule=PowerAvailability)

    # TIGHT BIG-M: Links EV power to assignment (Forces LP relaxation to 0 or 1)
    def Link_Grid_Assign(model, k, t):
        if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
            return model.P_b_EV_grid[k, t] == 0
        return model.P_b_EV_grid[k, t] <= EVdata['MaxPowerNeeded'][k] * sum(model.assign[k, i] for i in model.I)
    model.ConLinkGridAssign = pyo.Constraint(model.K, model.T, rule=Link_Grid_Assign)

    # 3. EV CHARGING & SOC
    def Charging_toEV(model, k, t):
        robot_contrib = sum(model.P_dch_rob[k,j,t] for j in model.J)
        grid_contrib = model.P_b_EV_grid[k, t]
        return model.P_ch_EV[k,t] == grid_contrib + robot_contrib
    model.ConCharging_toEV = pyo.Constraint(model.K, model.T, rule=Charging_toEV)

    def SOC_EV_f1(model, k, t):
        if t < EVdata['AT'][k]: return model.SOC_EV[k, t] == 0
        elif t == EVdata['AT'][k]: return model.SOC_EV[k, t] == EVdata['SOCin'][k] * EVdata['EVcap'][k]
        elif t <= EVdata['DT'][k]: return model.SOC_EV[k, t] == model.SOC_EV[k, t - 1] + (1 / SampPerH) * model.P_ch_EV[k, t]
        return pyo.Constraint.Skip
    model.ConSOC_EV_f1 = pyo.Constraint(model.K, model.T, rule=SOC_EV_f1)

    def SOC_EV_f2(model, k, t):
        if t == EVdata['DT'][k]: return model.SOC_EV[k, t] == EVdata['SOCout'][k] * EVdata['EVcap'][k]
        return pyo.Constraint.Skip
    model.ConSOC_EV_f2 = pyo.Constraint(model.K, model.T, rule=SOC_EV_f2)

    def SOC_Limit(model, k, t):
        return model.SOC_EV[k, t] <= EVdata['EVcap'][k]   
    model.ConSOC_Limit = pyo.Constraint(model.K, model.T, rule=SOC_Limit)

    # 4. ROBOT CHARGING & DISCHARGING
    def Ch_robot_limit_total(model, j, t):
        return model.P_ch_rob_total[j, t] <= model.u_rob[j, t] * ChargerCap
    model.ConCh_robot_total = pyo.Constraint(model.J, model.T, rule=Ch_robot_limit_total)

    def Dch_rob_limit_new(model, k, j, t):
        if t < EVdata['AT'][k] or t > EVdata['DT'][k]: return model.P_dch_rob[k, j, t] == 0
        return model.P_dch_rob[k, j, t] <= model.assignRobot[k, j] * DCchargerCap
    model.ConDch_rob_limit_new = pyo.Constraint(model.rob_ev_pairs, model.T, rule=Dch_rob_limit_new)

    def Dch_rob_cap_limit(model, k, j, t):
        return model.P_dch_rob[k, j, t] <= sum(model.CapRobot[j, kk] for kk in model.KK)
    model.ConDch_rob_cap_limit = pyo.Constraint(model.rob_ev_pairs, model.T, rule=Dch_rob_cap_limit)

    def SingleRobotPerEV(model, k, t):
        if t < EVdata['AT'][k] or t > EVdata['DT'][k]: return pyo.Constraint.Skip
        return sum(model.P_dch_rob[k, j, t] for j in model.J) <= DCchargerCap
    model.ConSingleRobotPerEV = pyo.Constraint(model.K, model.T, rule=SingleRobotPerEV)

    def Dch_robot_limit1_new(model, j, t):
        active_discharges = sum(model.P_dch_rob[k, j, t] for k in model.K)
        return active_discharges <= (1 - model.u_rob[j, t]) * NevSame * DCchargerCap
    model.ConDch_robot1_new = pyo.Constraint(model.J, model.T, rule=Dch_robot_limit1_new)

    # 5. ROBOT SOC & TYPE SELECTION
    def SOC_Robot(model, j, t):
        if t == 1: return model.SOC_rob[j,t] == 0.2 * sum(model.CapRobot[j,kk] for kk in model.KK)
        active_discharges = sum(model.P_dch_rob[k,j,t] for k in model.K)
        return model.SOC_rob[j, t] == model.SOC_rob[j,t-1] + (1/SampPerH)*model.P_ch_rob_total[j, t] - (1/SampPerH)*active_discharges
    model.ConSOC_Robot = pyo.Constraint(model.J, model.T, rule=SOC_Robot)

    def SOC_Robot2(model, j, t):
        return model.SOC_rob[j, t] >= 0.2 * sum(model.CapRobot[j, kk] for kk in model.KK)
    model.ConSOC_Robot2 = pyo.Constraint(model.J, model.T, rule=SOC_Robot2)

    def SOC_Robot_limit(model, j, t):
        return model.SOC_rob[j, t] <= sum(model.CapRobot[j, kk] for kk in model.KK)
    model.ConSOC_Robot_limit = pyo.Constraint(model.J, model.T, rule=SOC_Robot_limit)

    def RobotCapLimit4(model, j, kk):
        return model.CapRobot[j, kk] <= RobotTypes[kk - 1] * model.u_rob_type[j, kk]
    model.ConRobotCapLimit4 = pyo.Constraint(model.J, model.KK, rule=RobotCapLimit4)

    def RobotCapLimit2(model, j, kk):
        return sum(model.u_rob_type[j, kk] for kk in model.KK) <= 1
    model.ConRobotCapLimit2 = pyo.Constraint(model.J, model.KK, rule=RobotCapLimit2)

    def RobotTotNum(model):
        return sum(model.u_rob_type[j, kk] for j in model.J for kk in model.KK) <= MaxRobot
    model.ConRobotTotNum = pyo.Constraint(rule=RobotTotNum)

    # 6. SYMMETRY BREAKING (Crucial for speed)
    def symmetry_breaking(model, i):
        if i > 1: return model.z[i] <= model.z[i-1]
        return pyo.Constraint.Skip
    model.ConSymmetryBreaking = pyo.Constraint(model.I, rule=symmetry_breaking) 

    # 7. PEAK POWER & BENDERS
    def PeakPowerConstraint(model, t):
        return model.PeakPower >= model.P_btot[t]               
    model.ConPeakPower = pyo.Constraint(model.T, rule=PeakPowerConstraint)

    def AlphaFun(model):
        return model.Alpha >= model.AlphaDown
    model.ConAlphaFun = pyo.Constraint(rule=AlphaFun)

    # --- OBJECTIVE FUNCTION ---
    model.obj = pyo.Objective(
        expr=(1) * (PFV_Charger * model.Ns * Ch_cost ) +
             (1) * PFV_Rob * sum(model.u_rob_type[j, kk] * robotCC[kk - 1] for j in model.J for kk in model.KK) +
             (1/SampPerH) * sum(Price.iloc[t - 1] * 0.001 * model.P_btot[t] for t in model.T) +
             (1 / 30) * PeakPrice * model.PeakPower +
             1 * model.Alpha, 
        sense=pyo.minimize
    )

    # Benders Cut Placeholder
    model.cuts = pyo.ConstraintList()
    
    return model