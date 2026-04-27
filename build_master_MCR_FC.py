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

def build_master(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price):
    [parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

    model = pyo.ConcreteModel()

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
    PFV_Rob = (1 / 365) * ((IR * (1 + IR) ** NYearRob) / (-1 + (1 + IR) ** NYearRob))
    
    # TIGHT BIG-M CALCULATION
    EVdata['MaxPowerNeeded'] = EVdata.apply(
        lambda row: max(0.001, (row['SOCout'] - row['SOCin']) * row['EVcap'] / max(1, (row['DT'] - row['AT']) / SampPerH)), 
        axis=1
    )

    model.HORIZON = SampPerH * 24
    model.nEV = len(EVdata) - 1
    
    if s == 2:
        nMCR = 12
    else:
        nMCR = 10
    
    model.nMESS = nMCR
    model.RobType = 1 * len(robotCC)
    
    # NOTICE: model.I (Individual Chargers) is COMPLETELY REMOVED
    model.Nodes = pyo.Set(initialize=range(33))
    model.T = pyo.Set(initialize=[x + 1 for x in range(model.HORIZON)])
    model.K = pyo.Set(initialize=[x + 1 for x in range(model.nEV)])
    model.J = pyo.Set(initialize=[x + 1 for x in range(model.nMESS)])
    model.KK = pyo.Set(initialize=[x + 1 for x in range(model.RobType)])
    model.rob_ev_pairs = pyo.Set(dimen=2, initialize=lambda m: [(k,j) for k in m.K for j in m.J])
    model.y_indices = pyo.Set(dimen=3, initialize=lambda model: (
        (k, j, t)
        for k in model.K
        for j in model.J
        for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
    ))
    
    
    model.y_link = pyo.Var(model.y_indices, within=pyo.Binary)
    # --- VARIABLES ---
    # Replaced assign[k,i] and z[i] with a single EV decision and Ns bounds
    model.grid_charge = pyo.Var(model.K, within=pyo.Binary) # 1 if EV uses grid charger, 0 if robot
    model.assignRobot = pyo.Var(model.K, model.J, within=pyo.Binary)
    model.u_rob = pyo.Var(model.J, model.T, within=pyo.Binary) # 1 if robot is physically plugged into grid
    model.u_rob_type = pyo.Var(model.J, model.KK, within=pyo.Binary)
    
    # Ns bounded by hardcoded limits (replaces z[i].fix(1))
    if s == 1:
       lb_ns = 30
    elif s == 2 :
        lb_ns = 0
    else:
        lb_ns = 30
     
   #lb_ns = 0 if s == 2 or s== 1 else 0
    ub_ns = max_overlaps[s]
    model.Ns = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(lb_ns, ub_ns))
    
    model.P_btot = pyo.Var(model.T, within=pyo.NonNegativeReals)
    
    # AGGREGATED POWER VARIABLES
    model.P_b_EV_grid = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.Out1_P_b_EV = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.P_ch_rob_total = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    #model.P_dch_rob = pyo.Var(model.rob_ev_pairs, model.T, within=pyo.NonNegativeReals)
    model.P_dch_rob = pyo.Var(model.y_indices, within=pyo.NonNegativeReals)
    
    model.P_ch_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.SOC_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.SOC_rob = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    model.CapRobot = pyo.Var(model.J, model.KK, within=pyo.NonNegativeReals)
    model.PeakPower = pyo.Var(within=pyo.NonNegativeReals)
    
    # Benders Decomposition Variables
    model.Alpha = pyo.Var(within=pyo.Reals)
    model.AlphaDown = pyo.Param(initialize=-100, mutable=True)
    

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

    # 2. EV SOURCING (Grid vs Robot) - Extremely tight LP relaxation
    def ChargingOptions(model, k):
        return model.grid_charge[k] + sum(model.assignRobot[k, j] for j in model.J) <= 1    
    model.ConChargingOptions = pyo.Constraint(model.K, rule=ChargingOptions)

    def Link_Grid_Assign(model, k, t):
        if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
            return model.P_b_EV_grid[k, t] == 0
        # TIGHT! Forces grid_charge[k] to 1.0 in LP if EV needs grid power
        return model.P_b_EV_grid[k, t] <= EVdata['MaxPowerNeeded'][k] * model.grid_charge[k]
    model.ConLinkGridAssign = pyo.Constraint(model.K, model.T, rule=Link_Grid_Assign)

    # 3. AGGREGATE PHYSICAL LIMITS (Replaces millions of x_rob and occupancy variables)
    def PlugAvailability(model, t):
        # EVs taking a physical wall plug
        evs_plugged = sum(model.grid_charge[k] for k in model.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        # Robots taking a physical wall plug to charge themselves
        robots_charging = sum(model.u_rob[j, t] for j in model.J)
        return evs_plugged + robots_charging <= model.Ns
    model.ConPlugAvailability = pyo.Constraint(model.T, rule=PlugAvailability)
    
    
    
    def Robot_dischargeAvailability(model, k, j, t):
        return model.P_dch_rob[k, j, t] <= (1 - model.u_rob[j, t]) *DCchargerCap
    model.ConRobot_dischargeAvailability = pyo.Constraint(model.y_indices, rule=Robot_dischargeAvailability)
    
    

    def PowerAvailability(model, t):
        ev_power = sum(model.P_b_EV_grid[k, t] for k in model.K)
        rob_power = sum(model.P_ch_rob_total[j, t] for j in model.J)
        return ev_power + rob_power <= model.Ns * ChargerCap
    model.ConPowerAvailability = pyo.Constraint(model.T, rule=PowerAvailability)

    # 4. EV CHARGING & SOC
    def Charging_toEV(model, k, t):
        # CHANGED: Added 'if (k,j,t) in model.y_indices' to handle sparse variable
        robot_contrib = sum(model.P_dch_rob[k,j,t] for j in model.J if (k,j,t) in model.y_indices)
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

    # 5. ROBOT CHARGING & DISCHARGING
    def Ch_robot_limit_total(model, j, t):
        return model.P_ch_rob_total[j, t] <= model.u_rob[j, t] * ChargerCap
    model.ConCh_robot_total = pyo.Constraint(model.J, model.T, rule=Ch_robot_limit_total)


    def Dch_rob_cap_limit(model, k, j, t):
        return model.P_dch_rob[k, j, t] <= sum(model.CapRobot[j, kk] for kk in model.KK)
    model.ConDch_rob_cap_limit = pyo.Constraint(model.y_indices, rule=Dch_rob_cap_limit)

    def SingleRobotPerEV(model, k, t):
        if t < EVdata['AT'][k] or t > EVdata['DT'][k]: return pyo.Constraint.Skip
        return sum(model.P_dch_rob[k, j, t] for j in model.J if (k, j, t) in model.y_indices) <= DCchargerCap
    model.ConSingleRobotPerEV = pyo.Constraint(model.K, model.T, rule=SingleRobotPerEV)

    def Dch_rob_limit_new(model, k, j, t):
        # CHANGED: Removed the redundant 'if t < AT...' check because y_indices handles it
        return model.P_dch_rob[k, j, t] <= model.assignRobot[k, j] * DCchargerCap
    # CHANGED: Removed ', model.T' from the set initialization
    model.ConDch_rob_limit_new = pyo.Constraint(model.y_indices, rule=Dch_rob_limit_new)

    # 6. ROBOT SOC & TYPE SELECTION
    def SOC_Robot(model, j, t):
       if t == 1: return model.SOC_rob[j,t] == 0.2 * sum(model.CapRobot[j,kk] for kk in model.KK)
       # CHANGED: Added 'if (k,j,t) in model.y_indices'
       active_discharges = sum(model.P_dch_rob[k,j,t] for k in model.K if (k,j,t) in model.y_indices)
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

#######################   
    #def OneRobotOneEV_at_a_Time(model, j, t):
    # Sum of y_link for all EVs must be <= 1
    # one EV can be charged at the same time
    #    return sum(model.y_link[k, j, t] for k in model.K if (k, j, t) in model.y_indices) <= 1
    #model.ConOneRobotOneEV = pyo.Constraint(model.J, model.T, rule=OneRobotOneEV_at_a_Time)

    def OneRobotOneEV_at_a_Time(model, j, t):
        # Check if there are ANY valid EVs to connect to at this time
        # If not, skip creating the constraint
        valid_evs = [k for k in model.K if (k, j, t) in model.y_indices]
        if not valid_evs:
            return pyo.Constraint.Skip
            
        # Limit active connections to 1
        return sum(model.y_link[k, j, t] for k in valid_evs) <= 1
    
    model.ConOneRobotOneEV = pyo.Constraint(model.J, model.T, rule=OneRobotOneEV_at_a_Time)


    def OneEVOneRobot_at_a_Time(model, k, t):
        # If the EV is not present at this time, skip the constraint
        if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
            return pyo.Constraint.Skip
            
        # If present, ensure only 1 robot is connected
        # We don't need 'if (k,j,t) in model.y_indices' here because 
        # we already confirmed t is valid for this EV.
        return sum(model.y_link[k, j, t] for j in model.J) <= 1
    
    model.ConOneEVOneRobot_at_a_Time = pyo.Constraint(model.K, model.T, rule=OneEVOneRobot_at_a_Time)



    def LinkDchPower(model, k, j, t):
        # If P_dch_rob > 0, y_link must be 1.
        # If y_link is 0, P_dch_rob must be 0.
        return model.P_dch_rob[k, j, t] <= model.y_link[k, j, t] * DCchargerCap
    model.ConLinkDchPower = pyo.Constraint(model.y_indices, rule=LinkDchPower)


    def LinkToStaticAssign(model, k, j, t):
        # y_link cannot be 1 unless assignRobot is 1
        return model.y_link[k, j, t] <= model.assignRobot[k, j]
    model.ConLinkToStaticAssign = pyo.Constraint(model.y_indices, rule=LinkToStaticAssign)

######################
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