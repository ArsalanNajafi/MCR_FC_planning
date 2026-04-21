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
def build_masterBESS(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price):
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
    # Assuming Battery investment cost is similar to Robot cost structure (using NYearRob)
    PFV_Batt = (1 / 365) * ((IR * (1 + IR) ** NYearRob) / (-1 + (1 + IR) ** NYearRob))
    
    # TIGHT BIG-M CALCULATION
    EVdata['MaxPowerNeeded'] = EVdata.apply(
        lambda row: max(0.001, (row['SOCout'] - row['SOCin']) * row['EVcap'] / max(1, (row['DT'] - row['AT']) / SampPerH)), 
        axis=1
    )

    model.HORIZON = SampPerH * 24
    model.nEV = len(EVdata) - 1
    model.nMESS = 10  # Number of Batteries (previously Robots)
    model.BattType = 1 * len(RobotTypes) # Using RobotTypes as BatteryTypes
    
    # --- SETS ---
    model.Nodes = pyo.Set(initialize=range(33))
    model.T = pyo.Set(initialize=[x + 1 for x in range(model.HORIZON)])
    model.K = pyo.Set(initialize=[x + 1 for x in range(model.nEV)]) # EVs
    model.J = pyo.Set(initialize=[x + 1 for x in range(model.nMESS)]) # Batteries
    model.KK = pyo.Set(initialize=[x + 1 for x in range(model.BattType)]) # Battery Types
    
    # --- VARIABLES ---
    
    # EV Charging Variables
    # is_charging[k,t]: 1 if EV k is physically plugged in and charging at time t
    model.is_charging = pyo.Var(model.K, model.T, within=pyo.Binary)
    model.P_ch_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.SOC_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    
    # Battery Variables
    model.CapBatt = pyo.Var(model.J, model.KK, within=pyo.NonNegativeReals) # Capacity of battery j
    model.u_batt_type = pyo.Var(model.J, model.KK, within=pyo.Binary)      # 1 if battery j is of type kk
    model.SOC_Batt = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    model.P_ch_batt = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    model.P_dch_batt = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    
    # Binary for Battery State (to prevent simultaneous charge/discharge)
    model.z_ch_batt = pyo.Var(model.J, model.T, within=pyo.Binary)
    model.z_dch_batt = pyo.Var(model.J, model.T, within=pyo.Binary)
    model.Out1_P_b_EV = pyo.Var(model.T, within = pyo.NonNegativeReals)
    # Grid / Infrastructure Variables
    # Ns: Number of fixed chargers (Investment decision)
    
    lb_ns = max_overlaps[s] 
    ub_ns = max_overlaps[s] + 1
    model.Ns = pyo.Var(within=pyo.NonNegativeIntegers, bounds=(lb_ns, ub_ns))
    
    model.P_btot = pyo.Var(model.T, within=pyo.NonNegativeReals) # Total power from grid
    model.PeakPower = pyo.Var(within=pyo.NonNegativeReals)
    
    # Benders Decomposition Variables
    model.Alpha = pyo.Var(within=pyo.Reals)
    model.AlphaDown = pyo.Param(initialize=-100, mutable=True)

    # --- CONSTRAINTS ---
    
    # 1. POWER BALANCE
    # P_btot = Power to EVs + Power to charge Batteries - Power discharged from Batteries
    def PowerPurchased(model, t):
        ev_power = sum(model.P_ch_EV[k, t] for k in model.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        batt_net = sum(model.P_ch_batt[j, t] - model.P_dch_batt[j, t] for j in model.J)
        return model.P_btot[t] == ev_power + batt_net
    model.ConPurchasedPower = pyo.Constraint(model.T, rule=PowerPurchased) 

    def Out1_P_b_EV_rule(model, t):
        # Total power consumed by EVs (for tracking or sub-problem input)
        ev_power = sum(model.P_ch_EV[k, t] for k in model.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        return model.Out1_P_b_EV[t] == ev_power 
    model.ConOut1_P_b_EV = pyo.Constraint(model.T, rule=Out1_P_b_EV_rule)  

    def PowerPurchasedLimit(model, t):
        return model.P_btot[t] <= PgridMax
    model.ConPowerPurchasedLimit = pyo.Constraint(model.T, rule=PowerPurchasedLimit)

    # 2. FIXED CHARGER (PLUG) CONSTRAINTS
    
    # Link EV power to binary charging status
    def Link_EV_Power(model, k, t):
        if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
            return model.P_ch_EV[k, t] == 0
        return model.P_ch_EV[k, t] <= ChargerCap * model.is_charging[k, t]
    model.ConLink_EV_Power = pyo.Constraint(model.K, model.T, rule=Link_EV_Power)

    # Infrastructure Limit: Number of active plugs cannot exceed installed fixed chargers (Ns)
    # Note: Batteries are assumed stationary and do not consume "parking spots" or "plugs"
    def PlugAvailability(model, t):
        evs_plugged = sum(model.is_charging[k, t] for k in model.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        return evs_plugged <= model.Ns
    model.ConPlugAvailability = pyo.Constraint(model.T, rule=PlugAvailability)
    
    # Infrastructure Capacity Limit: Total EV power cannot exceed total installed charger capacity
    def PowerAvailability(model, t):
        ev_power = sum(model.P_ch_EV[k, t] for k in model.K if EVdata['AT'][k] <= t <= EVdata['DT'][k])
        return ev_power <= model.Ns * ChargerCap
    model.ConPowerAvailability = pyo.Constraint(model.T, rule=PowerAvailability)

    # 3. EV SOC CONSTRAINTS
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

    # 4. BATTERY CONSTRAINTS
    
    # Battery Power Limits (Based on DCchargerCap)
    def Batt_Ch_Limit(model, j, t):
        return model.P_ch_batt[j, t] <= DCchargerCap * model.z_ch_batt[j, t]
    model.ConBatt_Ch_Limit = pyo.Constraint(model.J, model.T, rule=Batt_Ch_Limit)
    
    def Batt_Dch_Limit(model, j, t):
        return model.P_dch_batt[j, t] <= DCchargerCap * model.z_dch_batt[j, t]
    model.ConBatt_Dch_Limit = pyo.Constraint(model.J, model.T, rule=Batt_Dch_Limit)
    
    # Prevent simultaneous charge and discharge
    def Batt_Single_Mode(model, j, t):
        return model.z_ch_batt[j, t] + model.z_dch_batt[j, t] <= 1
    model.ConBatt_Single_Mode = pyo.Constraint(model.J, model.T, rule=Batt_Single_Mode)

    # Battery SOC Dynamics
    def SOC_Batt_Dynamics(model, j, t):
       if t == 1: 
           return model.SOC_Batt[j,t] == 0.2 * sum(model.CapBatt[j,kk] for kk in model.KK)
       
       # Assuming 100% efficiency for simplicity as per original code style, 
       # or P_ch/P_dch are already net power values.
       return model.SOC_Batt[j, t] == model.SOC_Batt[j,t-1] + (1/SampPerH)*model.P_ch_batt[j, t] - (1/SampPerH)*model.P_dch_batt[j, t]
    model.ConSOC_Batt_Dynamics = pyo.Constraint(model.J, model.T, rule=SOC_Batt_Dynamics)

    def SOC_Batt_Min(model, j, t):
        return model.SOC_Batt[j, t] >= 0.2 * sum(model.CapBatt[j, kk] for kk in model.KK)
    model.ConSOC_Batt_Min = pyo.Constraint(model.J, model.T, rule=SOC_Batt_Min)

    def SOC_Batt_Max(model, j, t):
        return model.SOC_Batt[j, t] <= sum(model.CapBatt[j, kk] for kk in model.KK)
    model.ConSOC_Batt_Max = pyo.Constraint(model.J, model.T, rule=SOC_Batt_Max)

    # Battery Type Selection
    def BattCapLimit(model, j, kk):
        return model.CapBatt[j, kk] <= RobotTypes[kk - 1] * model.u_batt_type[j, kk]
    model.ConBattCapLimit = pyo.Constraint(model.J, model.KK, rule=BattCapLimit)

    def OneTypePerBatt(model, j):
        return sum(model.u_batt_type[j, kk] for kk in model.KK) <= 1
    model.ConOneTypePerBatt = pyo.Constraint(model.J, rule=OneTypePerBatt)

    def TotalBattNum(model):
        return sum(model.u_batt_type[j, kk] for j in model.J for kk in model.KK) <= MaxRobot
    model.ConTotalBattNum = pyo.Constraint(rule=TotalBattNum)

    # 5. PEAK POWER & BENDERS
    def PeakPowerConstraint(model, t):
        return model.PeakPower >= model.P_btot[t]               
    model.ConPeakPower = pyo.Constraint(model.T, rule=PeakPowerConstraint)

    def AlphaFun(model):
        return model.Alpha >= model.AlphaDown
    model.ConAlphaFun = pyo.Constraint(rule=AlphaFun)

    # --- OBJECTIVE FUNCTION ---
    # Costs:
    # 1. Investment in Fixed Chargers (Ns)
    # 2. Investment in Batteries (CapBatt)
    # 3. Cost of Energy from Grid (Price * P_btot)
    # 4. Peak Demand Charge
    model.obj = pyo.Objective(
        expr=(1) * (PFV_Charger * model.Ns * Ch_cost ) +
             (1) * PFV_Batt * sum(model.u_batt_type[j, kk] * robotCC[kk - 1] for j in model.J for kk in model.KK) +
             (1/SampPerH) * sum(Price.iloc[t - 1] * 0.001 * model.P_btot[t] for t in model.T) +
             (1 / 30) * PeakPrice * model.PeakPower +
             1 * model.Alpha, 
        sense=pyo.minimize
    )

    # Benders Cut Placeholder
    model.cuts = pyo.ConstraintList()
    
    return model    