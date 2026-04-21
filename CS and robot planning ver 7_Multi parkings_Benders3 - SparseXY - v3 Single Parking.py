# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 11:22:13 2025

@author: arsalann
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:25:24 2025

@author: arsalann
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:03:30 2025

@author: arsalann
"""

import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from PowerFlow import PowerFlow
from PowerFlow import PowerFlow
from DataCuration import DataCuration
from MaxOverlap import max_overlaps_per_parking

from GlobalData import GlobalData


model = pyo.ConcreteModel()



# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 14:25:24 2025

@author: arsalann
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:03:30 2025

@author: arsalann
"""





[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()



vb = 12.66
sb = 10
model.ParkNo = 1


OmkarsData = 0
MyCase = 1


current_directory = os.path.dirname(__file__)
current_directory = os.getcwd()
file_path         = os.path.join(current_directory, 'data.xlsx')
file_path2         = os.path.join(current_directory, 'filtered_charging_events2_Updated2.xlsx')
file_path3         = os.path.join(current_directory, 'day2PublicWork.xlsx')
Price = pd.read_excel(file_path, sheet_name='electricicty_price')
Pattern = pd.read_excel(file_path, sheet_name='DemandPattern')
Pattern = np.repeat(Pattern['Pattern'], SampPerH)
Price = np.repeat(Price['Price'], SampPerH)
    
if OmkarsData == 1:

#    EVdata = DataCuration(file_path2, SampPerH, ChargerCap)
    df = pd.read_excel(file_path3, sheet_name='Sheet1')
    EVdata = DataCuration(df, SampPerH, ChargerCap, model.ParkNo )
       
    # Calculate and print results
    max_overlaps = max_overlaps_per_parking(EVdata)
    for parking_no, max_overlap in max_overlaps.items():
        print(f"ParkingNo {parking_no}: Maximum overlaps = {max_overlap}")
    
    model.nCharger = 177


    
    #Price = pd.DataFrame(Price, columns=['Price'])
elif MyCase == 1:

    EVdata = pd.read_excel(file_path, sheet_name='ParkingData_ver5')

    # Calculate and print results
    max_overlaps = max_overlaps_per_parking(EVdata)
    for parking_no, max_overlap in max_overlaps.items():
        print(f"ParkingNo {parking_no}: Maximum overlaps = {max_overlap}")

    model.nCharger = 95



data = pyo.DataPortal()
data.load(filename='iee33_bus_data.dat')

line_data = data['lineData']

#################
ATT = EVdata['AT']
DTT = EVdata['DT']


plt.subplot(2,1,1)
plt.hist(ATT, bins=20, color='skyblue', edgecolor='black',label = 'Arrival time')
plt.legend()
plt.grid()
plt.ylabel('Frequency')

plt.xlim(1,48)
plt.subplot(2,1,2)
plt.hist(DTT, bins=20, color='red', edgecolor='black', label = 'Departure time')
plt.xlim(1,48)
plt.grid()
plt.legend()
plt.xlabel('Time sample')
plt.ylabel('Frequency')
plt.savefig('Histogram.png', dpi = 300)

plt.show()
#######################
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
                alpha[b, s] = -20  # No self-sensitivity
                continue
                
            try:
                path_indices = find_path(s, b, line_data)
                total_r = sum(line_data[k][2] for k in path_indices)
                alpha[b, s] = -total_r  # Negative sign: voltage drops with increased load
            except ValueError:
                alpha[b, s] = 0  # No path exists
    
    return alpha

AlphaSensi = compute_voltage_sensitivity(line_data, 33)
df_alpha = pd.DataFrame(
    AlphaSensi,
    index=[f"Bus_{i+1}" for i in range(34)],
    columns=[f"Bus_{i+1}" for i in range(34)]
)





# Export to Excel
#output_filename = "AlphaSensi_Matrix.xlsx"
#df_alpha.to_excel(output_filename, index=True)

#print(f"AlphaSensi matrix successfully exported to {output_filename}")
Path = find_path(1, 10, line_data)


M = 50 #Big number
M2 = 5000 # Big number
# present value factors
PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
PFV_Rob = (1 / 365) * ((IR * (1 + IR) ** NYearRob) / (-1 + (1 + IR) ** NYearRob))

print(len(EVdata))
model.HORIZON = SampPerH * 24
model.nEV = len(EVdata) - 1
model.nMESS = 1*10
model.RobType = 1 * len(robotCC)

model.Nodes = pyo.Set(initialize=range(33))  # Buses 0 to 32
model.T = pyo.Set(initialize=[x + 1 for x in range(model.HORIZON)])
model.I = pyo.Set(initialize=[x + 1 for x in range(model.nCharger)])
model.K = pyo.Set(initialize=[x + 1 for x in range(model.nEV)])
model.J = pyo.Set(initialize=[x + 1 for x in range(model.nMESS)])
model.KK = pyo.Set(initialize=[x + 1 for x in range(model.RobType)])
model.S = pyo.Set(initialize=[x + 1 for x in range(model.ParkNo)])



#for i in range(33):
#    for j in range(33):
#        if AlphaSensi[i,j] != 0:
#            print(f'AlphaSensi[{i},{j}] = {AlphaSensi[i,j]:.4f}')


# x = binary varible for CS, y = binary variable for robot charger, z = binary variable for choosing CS, zz= number of robots
model.x = pyo.Var(model.K, model.I, model.S, model.T, within=pyo.Binary)
model.y = pyo.Var(model.K, model.J, model.S, model.T, within=pyo.Binary)

model.x_indices = pyo.Set(dimen=4, initialize=lambda model: (
    (k, i, EVdata['ParkingNo'][k - 1], t)
    for k in model.K
    for i in model.I
    for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
))

model.y_indices = pyo.Set(dimen=4, initialize=lambda model: (
    (k, j, EVdata['ParkingNo'][k - 1], t)
    for k in model.K
    for j in model.J
    for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
))

# Then create the variables using these sparse index sets
model.x = pyo.Var(model.x_indices, within=pyo.Binary)
model.y = pyo.Var(model.y_indices, within=pyo.Binary)

model.z = pyo.Var(model.I, model.S, within=pyo.Binary)
model.zz = pyo.Var(model.J, within=pyo.Binary)

model.assign = pyo.Var(model.K, model.I, model.S, within=pyo.Binary)  # EV-charger assignment
model.assignRobot = pyo.Var(model.K, model.J, model.S, within=pyo.Binary)  # Robot assignment

# u = binary variable to buy either from the grid or from the robots
model.u = pyo.Var(model.K, within=pyo.Binary)
model.u_rob = pyo.Var(model.J, model.S, model.T, within=pyo.Binary)

# binary variable to define the type of the robot
model.u_rob_type = pyo.Var(model.J, model.S, model.KK, within=pyo.Binary)

# Ns = number of chargers, Nrob = number of robots
model.Ns = pyo.Var(model.S, within=pyo.Integers)
model.Nrob = pyo.Var(within=pyo.Integers)

# P_btot = P_buy_total from the grid to charge the EVs and robots, P_b_EV= Purchased electricity to charge EVS, P_b_rob = Purchased electricity to charge robots
model.P_btot = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.P_b_EV = pyo.Var(model.K, model.I, model.S, model.T, within=pyo.NonNegativeReals)
model.P_b_rob = pyo.Var(model.J, model.S, model.T, within=pyo.NonNegativeReals)

model.P_btotBar = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)

# P_ch_EV= charging to EV, P_dch_EV = discharging from the EV, SOC_EV = state of charge of the EV
model.P_ch_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
model.P_dch_EV = pyo.Var(model.K, model.I, model.T, within=pyo.NonNegativeReals)
model.SOC_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
model.P_ch_rob = pyo.Var(model.J, model.S, model.T, within=pyo.NonNegativeReals)
model.P_dch_rob = pyo.Var(model.y_indices, within=pyo.NonNegativeReals)
model.SOC_rob = pyo.Var(model.J, model.S, model.T, within=pyo.NonNegativeReals)

# Capacity of the robot
model.CapRobot = pyo.Var(model.J, model.S, model.KK, within=pyo.NonNegativeReals)
model.PeakPower = pyo.Var(model.S, within=pyo.NonNegativeReals)  # Peak grid power
model.robot_parking_assignment = pyo.Var(model.J, model.S, within=pyo.Binary)

model.Alpha = pyo.Var(within=pyo.Reals)
model.AlphaDown = pyo.Var(initialize=float('-inf'), within=pyo.Reals)

voltage_at_bus = {b: 1.0 for b in model.Nodes}  # All buses start feasible


################ Energy constraints ###########
def PowerPurchased(model, s, t):
    return model.P_btot[s, t] == (
            sum(model.P_b_EV[k, i, s, t] for k in model.K for i in model.I) +
            sum(model.P_ch_rob[j, s, t] for j in model.J)
    )


model.ConPurchasedPower = pyo.Constraint(model.S, model.T, rule=PowerPurchased)


def PowerPurchasedLimit(model, s, t):
    return model.P_btot[s, t] <= PgridMax


model.ConPowerPurchasedLimit = pyo.Constraint(model.S, model.T, rule=PowerPurchasedLimit)


##################### Charger assignment constraints ###############
def NoCharger(model, k, i, s):
    return model.assign[k, i, s] <= model.z[i, s]


model.ConNoCharger1 = pyo.Constraint(model.K, model.I, model.S, rule=NoCharger)


def NoCharger2(model, s):
    return model.Ns[s] == sum(model.z[i, s] for i in model.I)


model.ConNoCharger2 = pyo.Constraint(model.S, rule=NoCharger2)


def ChargingOptions(model, k, s):
    return sum(model.assign[k, i, s] for i in model.I) + sum(model.assignRobot[k, j, s] for j in model.J) <= 1


model.ConChargingOptions = pyo.Constraint(model.K, model.S, rule=ChargingOptions)


# Each EV assigned to <= 1 charger
def SingleChargerAssignment(model, k, s):
    return sum(model.assign[k, i, s] for i in model.I) <= 1


model.ConSingleAssign = pyo.Constraint(model.K, model.S, rule=SingleChargerAssignment)


def x_zero(model, k, i, s, t):
    if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
        return model.x[k, i, s, t] == 0
    else:
        return pyo.Constraint.Skip  # nothing to enforce outside the relevant time range


model.Conx_zero = pyo.Constraint(model.x_indices, rule=x_zero)


# def ChargerSingleEV_(model, k, i, s, t):
#    if EVdata['AT'][k] <= t <= EVdata['DT'][k]:
#       return sum(model.x[k, i, s, t] for k in model.K) <= 1
#    else:
#        return pyo.Constraint.Skip
# model.ConChargerSingleEV_ = pyo.Constraint(model.x_indices, rule=ChargerSingleEV_)


def ChargerSingleEV_(model, i, s, t):
    """Ensures each charger (i,s) is used by at most one EV at time t"""
    # Find all EVs that could use this charger at this time
    relevant_evs = [
        k for k in model.K
        if EVdata['ParkingNo'][k - 1] == s and  # EV is at this parking
           EVdata['AT'][k] <= t <= EVdata['DT'][k]  # EV is present at time t
    ]

    if not relevant_evs:
        # No EVs could use this charger at this time
        return pyo.Constraint.Feasible

    # Sum over x variables that actually exist
    sum_x = sum(model.x[k, i, s, t] for k in relevant_evs if (k, i, s, t) in model.x)
    return sum_x <= 1


# Declare the constraint over all possible (i,s,t) combinations
model.ConChargerSingleEV_ = pyo.Constraint(
    model.I, model.S, model.T,
    rule=ChargerSingleEV_
)


def Link_x_charge(model, k, i, s, t):
    return model.x[k, i, s, t] <= model.assign[k, i, s]


model.ConLink_x_charge = pyo.Constraint(model.x_indices, rule=Link_x_charge)


# Constraint: If an EV is assigned to a fixed charger, the charger is occupied from AT to DT
def charger_occupancy(model, k, i, s):
    """Ensures charger i at parking s is occupied from AT to DT if EV k is assigned"""
    # Only apply if EV is at this parking and AT <= DT
    if EVdata['ParkingNo'][k - 1] != s or EVdata['AT'][k] > EVdata['DT'][k]:
        return pyo.Constraint.Skip

    # Calculate required occupancy duration
    duration = EVdata['DT'][k] - EVdata['AT'][k] + 1

    # Sum only existing x variables (sparse-aware)
    occupied_sum = sum(
        model.x[k, i, s, t]
        for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
        if (k, i, s, t) in model.x  # Check if variable exists
    )

    # Big-M constraint reformulated for better numerical stability
    return occupied_sum >= duration * model.assign[k, i, s] - M * (1 - model.assign[k, i, s])


model.ConChargerOccupancy = pyo.Constraint(
    model.K, model.I, model.S,
    rule=charger_occupancy
)


# Constraint: No other EV can be assigned to the same charger during an occupied period
def no_overlapping_assignments(model, k1, k2, i, s):
    """Prevents EV k2 from being assigned to charger i at parking s if it overlaps with EV k1's stay"""
    if k1 != k2 and EVdata['ParkingNo'][k1 - 1] == s and EVdata['ParkingNo'][k2 - 1] == s:
        # Check if time windows overlap
        overlap_condition = (EVdata['AT'][k1] <= EVdata['DT'][k2]) and (EVdata['AT'][k2] <= EVdata['DT'][k1])
        if overlap_condition:
            return model.assign[k1, i, s] + model.assign[k2, i, s] <= 1
    return pyo.Constraint.Skip


model.ConNoOverlappingAssignments = pyo.Constraint(model.K, model.K, model.I, model.S, rule=no_overlapping_assignments)


###############  Robot assignment constraints
def robot_single_connection(model, k, j, s, t):
    """Ensures each robot (j,s) charges at most one EV at any time t"""
    # Since we're using y_indices, we know (k,j,s,t) is valid
    # We need to sum over all EVs that could be served by this robot at this time
    return sum(model.y[k, j, s, t] for k in model.K
               if (k, j, s, t) in model.y_indices) <= NevSame


model.ConRobotSingleConnection = pyo.Constraint(model.y_indices, rule=robot_single_connection)

# Create a mapping from (k,s,t) to possible robots
# First create proper index sets
model.ev_active_indices = pyo.Set(
    dimen=3,
    initialize=lambda model: [
        (k, EVdata['ParkingNo'][k - 1], t)
        for k in model.K
        for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
    ]
)


def single_robot_connection(model, k, s, t):
    """Ensures each EV is connected to ≤1 robot at any time"""
    # Get all robots at this parking location
    available_robots = [j for j in model.J
                        if (k, j, s, t) in model.y_indices]

    return sum(model.y[k, j, s, t] for j in available_robots) <= 1


model.ConSingleRobotConnection = pyo.Constraint(
    model.ev_active_indices,
    rule=single_robot_connection
)


def Link_y_charge(model, k, j, s, t):
    return model.y[k, j, s, t] <= model.assignRobot[k, j, s]


model.ConLink_y_charge = pyo.Constraint(model.y_indices, rule=Link_y_charge)


########## charging constraints###################

def Charging_toEV(model, k, t):
    robot_contrib = sum(
        model.P_dch_rob[k,j,s,t] 
        for j in model.J 
        for s in model.S 
        if (k,j,s,t) in model.y_indices
    )
    return model.P_ch_EV[k,t] == sum(model.P_b_EV[k,i,s,t] for i in model.I for s in model.S) + robot_contrib

model.ConCharging_toEV = pyo.Constraint(model.K, model.T, rule=Charging_toEV)


def Charging_UpLimit(model, k, i,s, t):
    return model.P_b_EV[k,i,s,t]  <= ChargerCap


model.ConCharging_UpLimit = pyo.Constraint(model.K, model.I, model.S, model.T, rule=Charging_UpLimit)


def SOC_EV_f1(model, k, t):
    if t < EVdata['AT'][k]:
        return model.SOC_EV[k, t] == 0
    elif t == EVdata['AT'][k]:
        return model.SOC_EV[k, t] == EVdata['SOCin'][k] * EVdata['EVcap'][k]
    elif t > EVdata['AT'][k] and t <= EVdata['DT'][k]:
        return model.SOC_EV[k, t] == model.SOC_EV[k, t - 1] + (1 / SampPerH) * model.P_ch_EV[k, t]
    else:
        return pyo.Constraint.Skip  # nothing to enforce outside the relevant time range


def SOC_EV_f2(model, k, t):
    if t == EVdata['DT'][k]:
        return model.SOC_EV[k, t] == 1 * EVdata['SOCout'][k] * EVdata['EVcap'][k]
    else:
        return pyo.Constraint.Skip  # nothing to enforce outside the relevant time range


model.ConSOC_EV_f1 = pyo.Constraint(model.K, model.T, rule=SOC_EV_f1)
model.ConSOC_EV_f2 = pyo.Constraint(model.K, model.T, rule=SOC_EV_f2)


def SOC_Charge_limit1(model, k, i, s, t):
    return model.P_b_EV[k, i, s, t] <= EVdata['EVcap'][k] * model.assign[k, i, s]


def SOC_Charge_limit2(model, k, i, s, t):
    return model.P_b_EV[k, i, s, t] <= EVdata['EVcap'][k] * model.x[k, i, s, t]


model.ConSOC_Charge_limit1 = pyo.Constraint(model.K, model.I, model.S, model.T, rule=SOC_Charge_limit1)
model.ConSOC_Charge_limit2 = pyo.Constraint(model.x_indices, rule=SOC_Charge_limit2)


def SOC_Limit(model, k, t):
    return model.SOC_EV[k, t] <= EVdata['EVcap'][k]


model.ConSOC_Limit = pyo.Constraint(model.K, model.T, rule=SOC_Limit)


############## Robot charging constraints #####################


def Ch_robot_limit(model, j, s, t):
    return model.P_ch_rob[j, s, t] <= model.u_rob[j, s, t] *  (NevSame)*DCchargerCap
model.ConCh_robot = pyo.Constraint(model.J, model.S, model.T, rule=Ch_robot_limit)


def Dch_robot_limit1(model, j, s, t):
    active_discharges = sum(
        model.P_dch_rob[k,j,s,t] 
        for k in model.K 
        if (k,j,s,t) in model.y_indices  # Only sum existing variables
    )
    return active_discharges <= (1-model.u_rob[j,s,t])*NevSame*DCchargerCap
model.ConDch_robot1 = pyo.Constraint(model.J, model.S, model.T, rule=Dch_robot_limit1)




def Dch_robot_limit2(model, k, j, s, t):
        return model.P_dch_rob[k, j, s, t] <= 1*DCchargerCap * model.y[k, j, s, t]
model.Dch_robot_limit2 = pyo.Constraint(model.y_indices, rule=Dch_robot_limit2)



#def Dch_robot_zero(model, k, j, s, t):
#    """Force zero discharge for non-y_indices cases"""
#    return model.P_dch_rob[k, j, s, t] == 0
#model.Dch_robot_zero = pyo.Constraint(
#    model.K * model.J * model.S * model.T - model.y_indices,  # All indices NOT in y_indices
#    rule=Dch_robot_zero
#)



def Dch_robot_limit3(model, k, j, s, t):
    return model.P_dch_rob[k, j, s, t] <= sum(model.CapRobot[j, s, kk] for kk in model.KK)
model.Dch_robot_limit3 = pyo.Constraint(model.y_indices, rule=Dch_robot_limit3)


def SOC_Robot(model, j, s, t):
    if t == 1:
        return model.SOC_rob[j,s,t] == 0.2*sum(model.CapRobot[j,s,kk] for kk in model.KK)
    else:
        active_discharges = sum(
            model.P_dch_rob[k,j,s,t] 
            for k in model.K 
            if (k,j,s,t) in model.y_indices
        )
        return model.SOC_rob[j,s,t] == model.SOC_rob[j,s,t-1] + (1/SampPerH)*model.P_ch_rob[j,s,t] - (1/SampPerH)*active_discharges
model.ConSOC_Robot = pyo.Constraint(model.J, model.S, model.T, rule=SOC_Robot)


def SOC_Robot2(model, j, s, t):
    return model.SOC_rob[j, s, t] >= 0.2 * sum(model.CapRobot[j, s, kk] for kk in model.KK)


model.ConSOC_Robot2 = pyo.Constraint(model.J, model.S, model.T, rule=SOC_Robot2)


def SOC_Robot_limit(model, j, s, t):
    return model.SOC_rob[j, s, t] <= sum(model.CapRobot[j, s, kk] for kk in model.KK)


model.ConSOC_Robot_limit = pyo.Constraint(model.J, model.S, model.T, rule=SOC_Robot_limit)


# def Ch_Robot_limit2(model, j, s, t):
#    return model.P_ch_rob[j, s, t] <= 0.5*sum(model.CapRobot[j, s, kk] for kk in model.KK)
# model.ConCh_Robot_limit2 = pyo.Constraint(model.J, model.S, model.T, rule= Ch_Robot_limit2 )


def Ch_Robot_limit3(model, j, s, t):
    return sum(model.P_dch_rob[k, j, s, t] for k in model.K) <= (NevSame)*DCchargerCap


#model.ConCh_Robot_limit3 = pyo.Constraint(model.J, model.S, model.T, rule= Ch_Robot_limit3 )

def RobotCapLimit4(model, j, s, kk):
    return model.CapRobot[j, s, kk] <= RobotTypes[kk - 1] * model.u_rob_type[j, s, kk]


model.ConRobotCapLimit4 = pyo.Constraint(model.J, model.S, model.KK, rule=RobotCapLimit4)


def RobotCapLimit2(model, j, s, kk):
    return sum(model.u_rob_type[j, s, kk] for kk in model.KK) <= 1


model.ConRobotCapLimit2 = pyo.Constraint(model.J, model.S, model.KK, rule=RobotCapLimit2)



def RobotTotNum(model, j, s, kk):
    return sum(model.u_rob_type[j, s, kk] for kk in model.KK for j in model.J) <= MaxRobot
#model.ConRobotTotNum = pyo.Constraint(model.J, model.S, model.KK, rule=RobotTotNum)

def Dch_rob_zero(model, k, j, s, t):
    if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
        return model.P_dch_rob[k, j, s, t] == 0
    else:
        return pyo.Constraint.Skip  # nothing to enforce outside the relevant time range


#model.ConDch_rob_zero = pyo.Constraint(model.K, model.J, model.S, model.T, rule=Dch_rob_zero)



def Robot_Parking_Assignment(model, j):
    """Each robot must be assigned to exactly one parking"""
    return sum(model.robot_parking_assignment[j,s] for s in model.S) <= 1  # <=1 allows unassigned robots
#model.ConRobotParkingAssignment = pyo.Constraint(model.J, rule=Robot_Parking_Assignment)


def Robot_Usage_Constraint(model, k, j, s, t):
    """Robot can only be used in its assigned parking"""
    return model.y[k,j,s,t] <= model.robot_parking_assignment[j,s]
#model.ConRobotUsage = pyo.Constraint(model.y_indices, rule=Robot_Usage_Constraint)


def Robot_Usage_Constraint2(model, j, s, kk):
    """Robot can only be used in its assigned parking"""
    return model.u_rob_type[j, s, kk] <= model.robot_parking_assignment[j,s]
#model.ConRobotUsage2 = pyo.Constraint(model.J, model.S, model.KK, rule=Robot_Usage_Constraint2)
##################### PARKING CONSTRAINTS ##################




def PeakPowerConstraint(model, s, t):
    return model.PeakPower[s] >= model.P_btot[s, t]


model.ConPeakPower = pyo.Constraint(model.S, model.T, rule=PeakPowerConstraint)



def AlphaFun(model):
    return model.Alpha >= -100

model.ConAlphaFun = pyo.Constraint(rule=AlphaFun)


######################### TO CONSIDER BESS ##################
def NoCharger3(model,s):
    return  model.Ns[s] >= max_overlaps[s]
#model.ConNoCharger3 = pyo.Constraint(model.S, rule = NoCharger3)

############## OBJECTIVE FUNCTION ##################

model.obj = pyo.Objective(expr=(1) * (PFV_Charger * sum(model.Ns[s] * Ch_cost for s in model.S)) +
                               (1) * PFV_Rob * sum(
    model.u_rob_type[j, s, kk] * robotCC[kk - 1] for j in model.J for s in model.S for kk in model.KK) +
                               sum(Price.iloc[t - 1] * (0.001) * model.P_btot[s, t] for s in model.S for t in model.T) +
                               (1 / 30) * PeakPrice * sum(model.PeakPower[s] for s in model.S)
                                + 1*model.Alpha, sense=pyo.minimize)

#####################################################

#####################


# model.cuts = pyo.Constraint(model.Nodes, rule=voltage_feasibility_cut)
model.cuts = pyo.ConstraintList()  # THIS IS CRUCIAL

solver = pyo.SolverFactory('gurobi')
#solver.options['MIPGap'] = 0.01  # Allow 5% relative gap
#solver.options['cuts'] = 2  # More aggressive cut generation

solver.options = {
    'Presolve': 2,          # Aggressive presolve
    'MIPGap': 0.05,         # Accept 5% gap early
    'Heuristics': 0.8,      # More time on heuristics
    'Cuts': 2,              # Maximum cut generation
    'Threads': 8,           # Full CPU utilization
    'NodeMethod': 2,        # Strong branching
#    'SolutionLimit': 1      # Stop at first feasible (if applicable)
}
# (3) Update Master Problem Iteratively

# Verify all constraints are necessary
#print(f"Total constraints: {len(list(model.component_objects(pyo.Constraint)))}")

# Check binary variable count
#print(f"Binary variables: {sum(v.is_binary() for v in model.component_objects(pyo.Var))}")

def compute_alpha(duals, parking_to_bus):
    """Convert duals to sensitivity matrix alpha[b, s]."""
    alpha = {}
    for (n, t), dual_val in duals.items():
        for s, bus in parking_to_bus.items():
            if bus == n:  # Parking s injects power at bus n
                alpha[(n, s)] = dual_val  # ∂V_n/∂P_s
    return alpha
kl = 0


for s in model.S:
    for t in model.T:
        model.P_btotBar[s, t].value = 0.0  # Or PgridMax / len(model.S) for a warm start


'''

# Precompute sensitivity matrix once (outside the loop)
#voltage_sensitivity_matrix  = compute_voltage_sensitivity(line_data, num_buses=33)

while True:
    kl += 1
    print(f"Iteration {kl}")

    # Solve master problem
    results = solver.solve(model, tee=True)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        print("Master problem infeasible. Terminating.")
        break

    # Get current solution values
    P_btot_current =  {
        (p, t): pyo.value(model.P_btot[s, t])
        for s in model.S
        for p in [s]  # Assuming model.S indices match parking_to_node keys
        if p in parking_to_bus
        for t in model.T}

    # Call PowerFlow subproblem
    [min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj] = PowerFlow(P_btot_current, Pattern, Price)

    # Termination check
    # Check violations and add cuts
#    if all(v >= Vmin for v in volt_per_node): 
#        break  # Converged


    for b in model.Nodes:
        for t in model.T:
            if volt_per_node[b-1, t-1] < Vmin:
                print(f'volt_per_node{b-1, t-1}' )
                print(f"AlphaSensi[{b},{parking_to_bus[s]}] = {AlphaSensi[b, parking_to_bus[s]]:.6f}")
#                print(f"P_btot_current{}")
                model.cuts.add(
                    sum(
                        AlphaSensi[b, parking_to_bus[s]] * (10000000)* (model.P_btot[s, t] - 0*model.P_btotBar[s, t])
#                        -10000000*model.P_btot[s, t]
                        for s in model.S 
                        if s in parking_to_bus  # Only consider defined parkings
                    ) >= (Vmin - 1)
                )
                print(f"Total cuts after iteration {kl}: {len(model.cuts)}")
#                print(f' AlphaSensi{b, s}' )
                       
    if not any(v < Vmin for row in volt_per_node for v in row):
        break

    # Update for next iteration
    for s in model.S:
        for t in model.T:
            model.P_btotBar[s, t].value = pyo.value(model.P_btot[s, t])

    # Update lower bound
    current_alpha = pyo.value(model.Alpha)
    if current_alpha > pyo.value(model.AlphaDown):
        model.AlphaDown.value = current_alpha

    print(f"Iter {kl}: Alpha={current_alpha:.2f}, Min V={np.min(volt_per_node):.4f}")

'''

while True:
    kl += 1
    print(f"Iteration {kl}")

    # Solve master problem
    results = solver.solve(model, tee=True)
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        print("Master problem infeasible. Terminating.")
        break

    # Get current solution values
    P_btot_current =  {
        (p, t): pyo.value(model.P_btot[s, t])
        for s in model.S
        for p in [s]  # Assuming model.S indices match parking_to_node keys
        if p in parking_to_bus
        for t in model.T}
    # Call PowerFlow subproblem
  #  [_, volt_per_node, duals2, duals_balance_p, duals_Dev, SPobj] = PowerFlow(P_btot_current, Pattern, Price)
    [min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj] = PowerFlow(
        P_btot_current, Pattern, Price)

    # Termination check (using volt_per_node)
    if np.all(volt_per_node >= Vmin - 1e-3):
        print("Voltage constraints satisfied. Optimal solution found.")
        break

    # Generate cuts for violated voltages
    for t in model.T:
        worst_bus = np.argmin(volt_per_node[:, t-1]) + 1  # +1 for 1-based indexing


        if volt_per_node[worst_bus-1, t-1] < Vmin - 1e-3:
            min_volt = np.min(volt_per_node)
            violation = max(0, Vmin - min_volt)
            print(f"t={t}, worst_bus={worst_bus}, dual={duals_DevLow.get((worst_bus,t),0):.4f}, dual={duals_DevUp.get((worst_bus,t),0):.4f}")
            print(f"t={t}, worst_bus={worst_bus}, dual={duals_Dev.get((worst_bus,t),0):.4f}")
            relevant_parkings = [s for s in model.S if parking_to_bus.get(s, -1) == worst_bus]
            if relevant_parkings:
                changes = {
                    s: sum(
                        abs(pyo.value(model.P_btot[s,t]) - pyo.value(model.P_btotBar[s,t]))
                        for t in model.T
                    )
                    for s in model.S
                }
                print("Power changes:", changes)
                model.cuts.add(
                    SPobj + sum(
                        -(duals_DevLow.get((worst_bus, t), 0.0)  + duals_balance_p.get((worst_bus, t), 0.0) + duals_DevUp.get((worst_bus, t), 0.0) )* 100 *violation* (
                            (model.P_btot[s, t] / sb) - 0*model.P_btotBar[s, t]  # Uses previous iteration's P_btotBar
                        )
                        for s in relevant_parkings
                    ) <= model.Alpha
                )
                print("Sub problem objective = ",SPobj)

    # Update P_btotBar for NEXT iteration (AFTER cuts are generated)
    for s in model.S:
        for t in model.T:
            model.P_btotBar[s, t].value = pyo.value(model.P_btot[s, t]) / sb

    # Update lower bound
    if pyo.value(model.Alpha) > pyo.value(model.AlphaDown):
        model.AlphaDown.value = pyo.value(model.Alpha)

    print(f"Alpha={pyo.value(model.Alpha)}, Min Voltage={np.min(volt_per_node)}")

'''


while True:
    kl = kl + 1
              
    # Solve master problem
    solver.solve(model)

  
    
    P_btot_current =  {
        (p, t): pyo.value(model.P_btot[s, t])
        for s in model.S
        for p in [s]  # Assuming model.S indices match parking_to_node keys
        if p in parking_to_bus
        for t in model.T
    }
#    for s in model.S:
#        for t in model.T:
#            model.P_btot[s, t] = 1 * model.P_btot[s, t]

    # power flow
    [min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj] = PowerFlow(
        P_btot_current,
        Pattern, Price
    )
    min_volt = np.min(volt_per_node)
    violation = max(0, Vmin - min_volt) 
    if all(v >= Vmin for v in voltage_at_bus):
        break  # Converged

    for b in model.Nodes:
        for t in model.T:
            if volt_per_node[b - 1, t - 1] < Vmin - 1e-3:
                # First check if there are any parking stations at this bus
                relevant_parkings = [s for s in model.S if parking_to_bus.get(s, -1) == b]

                if not relevant_parkings:  # No parking stations at this bus
                    print(f"Warning: No parking stations at bus {b} (time {t})")
                    continue  # Skip this constraint
                changes = {
                    s: sum(
                        abs(pyo.value(model.P_btot[s,t]) - pyo.value(model.P_btotBar[s,t]))
                        for t in model.T
                    )
                    for s in model.S
                }
                print("Power changes:", changes)
                model.cuts.add(
                    SPobj +  50*violation + sum(
                        (duals_Dev.get((b, t), 0.0) + duals_balance_p.get((b,t),0.0) ) * (0.001) * ((model.P_btot[s, t] / sb) - 1*model.P_btotBar[s, t])
                        for s in relevant_parkings
                    ) <= model.Alpha
                )
                print("Sub problem objective = ",SPobj)
    # Terminate if all voltages are feasible (with tolerance)

    # Update P_btotBar for NEXT iteration (AFTER cuts are generated)
    for s in model.S:
        for t in model.T:
            model.P_btotBar[s, t].value = pyo.value(model.P_btot[s, t]) / sb

    # Update lower bound
    if pyo.value(model.Alpha) > pyo.value(model.AlphaDown):
        model.AlphaDown.value = pyo.value(model.Alpha)

    print(f"Alpha={pyo.value(model.Alpha)}, Min Voltage={np.min(volt_per_node)}")
    # Terminate if all voltages are feasible (with tolerance)
    if all(v >= Vmin - 1e-3 for row in volt_per_node for v in row):
        break
 '''

Ncharger_val = {(s): pyo.value(model.Ns[s]) for s in model.S}
print(Ncharger_val)
# x_val = {(k,i,t): sum( pyo.value(model.x[k,i,s,t]) for s in model.S) for k in model.K for i in model.I for t in model.T}
assign_val = {(k, i): sum(pyo.value(model.assign[k, i, s]) for s in model.S) for k in model.K for i in model.I}
P_ch_rob_val = {(j, t): sum(pyo.value(model.P_ch_rob[j, s, t]) for s in model.S) for j in model.J for t in model.T}
P_Dch_rob_val = {(j, t): sum(pyo.value(model.P_ch_rob[j, s, t]) for s in model.S) for j in model.J for t in model.T}

CapRobot_val = {(j, s, kk): pyo.value(model.CapRobot[j, s, kk]) * pyo.value(model.u_rob_type[j, s, kk]) for j in model.J
                for s in model.S for kk in model.KK}
CapRobot_val2 = {(j, s, kk): RobotTypes[kk - 1] * pyo.value(model.u_rob_type[j, s, kk]) for j in model.J for s in
                 model.S for kk in model.KK}

Ns_val = {(s): pyo.value(model.Ns[s]) for s in model.S}

print("Non-zero chargers:")
for s, val in Ns_val.items():
    if val != 0:  # Check if the value is not zero
        print(f"Chargers in parking {s}: {val}")

print("Non-zero robot capacities:")
for (j, s, kk), val in CapRobot_val.items():
    if val != 0:
        print(f"Robot {j}, parking {s}, Type {kk}: {val}")

# Print total number of robots (sum of non-zero capacities)
total_robots = sum(val for val in CapRobot_val.values() if val != 0)
print(f"\nTotal number of robots (non-zero): {total_robots}")

for (j, s, kk), val in CapRobot_val2.items():
    if val != 0:
        print(f"Robot {j}, parking {s}, Type {kk}: {val}")


                    
                    

                    
#print(sum(pyo.value(model.P_dch_rob[k, j, s, t]) for (k,j,s,t) in model.y_indices))


# Prepare data
# Create figure with 2 rows and 2 columns
# Store parking IDs and energy values
parkings = []
grid_energy = []
robot_energy = []

# First, collect the data
for s in model.S:
    grid = sum(pyo.value(model.P_b_EV[k, i, s, t]) for k in model.K for i in model.I for t in model.T)
    robot = sum(pyo.value(model.P_dch_rob[k, j, s, t]) for (k, j, s_t, t) in model.y_indices if s_t == s)

    parkings.append(str(s))  # Make sure parking names are strings
    grid_energy.append(grid)
    robot_energy.append(robot)

# Calculate total energy per parking for percentages
total_energy = [g + r for g, r in zip(grid_energy, robot_energy)]
grid_pct = [100 * g / tot if tot > 0 else 0 for g, tot in zip(grid_energy, total_energy)]
robot_pct = [100 * r / tot if tot > 0 else 0 for r, tot in zip(robot_energy, total_energy)]

# Plot grouped bar chart
x = np.arange(len(parkings))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, grid_energy, width, label='FC', color = 'black')
bars2 = ax.bar(x + width/2, robot_energy, width, label='MCR', color = 'orange')

# Add percentage text on top of each bar
for i in range(len(parkings)):
    ax.text(x[i] - width/2, grid_energy[i] + 1, f'{grid_pct[i]:.1f}%', ha='center', va='bottom', fontsize=14)
    ax.text(x[i] + width/2, robot_energy[i] + 1, f'{robot_pct[i]:.1f}%', ha='center', va='bottom', fontsize=14)

# Labels and title
ax.set_ylabel('Total Energy (kWh)', fontsize=14)  # Increased label font size
ax.set_xlabel('Parking lot', fontsize=14)        # Increased label font size

# Set tick parameters for both axes
ax.tick_params(axis='both', which='major', labelsize=14)  # This increases both x and y tick sizes

#ax.set_title('Energy Source Comparison Across Parkings')
ax.set_xticks(x)
ax.set_xticklabels([f'Parking {i+1}' for i in range(len(parkings))])
ax.legend(fontsize = 14)
plt.tight_layout()
plt.savefig("Combined_Energy_Source_Comparison.png", dpi=600)
plt.show()

#########################################
for s in model.S:
    # First check if there's any discharge activity in this parking
    has_discharge = any(
        pyo.value(model.P_dch_rob[k, j, s, t]) > 0.1  # small threshold to account for numerical precision
        for (k, j, parking_no, t) in model.y_indices
        if parking_no == s
    )
    
    if not has_discharge:
        continue

    # Find active robots in this parking
    active_robots = set()
    for (k, j, parking_no, t) in model.y_indices:
        if parking_no == s and pyo.value(model.P_dch_rob[k, j, s, t]) > 0.1:
            active_robots.add(j)

    if not active_robots:
        continue

    # Plot each active robot
    fig, axes = plt.subplots(len(active_robots), 1, figsize=(12, 2*len(active_robots)), squeeze=False)
    
    for idx, j in enumerate(active_robots):
        ax = axes[idx, 0]
        robot_capacity = sum(pyo.value(model.CapRobot[j, s, kk]) for kk in model.KK)
        
        # Get SOC data
        soc = [pyo.value(model.SOC_rob[j, s, t]) for t in model.T]
        ax.plot(model.T, soc, 'b-', label='SOC (kWh)')
        
        # Get charging data
        charge = [pyo.value(model.P_ch_rob[j, s, t]) for t in model.T]
        
        # Get discharge data - properly aggregated by time
        discharge = [0.0]*len(model.T)
        for (k, j_rob, parking_no, t) in model.y_indices:
            if j_rob == j and parking_no == s:
                discharge[t-1] += pyo.value(model.P_dch_rob[k, j, s, t])  # t-1 because Python is 0-indexed
        
        # Create stacked bars
        ax.bar(model.T, charge, color='g', alpha=0.3, label='Charging')
        ax.bar(model.T, discharge, bottom=charge, color='r', alpha=0.3, label='Discharging')
        
        ax.set_title(f'Robot {j} in Parking {s}')
#        ax.set_title(f'Robot {j} in Parking {s} (Capacity: {robot_capacity:.1f} kWh)')

        ax.grid(True)
        if idx == len(active_robots)-1:
            ax.legend()
            ax.set_xlabel('Time period')
        ax.set_ylabel('Power (kW)')
    
    plt.tight_layout()
    plt.savefig(f'Robot_Discharge_Parking_{s}.png', dpi=300)
    plt.tight_layout()
    plt.savefig(f"SOC_Robot_{s}.png", dpi=600)

# Prepare SOC matrix HEAT MAP
soc_matrix = [[pyo.value(model.SOC_EV[k, t]) / EVdata.loc[k - 1, 'EVcap'] * 100
               for t in model.T]
              for k in model.K]

plt.figure(figsize=(12, 8))
sns.heatmap(soc_matrix, annot=False, fmt='.0f', cmap='YlGnBu',
            xticklabels=model.T)
# , yticklabels=range(1,model.nEV+1)
plt.xlabel('Time period')
plt.ylabel('EV ID')
plt.title('EV SOC Heatmap (%)')
plt.tight_layout()
plt.savefig("HeatMapEVs.png", dpi=600)

soc_df = pd.DataFrame({k: [pyo.value(model.SOC_EV[k, t]) for t in model.T]
                       for k in model.K})



time = range(1, SampPerH * 24 + 1)  # 1-24 hours

for s in model.S:
    plt.figure(figsize=(12, 8))  # Larger figure for better readability
    
    # --- Grid Charging Plot ---
    plt.subplot(2, 1, 1)
    P_b_EV_total = [
        sum(
            model.P_b_EV[k, i, s, t].value 
            for k in model.K 
            for i in model.I 
            if model.P_b_EV[k, i, s, t].value is not None
        )
        for t in model.T
    ]
    
    plt.plot(time, P_b_EV_total, 'b-', linewidth=2, label='Grid Charging')
    plt.fill_between(time, P_b_EV_total, color='blue', alpha=0.1)
    plt.xlabel('Time period', fontsize = 14)
    plt.ylabel('Power (kW)', fontsize = 14)
    plt.tick_params(axis='both', labelsize=14)
    plt.title(f'Parking {s} - Grid Charging Power')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    # --- Robot Discharging Plot (Sparse-Aware) ---
    plt.subplot(2, 1, 2)
    P_dch_rob_total = [
        sum(
            model.P_dch_rob[k, j, s, t].value 
            for k in model.K 
            for j in model.J 
            if (k,j,s,t) in model.y_indices and model.P_dch_rob[k,j,s,t].value is not None
        )
        for t in model.T
    ]
    
    plt.plot(time, P_dch_rob_total, 'r-', linewidth=2, label='Robot Discharging')
    plt.fill_between(time, P_dch_rob_total, color='red', alpha=0.1)
    plt.xlabel('Time period', fontsize = 14)
    plt.ylabel('Power (kW)', fontsize = 14)
    plt.tick_params(axis='both', labelsize=14)
    plt.title(f'Parking {s} - Robot Discharging Power')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Parking_{s}_Power_Profile.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print(f"\nParking {s} Power Summary:")
    print(f"Max Grid Charging: {max(P_b_EV_total):.2f} kW at hour {np.argmax(P_b_EV_total)+1}")
    print(f"Total Robot Discharge: {sum(P_dch_rob_total):.2f} kWh")
plt.figure(figsize=(10, 5))
time = range(1, SampPerH * 24 + 1)
plt.plot(time, 10*Price, color = 'red', marker = '*')
plt.xlabel('Time period', fontsize = 14)
plt.ylabel('Price (SEK/MWh)', fontsize = 14)
#plt.title('Electricity price')
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.savefig('InputPrice.png', dpi=600)




plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
time = range(1, SampPerH * 24 + 1)
plt.plot(time, [sum(model.P_btot[s, t].value for s in model.S) for t in model.T])
plt.xlabel('Time period')
plt.ylabel('Power (kW)')
plt.title('Overall purchased electricity')
plt.grid(True)

plt.subplot(2, 1, 2)
time = range(1, SampPerH * 24)
time_array = np.array(time)  # Convert to NumPy array
time_transposed = np.transpose(time_array)
for s in model.S:
    P_Tot_Purch = [model.P_btot[s, t].value for t in model.T]
    plt.plot(P_Tot_Purch, label=f'Parking {s}')
plt.xlabel('Time period')
plt.ylabel('Power (kW)')
plt.title('Purchased electricity with respect to the parking')
plt.grid(True)
plt.legend()  # Show legend with labels
plt.tight_layout()
plt.savefig('PurchasePower.png', dpi=600)

plt.show()

# Create DataFrames for x and y variables
x_data = []
for index in model.x_indices:  # Only iterate over existing indices
    k, i, s, t = index  # Unpack the index
    x_data.append({
        'EV': k,
        'Charger': i,
        'Parking': s,
        'Time': t,
        'Value': pyo.value(model.x[k, i, s, t])
    })

y_data = []
# Only iterate over existing y_indices
for index in model.y_indices:  # (k,j,s,t) tuples that actually exist
    k, j, s, t = index
    y_data.append({
        'EV': k,
        'Robot': j,
        'Parking': s,
        'Time': t,
        'Value': pyo.value(model.y[k, j, s, t])
    })

# Convert to DataFrames
df_x = pd.DataFrame(x_data)
df_y = pd.DataFrame(y_data)

# Create Excel writer object
with pd.ExcelWriter('charging_schedule_MultipleParkings.xlsx') as writer:
    # Write x variables (station charging)
    df_x.to_excel(writer, sheet_name='Station_Charging', index=False)

    # Write y variables (robot charging)
    df_y.to_excel(writer, sheet_name='Robot_Charging', index=False)

    # Add summary sheets
    df_x.groupby(['Charger', 'Time'])['Value'].sum().unstack().to_excel(
        writer, sheet_name='Charger_Utilization')

    df_y.groupby(['Robot', 'Time'])['Value'].sum().unstack().to_excel(
        writer, sheet_name='Robot_Utilization')

print("Successfully exported to charging_schedule_MultipleParkings.xlsx")

TotalChargerCost = sum(Ch_cost * pyo.value(model.Ns[s]) for s in model.S)
print("Total Charger Cost = ", TotalChargerCost)

TotalPurchaseCost = sum(Price.iloc[t - 1] * (0.001) * pyo.value(model.P_btot[s, t]) for s in model.S for t in model.T)
print("Total cost of electricity purchased  = ", TotalPurchaseCost)

TotalPeakCost = (1 / 30) * PeakPrice * sum(pyo.value(model.PeakPower[s]) for s in model.S)
print("Total cost of Peak Power  = ", TotalPeakCost)

TotalMCRcost = sum(
    pyo.value(model.u_rob_type[j, s, kk]) * robotCC[kk - 1] for j in model.J for s in model.S for kk in model.KK)
print("Total cost of MCRs = ", TotalMCRcost)

ObjValue = pyo.value(model.obj)
print("Objective function = ", ObjValue)




# Create a pivot table for charger utilization
charger_utilization = df_x.pivot_table(index='Time', 
                                      columns='Charger', 
                                      values='Value',
                                      aggfunc='sum',
                                      fill_value=0)

# Plot settings
plt.figure(figsize=(15, 8))

# Create heatmap
sns.heatmap(charger_utilization.T, 
            cmap=['white', 'green'],  # White=unused, Green=used
            linewidths=0.5,
            linecolor='lightgray',
            cbar=False)

# Customize plot
plt.title('Charger Utilization Over Time', fontsize=16, pad=20)
plt.xlabel('Time Period', fontsize=14)
plt.ylabel('Charger ID', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()

# Add annotation for better understanding
plt.text(0, -1.5, 
         'Green cells indicate when a charger is being used by an EV',
         fontsize=10, 
         ha='left')

plt.savefig('charger_utilization_heatmap.png', dpi=300, bbox_inches='tight')





# Group data by parking
parkings = sorted(df_x['Parking'].unique())  # Ensure parkings are in order (1, 2, 3)

for parking in parkings:
    # Filter data for the current parking
    df_parking = df_x[df_x['Parking'] == parking].copy()
    
    # Get unique charger IDs in this parking and sort them
    original_charger_ids = sorted(df_parking['Charger'].unique())
    
    # Map original charger IDs to sequential numbers (1, 2, 3...)
    charger_id_mapping = {orig_id: new_id + 1 
                          for new_id, orig_id in enumerate(original_charger_ids)}
    df_parking['Charger_Sequential'] = df_parking['Charger'].map(charger_id_mapping)
    
    # Pivot table with sequential charger IDs
    charger_utilization = df_parking.pivot_table(
        index='Time', 
        columns='Charger_Sequential', 
        values='Value',
        aggfunc='sum',
        fill_value=0
    )
    
    # Remove completely inactive chargers (if any)
    active_chargers = charger_utilization.columns[(charger_utilization > 0).any()]
    charger_utilization = charger_utilization[active_chargers]
    
    # Skip if no active chargers
    if charger_utilization.empty:
        print(f"No active chargers in Parking {parking}. Skipping.")
        continue
    
    # Create complete time index starting from 1
    min_time = charger_utilization.index.min()
    max_time = charger_utilization.index.max()
    full_index = range(1, max_time + 1)  # Force start at 1
    
    # Reindex to ensure all time periods are shown
    charger_utilization = charger_utilization.reindex(full_index, fill_value=0)
    
    # Plot heatmap
    plt.figure(figsize=(15, 6))
    ax = sns.heatmap(
        charger_utilization.T,
        cmap=['white', 'green'],
        linewidths=0.5,
        linecolor='lightgray',
        cbar=False
    )
    
    # Set x-axis labels to show every nth time period for clarity
    n = max(1, len(full_index) // 10)  # Show ~10 labels
    xticks = [i for i in full_index if i % n == 1 or i == full_index[-1]]
    ax.set_xticks([x - 1.5 for x in xticks])  # Adjust for heatmap bin centers
    ax.set_xticklabels(xticks)
    
    plt.title(f'Parking {parking} - Charger Utilization', fontsize=14)
    plt.xlabel('Time Period', fontsize=14)
    plt.ylabel('Charger ID', fontsize=14)
    
    # Save individual figure per parking
    plt.tight_layout()
    plt.savefig(f'Parking_{parking}_Charger_Utilization.png', dpi=300, bbox_inches='tight')


    
#############% robot utilization 


# Filter parkings with actual robot utilization
valid_parkings = []
robot_utils = {}  # Cache robot utilization for each valid parking

for parking in sorted(df_y['Parking'].unique()):
    df_parking = df_y[df_y['Parking'] == parking].copy()
    if df_parking.empty:
        continue

    original_ids = sorted(df_parking['Robot'].unique())
    id_mapping = {orig: new + 1 for new, orig in enumerate(original_ids)}
    df_parking['Robot_Seq'] = df_parking['Robot'].map(id_mapping)

    robot_util = df_parking.pivot_table(
        index='Time',
        columns='Robot_Seq',
        values='Value',
        aggfunc='sum',
        fill_value=0
    )

    if robot_util.empty:
        continue

    active_robots = robot_util.columns[robot_util.sum() > 0]
    if len(active_robots) == 0:
        continue

    valid_parkings.append(parking)
    robot_utils[parking] = robot_util[active_robots]

# If no valid data, stop here
if not valid_parkings:
    print("No robot utilization to plot.")
else:
    # Setup subplots
    fig, axes = plt.subplots(len(valid_parkings), 1, figsize=(15, 5 * len(valid_parkings)))
    if len(valid_parkings) == 1:
        axes = [axes]  # Ensure iterable

    plt.subplots_adjust(hspace=0.4)

    for idx, parking in enumerate(valid_parkings):
        util = robot_utils[parking]
        sns.heatmap(
            util.T,
            cmap=['white', 'blue'],
            linewidths=0.5,
            linecolor='black',
            cbar=False,
            ax=axes[idx]
        )

        axes[idx].set_title(f'Parking {parking} - Robot Utilization', pad=12)
        axes[idx].set_xlabel('Time Period')
        axes[idx].set_ylabel('Robot ID')
        axes[idx].set_yticks(
            ticks=np.arange(len(util.columns)),
            labels=util.columns,
            rotation=0
        )

        for spine in axes[idx].spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)

#    plt.suptitle('Robot Utilization Across All Parkings', y=1.02, fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('Combined_Robot_Utilization_Grid.png', dpi=300, bbox_inches='tight')
    plt.show()

plt.show()



###############################
######################### SAVING VARS FOR FUTURE NEEDS #########################
