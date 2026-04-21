# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 10:24:28 2025

@author: arsalann
"""

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

ParkNo = 1

model.ParkNumber = ParkNo 

model.S = pyo.Set(initialize=[x + 1 for x in range(model.ParkNumber)])

OmkarsData = 1
MyCase = 0


current_directory = os.path.dirname(__file__)
current_directory = os.getcwd()
file_path         = os.path.join(current_directory, 'data.xlsx')
file_path2         = os.path.join(current_directory, 'filtered_charging_events2_Updated2.xlsx')
file_path3         = os.path.join(current_directory, 'day2PublicWork.xlsx')
Price = pd.read_excel(file_path, sheet_name='electricicty_price')
Pattern = pd.read_excel(file_path, sheet_name='DemandPattern')
Pattern = np.repeat(Pattern['Pattern'], SampPerH)
Price = np.repeat(Price['Price'], SampPerH)
 

df = pd.read_excel(file_path3, sheet_name='Sheet1')
 

#    EVdata = DataCuration(file_path2, SampPerH, ChargerCap)
P_btot_Parkings = {}  # Format: {(s,t): value}
P_b_EVf = {}
P_dch_robf = {}
xf = {}
yf = {}
SOC_robf = {}
zf = {}
assignf = {}
assignRobotf = {}
u_robf = {}
u_rob_typef = {}
Nsf = {}
P_ch_EVf = {}
P_ch_robf = {}
CapRobotf ={}
PeakPowerf = {}

Alphaf = {}


parking_data = DataCuration(df, SampPerH, ChargerCap, ParkNo)

kl = 0

 
volt_per_node = np.zeros([32,24])
while True:
    kl += 1
    print(f"Iteration {kl}")

    if np.all(volt_per_node >= Vmin - 1e-3) or kl>3:
        print("Voltage constraints satisfied. Optimal solution found.")
        break



    #for s in range(1,ParkNo+1):
    for s in model.S:    
        print("s=",s)
        EVdata = parking_data[parking_data['ParkingNo'] == s].reset_index(drop=True)
         
        # Calculate and print results
        max_overlaps = max_overlaps_per_parking(EVdata)
        print('max_overlaps=', max_overlaps)
        model.nCharger = max_overlaps[s]
        
    
    
        
        #Price = pd.DataFrame(Price, columns=['Price'])
    
    
    
    
    
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
    
        
        
        # Export to Excel
        #output_filename = "AlphaSensi_Matrix.xlsx"
        #df_alpha.to_excel(output_filename, index=True)
        
        #print(f"AlphaSensi matrix successfully exported to {output_filename}")
        
        
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
        
        
        
        
        #for i in range(33):
        #    for j in range(33):
        #        if AlphaSensi[i,j] != 0:
        #            print(f'AlphaSensi[{i},{j}] = {AlphaSensi[i,j]:.4f}')
        
        
        # x = binary varible for CS, y = binary variable for robot charger, z = binary variable for choosing CS, zz= number of robots
    
        
        model.x_indices = pyo.Set(dimen=3, initialize=lambda model: (
            (k, i, t)
            for k in model.K
            for i in model.I
            for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
        ))
        
        model.y_indices = pyo.Set(dimen=3, initialize=lambda model: (
            (k, j, t)
            for k in model.K
            for j in model.J
            for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
        ))
        
        # Then create the variables using these sparse index sets
        model.x = pyo.Var(model.x_indices, within=pyo.Binary)
        model.y = pyo.Var(model.y_indices, within=pyo.Binary)
        
        model.z = pyo.Var(model.I, within=pyo.Binary)
        model.zz = pyo.Var(model.J, within=pyo.Binary)
        
        model.assign = pyo.Var(model.K, model.I, within=pyo.Binary)  # EV-charger assignment
        model.assignRobot = pyo.Var(model.K, model.J, within=pyo.Binary)  # Robot assignment
        
        # u = binary variable to buy either from the grid or from the robots
        model.u = pyo.Var(model.K, within=pyo.Binary)
        model.u_rob = pyo.Var(model.J,  model.T, within=pyo.Binary)
        
        # binary variable to define the type of the robot
        model.u_rob_type = pyo.Var(model.J,  model.KK, within=pyo.Binary)
        
        # Ns = number of chargers, Nrob = number of robots
        model.Ns = pyo.Var( within=pyo.Integers)
        model.Nrob = pyo.Var(within=pyo.Integers)
        
        # P_btot = P_buy_total from the grid to charge the EVs and robots, P_b_EV= Purchased electricity to charge EVS, P_b_rob = Purchased electricity to charge robots
        model.P_btot = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
        model.P_btotS = pyo.Var(model.T, within=pyo.NonNegativeReals)

        model.P_b_EV = pyo.Var(model.x_indices, within=pyo.NonNegativeReals)
        model.P_b_rob = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
        
        model.P_btotBar = pyo.Var(model.T, within=pyo.NonNegativeReals)
        
        # P_ch_EV= charging to EV, P_dch_EV = discharging from the EV, SOC_EV = state of charge of the EV
        model.P_ch_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
        model.P_dch_EV = pyo.Var(model.K, model.I, model.T, within=pyo.NonNegativeReals)
        model.SOC_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
        model.P_ch_rob = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
        model.P_dch_rob = pyo.Var(model.y_indices, within=pyo.NonNegativeReals)
        model.SOC_rob = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
        
        # Capacity of the robot
        model.CapRobot = pyo.Var(model.J, model.KK, within=pyo.NonNegativeReals)
        model.PeakPower = pyo.Var(within=pyo.NonNegativeReals)  # Peak grid power
        model.robot_parking_assignment = pyo.Var(model.J, within=pyo.Binary)
        
        model.Alpha = pyo.Var( within=pyo.Reals)
        #model.AlphaDown = pyo.Var(model.S, initialize=float('-inf'), within=pyo.Reals)
        
        
        #parking_value = {1: -100, 2: -100, 3: -100}
        parking_value = {1: -1000000}
        model.AlphaDown = pyo.Param(model.S, initialize=parking_value)
        
        voltage_at_bus = {b: 1.0 for b in model.Nodes}  # All buses start feasible
        
        
        ################ Energy constraints ###########
        def PowerPurchased(model, s,t):
            # Sum P_b_EV only for existing (k,i,t) combinations in x_indices
            ev_power = sum(
                model.P_b_EV[k, i, t] 
                for (k, i, t_var) in model.x_indices 
                if t_var == t  # Only sum for current time t
            )
            
            # Sum P_ch_rob (assuming it's defined over (j,t))
            robot_power = sum(model.P_ch_rob[j, t] for j in model.J)
            
            return model.P_btot[s,t] == (ev_power + robot_power)
        
        
        model.ConPurchasedPower = pyo.Constraint(model.S, model.T, rule=PowerPurchased)
        
        
        def PowerPurchasedLimit(model, s,t):
            return model.P_btot[s,t] <= PgridMax
        
        
        model.ConPowerPurchasedLimit = pyo.Constraint(model.S, model.T, rule=PowerPurchasedLimit)
        
        
        ##################### Charger assignment constraints ###############
        def NoCharger(model, k, i):
            return model.assign[k, i] <= model.z[i]
        
        
        model.ConNoCharger1 = pyo.Constraint(model.K, model.I, rule=NoCharger)
        
        
        def NoCharger2(model):
            return model.Ns == sum(model.z[i] for i in model.I)
        
        
        model.ConNoCharger2 = pyo.Constraint( rule=NoCharger2)
        
        
        def ChargingOptions(model, k):
            return sum(model.assign[k, i] for i in model.I) + sum(model.assignRobot[k, j] for j in model.J) <= 1
        
        
        model.ConChargingOptions = pyo.Constraint(model.K, rule=ChargingOptions)
        
        
        # Each EV assigned to <= 1 charger
        def SingleChargerAssignment(model, k):
            return sum(model.assign[k, i] for i in model.I) <= 1
        
        
        model.ConSingleAssign = pyo.Constraint(model.K,  rule=SingleChargerAssignment)
        
        
        def x_zero(model, k, i, t):
            if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
                return model.x[k, i, t] == 0
            else:
                return pyo.Constraint.Skip  # nothing to enforce outside the relevant time range
        
        
        model.Conx_zero = pyo.Constraint(model.x_indices, rule=x_zero)
        
        
        # def ChargerSingleEV_(model, k, i, s, t):
        #    if EVdata['AT'][k] <= t <= EVdata['DT'][k]:
        #       return sum(model.x[k, i, s, t] for k in model.K) <= 1
        #    else:
        #        return pyo.Constraint.Skip
        # model.ConChargerSingleEV_ = pyo.Constraint(model.x_indices, rule=ChargerSingleEV_)
        
        
        def ChargerSingleEV_(model, i, t):
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
            sum_x = sum(model.x[k, i, t] for k in relevant_evs if (k, i, t) in model.x)
            return sum_x <= 1
        
        
        # Declare the constraint over all possible (i,s,t) combinations
        model.ConChargerSingleEV_ = pyo.Constraint(
            model.I, model.T,
            rule=ChargerSingleEV_
        )
        
        
        def Link_x_charge(model, k, i, t):
            return model.x[k, i, t] <= model.assign[k, i]
        
        
        model.ConLink_x_charge = pyo.Constraint(model.x_indices, rule=Link_x_charge)
        
        
        # Constraint: If an EV is assigned to a fixed charger, the charger is occupied from AT to DT
        def charger_occupancy(model, k, i):
            """Ensures charger i at parking s is occupied from AT to DT if EV k is assigned"""
            # Only apply if EV is at this parking and AT <= DT
            if  EVdata['AT'][k] > EVdata['DT'][k]:
                return pyo.Constraint.Skip
        
            # Calculate required occupancy duration
            duration = EVdata['DT'][k] - EVdata['AT'][k] + 1
        
            # Sum only existing x variables (sparse-aware)
            occupied_sum = sum(
                model.x[k, i, t]
                for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
                if (k, i, t) in model.x  # Check if variable exists
            )
        
            # Big-M constraint reformulated for better numerical stability
            return occupied_sum >= duration * model.assign[k, i] - M * (1 - model.assign[k, i])
        
        
        model.ConChargerOccupancy = pyo.Constraint(
            model.K, model.I, 
            rule=charger_occupancy
        )
        
        
        # Constraint: No other EV can be assigned to the same charger during an occupied period
        def no_overlapping_assignments(model, k1, k2, i):
            """Prevents EV k2 from being assigned to charger i at parking s if it overlaps with EV k1's stay"""
            if k1 != k2:
                # Check if time windows overlap
                overlap_condition = (EVdata['AT'][k1] <= EVdata['DT'][k2]) and (EVdata['AT'][k2] <= EVdata['DT'][k1])
                if overlap_condition:
                    return model.assign[k1, i] + model.assign[k2, i] <= 1
            return pyo.Constraint.Skip
        
        
        model.ConNoOverlappingAssignments = pyo.Constraint(model.K, model.K, model.I, rule=no_overlapping_assignments)
        
        
        ###############  Robot assignment constraints
        def robot_single_connection(model, k, j, t):
            """Ensures each robot (j,s) charges at most one EV at any time t"""
            # Since we're using y_indices, we know (k,j,s,t) is valid
            # We need to sum over all EVs that could be served by this robot at this time
            return sum(model.y[k, j, t] for k in model.K
                       if (k, j, t) in model.y_indices) <= NevSame
        
        
        model.ConRobotSingleConnection = pyo.Constraint(model.y_indices, rule=robot_single_connection)
        
        # Create a mapping from (k,s,t) to possible robots
        # First create proper index sets
        model.ev_active_indices = pyo.Set(
            dimen=2,
            initialize=lambda model: [
                (k,  t)
                for k in model.K
                for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
            ]
        )
        
        
        def single_robot_connection(model, k, t):
            """Ensures each EV is connected to ≤1 robot at any time"""
            # Get all robots at this parking location
            available_robots = [j for j in model.J
                                if (k, j, t) in model.y_indices]
        
            return sum(model.y[k, j, t] for j in available_robots) <= 1
        
        
        model.ConSingleRobotConnection = pyo.Constraint(
            model.ev_active_indices,
            rule=single_robot_connection
        )
        
        
        def Link_y_charge(model, k, j, t):
            return model.y[k, j, t] <= model.assignRobot[k, j]
        
        
        model.ConLink_y_charge = pyo.Constraint(model.y_indices, rule=Link_y_charge)
        
        
        ########## charging constraints###################
        
        def Charging_toEV(model, k, t):
            robot_contrib = sum(
                model.P_dch_rob[k,j,t] 
                for j in model.J 
                 if (k,j, t) in model.y_indices
            )
            
            grid_contrib = sum(
                model.P_b_EV[k,i,t] 
                for i in model.I 
                 if (k,i, t) in model.x_indices
            )
            
            
            
            
            
            return model.P_ch_EV[k,t] == grid_contrib + robot_contrib
        
        model.ConCharging_toEV = pyo.Constraint(model.K, model.T, rule=Charging_toEV)
        
        
        def Charging_UpLimit(model, k, i, t):
            return model.P_b_EV[k,i,t]  <= ChargerCap
        
        
        model.ConCharging_UpLimit = pyo.Constraint(model.x_indices, rule=Charging_UpLimit)
        
        
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
        
        
        def SOC_Charge_limit1(model, k, i, t):
            return model.P_b_EV[k, i,  t] <= EVdata['EVcap'][k] * model.assign[k, i]
        model.ConSOC_Charge_limit1 = pyo.Constraint(model.x_indices, rule=SOC_Charge_limit1)

        
        def SOC_Charge_limit2(model, k, i, t):
            return model.P_b_EV[k, i, t] <= EVdata['EVcap'][k] * model.x[k, i, t]
        
        
        model.ConSOC_Charge_limit2 = pyo.Constraint(model.x_indices, rule=SOC_Charge_limit2)
        
        
        def SOC_Limit(model, k, t):
            return model.SOC_EV[k, t] <= EVdata['EVcap'][k]
        
        
        model.ConSOC_Limit = pyo.Constraint(model.K, model.T, rule=SOC_Limit)
        
        
        ############## Robot charging constraints #####################
        
        
        def Ch_robot_limit(model, j, t):
            return model.P_ch_rob[j, t] <= model.u_rob[j, t] *  (NevSame)*DCchargerCap
        model.ConCh_robot = pyo.Constraint(model.J, model.T, rule=Ch_robot_limit)
        
        
        def Dch_robot_limit1(model, j, t):
            active_discharges = sum(
                model.P_dch_rob[k,j, t] 
                for k in model.K 
                if (k,j, t) in model.y_indices  # Only sum existing variables
            )
            return active_discharges <= (1-model.u_rob[j,t])*NevSame*DCchargerCap
        model.ConDch_robot1 = pyo.Constraint(model.J,  model.T, rule=Dch_robot_limit1)
        
        
        
        
        def Dch_robot_limit2(model, k, j, t):
                return model.P_dch_rob[k, j, t] <= 1*DCchargerCap * model.y[k, j, t]
        model.Dch_robot_limit2 = pyo.Constraint(model.y_indices, rule=Dch_robot_limit2)
        
        
        
        #def Dch_robot_zero(model, k, j, s, t):
        #    """Force zero discharge for non-y_indices cases"""
        #    return model.P_dch_rob[k, j, s, t] == 0
        #model.Dch_robot_zero = pyo.Constraint(
        #    model.K * model.J * model.S * model.T - model.y_indices,  # All indices NOT in y_indices
        #    rule=Dch_robot_zero
        #)
        
        
        
        def Dch_robot_limit3(model, k, j, t):
            return model.P_dch_rob[k, j, t] <= sum(model.CapRobot[j, kk] for kk in model.KK)
        model.Dch_robot_limit3 = pyo.Constraint(model.y_indices, rule=Dch_robot_limit3)
        
        
        def SOC_Robot(model, j,  t):
            if t == 1:
                return model.SOC_rob[j,t] == 0.2*sum(model.CapRobot[j,kk] for kk in model.KK)
            else:
                active_discharges = sum(
                    model.P_dch_rob[k,j,t] 
                    for k in model.K 
                    if (k,j,t) in model.y_indices
                )
                return model.SOC_rob[j, t] == model.SOC_rob[j,t-1] + (1/SampPerH)*model.P_ch_rob[j, t] - (1/SampPerH)*active_discharges
        model.ConSOC_Robot = pyo.Constraint(model.J, model.T, rule=SOC_Robot)
        
        
        def SOC_Robot2(model, j, t):
            return model.SOC_rob[j, t] >= 0.2 * sum(model.CapRobot[j, kk] for kk in model.KK)
        
        
        model.ConSOC_Robot2 = pyo.Constraint(model.J, model.T, rule=SOC_Robot2)
        
        
        def SOC_Robot_limit(model, j, t):
            return model.SOC_rob[j, t] <= sum(model.CapRobot[j, kk] for kk in model.KK)
        
        
        model.ConSOC_Robot_limit = pyo.Constraint(model.J,  model.T, rule=SOC_Robot_limit)
        
        
        # def Ch_Robot_limit2(model, j, s, t):
        #    return model.P_ch_rob[j, s, t] <= 0.5*sum(model.CapRobot[j, s, kk] for kk in model.KK)
        # model.ConCh_Robot_limit2 = pyo.Constraint(model.J, model.S, model.T, rule= Ch_Robot_limit2 )
        
        
        def Ch_Robot_limit3(model, j, t):
            return sum(model.P_dch_rob[k, j, t] for k in model.K) <= (NevSame)*DCchargerCap
        
        
        #model.ConCh_Robot_limit3 = pyo.Constraint(model.J, model.S, model.T, rule= Ch_Robot_limit3 )
        
        def RobotCapLimit4(model, j, kk):
            return model.CapRobot[j, kk] <= RobotTypes[kk - 1] * model.u_rob_type[j, kk]
        
        
        model.ConRobotCapLimit4 = pyo.Constraint(model.J, model.KK, rule=RobotCapLimit4)
        
        
        def RobotCapLimit2(model, j, kk):
            return sum(model.u_rob_type[j, kk] for kk in model.KK) <= 1
        
        
        model.ConRobotCapLimit2 = pyo.Constraint(model.J, model.KK, rule=RobotCapLimit2)
        
        
        
        def RobotTotNum(model, j, kk):
            return sum(model.u_rob_type[j,  kk] for kk in model.KK for j in model.J) <= MaxRobot
        #model.ConRobotTotNum = pyo.Constraint(model.J, model.S, model.KK, rule=RobotTotNum)
        
        def Dch_rob_zero(model, k, j, t):
            if t < EVdata['AT'][k] or t > EVdata['DT'][k]:
                return model.P_dch_rob[k, j, t] == 0
            else:
                return pyo.Constraint.Skip  # nothing to enforce outside the relevant time range
        
        
        #model.ConDch_rob_zero = pyo.Constraint(model.K, model.J, model.S, model.T, rule=Dch_rob_zero)
        
        
    
        ##################### PARKING CONSTRAINTS ##################
        
        
        
        
        def PeakPowerConstraint(model, s,t):
            return model.PeakPower >= model.P_btot[s, t]               
        model.ConPeakPower = pyo.Constraint( model.S, model.T, rule=PeakPowerConstraint)
        
 
        def PTotalS(model, s,t):
            return model.P_btotS[t] == model.P_btot[s, t]               
        #model.ConPTotalS = pyo.Constraint( model.S, model.T, rule=PTotalS)

        
        def AlphaFun(model, s):
            return model.Alpha >= model.AlphaDown[s]
            #return model.Alpha[s] >= -100
        
        model.ConAlphaFun = pyo.Constraint(model.S, rule=AlphaFun)
        
        
        ######################### TO CONSIDER BESS ##################
        def NoCharger3(model):
            return  model.Ns >= max_overlaps
        #model.ConNoCharger3 = pyo.Constraint(model.S, rule = NoCharger3)
        
        ############## OBJECTIVE FUNCTION ##################
        
        model.obj = pyo.Objective(expr=(1) * (PFV_Charger * model.Ns * Ch_cost ) +
                                       (1) * PFV_Rob * sum(
            model.u_rob_type[j, kk] * robotCC[kk - 1] for j in model.J  for kk in model.KK) +
                                       sum(Price.iloc[t - 1] * (0.001) * model.P_btot[s, t]  for t in model.T) +
                                       (1 / 30) * PeakPrice * model.PeakPower
                                        + 1*(model.Alpha), sense=pyo.minimize)
        
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
        
        results = solver.solve(model, tee=True)
        
        for t in model.T:
            P_btot_Parkings[(s, t)] = pyo.value(model.P_btot[s,t])


        P_b_EVf.update({
            (k,i,s,t): pyo.value(model.P_b_EV[k,i,t]) 
            for (k,i,t) in model.x_indices
        })
        
        P_dch_robf.update({
            (k, j, s, t): pyo.value(model.P_dch_rob[k, j, t])
            for (k, j, t) in model.y_indices  # Uses predefined sparse indices
        })



        xf.update( {
            (k, i , s, t):  pyo.value(model.x[k, i, t]) 
            for (k, i, t) in model.x_indices  # Uses predefined sparse indices
        })

        yf.update(  {
            (k, j, s, t): pyo.value(model.y[k, j, t])
            for (k, j, t) in model.y_indices  # Uses predefined sparse indices
        })
        


        for j in model.J:
            for t in model.T:
                SOC_robf[(j,s,t)] =   pyo.value(model.SOC_rob[j, t]) 
        
        for  i  in model.I:
            zf[(i,s)] = pyo.value(model.z[i])
        
        
        for k in model.K:
            for i in model.I:
                assignf[(k,i,s)] = pyo.value(model.assign[k, i])
        
       
        for k in model.K:
            for j in model.J:
                assignRobotf[(k,i,s)] = pyo.value(model.assignRobot[k, j])
                    
        for j in model.J:
            for t in model.T:
                u_robf[(j,s,t)] =   pyo.value(model.u_rob[j, t])         
    
 
        for j in model.J:
            for kk in model.KK:
                u_rob_typef[(j,s,kk)] =   pyo.value(model.u_rob_type[j, kk])  

        Nsf[(s)] = pyo.value(model.Ns)
 
        for k in model.K:
            for t in model.T:
                P_ch_EVf[(k , s, t)] =   pyo.value(model.P_ch_EV[k, t])  
     
     
     
        for j in model.J:
            for t in model.T:
                P_ch_robf[(j,s,t)] =   pyo.value(model.P_ch_rob[j, t]) 
     

        for j in model.J:
            for kk in model.KK:
                CapRobotf[(j,s,kk)] =   pyo.value(model.CapRobot[j, kk])       
 
        PeakPowerf[(s)] = pyo.value(model.PeakPower)
        Alphaf[(s)] = pyo.value(model.Alpha)
       




#######################################
########################################
########################################
#######################################







    # Solve master problem
#    results = solver.solve(model, tee=True)
#    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
#        print("Master problem infeasible. Terminating.")
#        break

    # Get current solution values
    P_btot_current =  {
        (s, t): P_btot_Parkings[s,t]
        for s in model.S
        #for p in [s]  # Assuming model.S indices match parking_to_node keys
        if s in parking_to_bus
        for t in model.T}
    # Call PowerFlow subproblem
  #  [_, volt_per_node, duals2, duals_balance_p, duals_Dev, SPobj] = PowerFlow(P_btot_current, Pattern, Price)
    [min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj] = PowerFlow(
        P_btot_current, Pattern, Price)

    # Termination check (using volt_per_node)


    # Generate cuts for violated voltages
    for s in model.S:
        # Get the bus connected to parking s (skip if not found)
        worst_bus = parking_to_bus.get(s, None)
        if worst_bus is None:
            continue  # Skip if parking s has no bus connection
        
        for t in model.T:
            # Check voltage violation at the connected bus
            if volt_per_node[worst_bus-1, t-1] < Vmin - 1e-3:
                violation = max(0, Vmin - volt_per_node[worst_bus-1, t-1])
                
                print(f"Parking {s} (Bus {worst_bus}), t={t}:")
                print(f"  Voltage violation = {violation:.4f}")
                print(f"  Duals (DevLow, DevUp, BalanceP): {duals_DevLow.get((worst_bus,t),0):.4f}, {duals_DevUp.get((worst_bus,t),0):.4f}, {duals_balance_p.get((worst_bus,t),0):.4f}")
    
                # Generate cut ONLY for this parking s (no sum over s)
                model.cuts.add(
                    SPobj 
                    - (duals_DevLow.get((worst_bus, t), 0.0) 
                    + duals_balance_p.get((worst_bus, t), 0.0) 
                    - duals_DevUp.get((worst_bus, t), 0.0)) * 100* violation * (model.P_btot[s, t] / sb)
                    <= model.Alpha
                )
                
                print(f"Added cut for parking {s} (bus {worst_bus}), t={t}: SPobj = {SPobj}")

    # Update lower bound
    for s in model.S:
        if Alphaf.get((s),0.0) > model.AlphaDown[s]:
            model.AlphaDown[s] = Alphaf.get((s),0.0)
            print(f"Scenario {s}: Alpha[s]={model.Alpha[s].value}, Min Voltage={np.min(volt_per_node)}")




Ncharger_val = {(s): Nsf[s] for s in model.S}
print(Ncharger_val)
# x_val = {(k,i,t): sum( pyo.value(model.x[k,i,s,t]) for s in model.S) for k in model.K for i in model.I for t in model.T}
#assign_val = {(k, i): sum(assignf[k, i, s] for s in model.S) for k in model.K for i in model.I}
P_ch_rob_val = {(j, t): sum(P_ch_robf[j, s, t] for s in model.S) for j in model.J for t in model.T}
P_Dch_rob_val = {(j, t): sum(P_ch_robf[j, s, t] for s in model.S) for j in model.J for t in model.T}

CapRobot_val = {(j, s, kk): CapRobotf[j, s, kk] * u_rob_typef[j, s, kk] for j in model.J
                for s in model.S for kk in model.KK}
CapRobot_val2 = {(j, s, kk): RobotTypes[kk - 1] * u_rob_typef[j, s, kk] for j in model.J for s in
                 model.S for kk in model.KK}



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
    print("s=",s)
    grid = sum(P_b_EVf.get((k,i,s,t), 0) for k in model.K for i in model.I for t in model.T)
    print("grid = ", grid)
    robot = sum(P_dch_robf.get((k,j,s,t), 0) for (k,j,t) in model.y_indices)

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
    # Check discharge activity using saved values
    has_discharge = any(
        P_dch_robf.get((k,j,s,t), 0) > 0.1  # Using dictionary lookup
        for (k,j,t) in model.y_indices
    )
    
    if not has_discharge:
        continue

    # Find active robots
    active_robots = {
        j for (k,j,t) in model.y_indices 
        if P_dch_robf.get((k,j,s,t), 0) > 0.1
    }

    if not active_robots:
        continue

    # Plot each active robot
    fig, axes = plt.subplots(len(active_robots), 1, 
                           figsize=(12, 2*len(active_robots)), 
                           squeeze=False)
    
    for idx, j in enumerate(active_robots):
        ax = axes[idx, 0]
        # Get capacity from saved values
        robot_capacity = sum(
            CapRobotf.get((j,s,kk), 0) 
            for kk in model.KK
        )
        
        # Get SOC data from saved values
        soc = [SOC_robf.get((j,s,t), 0) for t in model.T]
        ax.plot(model.T, soc, 'b-', label='SOC (kWh)')
        
        # Get charging data from saved values
        charge = [P_ch_robf.get((j,s,t), 0) for t in model.T]
        
        # Calculate discharge from saved values
        discharge = [0.0]*len(model.T)
        for (k,j,t) in model.y_indices:
            discharge[t-1] += P_dch_robf.get((k,j,s,t), 0)
        
        # Create stacked bars
        ax.bar(model.T, charge, color='g', alpha=0.3, label='Charging')
        ax.bar(model.T, discharge, bottom=charge, color='r', alpha=0.3, label='Discharging')
        
        ax.set_title(f'Robot {j} in Parking {s}')
        ax.grid(True)
        
        if idx == len(active_robots)-1:
            ax.legend()
            ax.set_xlabel('Time period')
        ax.set_ylabel('Power (kW)')
    
    plt.tight_layout()
    plt.savefig(f'Robot_Discharge_Parking_{s}.png', dpi=300)
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
    # Get grid charging from P_b_EVf dictionary
    P_b_EV_total = [
        sum(
            P_b_EVf.get((k,i,s,t), 0)  # Using .get() with default 0
            for k in model.K 
            for i in model.I 
        )
        for t in model.T
    ]
    
    plt.plot(time, P_b_EV_total, 'b-', linewidth=2, label='Grid Charging')
    plt.fill_between(time, P_b_EV_total, color='blue', alpha=0.1)
    plt.xlabel('Time period', fontsize=14)
    plt.ylabel('Power (kW)', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.title(f'Parking {s} - Grid Charging Power')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    # --- Robot Discharging Plot ---
    plt.subplot(2, 1, 2)
    # Get robot discharging from P_dch_robf dictionary
    P_dch_rob_total = [
        sum(
            P_dch_robf.get((k,j,s,t), 0)  # Using .get() with default 0
            for (k,j,t) in model.y_indices
        )
        for t in model.T
    ]
    
    plt.plot(time, P_dch_rob_total, 'r-', linewidth=2, label='Robot Discharging')
    plt.fill_between(time, P_dch_rob_total, color='red', alpha=0.1)
    plt.xlabel('Time period', fontsize=14)
    plt.ylabel('Power (kW)', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.title(f'Parking {s} - Robot Discharging Power')
    plt.grid(True, linestyle=':')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Parking_{s}_Power_Profile.png', dpi=300, bbox_inches='tight')
     
    # Print summary statistics
    print(f"\nParking {s} Power Summary:")
    print(f"Max Grid Charging: {max(P_b_EV_total):.2f} kW at hour {np.argmax(P_b_EV_total)+1}")
    print(f"Total Robot Discharge: {sum(P_dch_rob_total):.2f} kWh")

# Electricity price plot (unchanged as it doesn't use model variables)
plt.figure(figsize=(10, 5))
plt.plot(time, 10*Price, color='red', marker='*')
plt.xlabel('Time period', fontsize=14)
plt.ylabel('Price (SEK/MWh)', fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.savefig('InputPrice.png', dpi=600)
plt.show()




plt.figure(figsize=(10, 8))  # Slightly taller figure for two subplots

# --- Overall Purchased Electricity ---
plt.subplot(2, 1, 1)
time = range(1, SampPerH * 24 + 1)
total_power = [
    sum(P_btot_Parkings.get((s,t), 0) for s in model.S)  # Sum across all parkings
    for t in model.T
]
plt.plot(time, total_power, 'b-', linewidth=1.5)
plt.xlabel('Time period', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.title('Overall Purchased Electricity', fontsize=14)
plt.grid()

# --- Per-Parking Purchased Electricity ---
plt.subplot(2, 1, 2)
colors = ['r', 'g', 'b', 'm', 'c']  # Different colors for each parking
for s in model.S:
    P_Tot_Purch = [
        P_btot_Parkings.get((s,t), 0)  # Get from saved results
        for t in model.T
    ]
    plt.plot(time, P_Tot_Purch, 
             label=f'Parking {s}',
             color=colors[s-1],  # Use different color for each parking
             linewidth=1.5)

plt.xlabel('Time period', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.title('Purchased Electricity by Parking', fontsize=14)
plt.grid()
plt.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('PurchasePower.png', dpi=600, bbox_inches='tight')
 
plt.show()

# Create DataFrames for x and y variables
# Create dataframes from saved variables
x_data = []
for (k, i, s, t), val in xf.items():  # Use xf dictionary
    x_data.append({
        'EV': k,
        'Charger': i,
        'Parking': s,
        'Time': t,
        'Value': val  # Already contains the value
    })

y_data = []
for (k, j, s, t), val in yf.items():  # Use yf dictionary
    y_data.append({
        'EV': k,
        'Robot': j,
        'Parking': s,
        'Time': t,
        'Value': val  # Already contains the value
    })

# Convert to DataFrames
df_x = pd.DataFrame(x_data)
df_y = pd.DataFrame(y_data)

# Calculate costs using saved values
TotalChargerCost = sum(Ch_cost * Nsf[s] for s in model.S)
print("Total Charger Cost = ", TotalChargerCost)

TotalPurchaseCost = sum(Price.iloc[t - 1] * (0.001) * P_btot_Parkings[(s,t)] 
                   for s in model.S for t in model.T)
print("Total cost of electricity purchased = ", TotalPurchaseCost)

TotalPeakCost = (1 / 30) * PeakPrice * sum(PeakPowerf[s] for s in model.S)
print("Total cost of Peak Power = ", TotalPeakCost)

TotalMCRcost = sum(u_rob_typef[(j,s,kk)] * robotCC[kk - 1] 
               for j in model.J for s in model.S for kk in model.KK)
print("Total cost of MCRs = ", TotalMCRcost)

# Calculate objective value from components
ObjValue = (TotalChargerCost + TotalMCRcost + TotalPurchaseCost + TotalPeakCost)
print("Objective function = ", ObjValue)

# Create and plot charger utilization heatmap
charger_utilization = df_x.pivot_table(index='Time', 
                                      columns='Charger', 
                                      values='Value',
                                      aggfunc='sum',
                                      fill_value=0)

plt.figure(figsize=(15, 8))
sns.heatmap(charger_utilization.T, 
            cmap=['white', 'green'],
            linewidths=0.5,
            linecolor='lightgray',
            cbar=False)

plt.title('Charger Utilization Over Time', fontsize=16, pad=20)
plt.xlabel('Time Period', fontsize=14)
plt.ylabel('Charger ID', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.text(0, -1.5, 'Green cells indicate when a charger is being used by an EV',
         fontsize=10, ha='left')
plt.savefig('charger_utilization_heatmap.png', dpi=300, bbox_inches='tight')

# Parking-specific charger utilization plots
parkings = sorted(df_x['Parking'].unique())

for parking in parkings:
    df_parking = df_x[df_x['Parking'] == parking].copy()
    
    # Map charger IDs to sequential numbers
    original_charger_ids = sorted(df_parking['Charger'].unique())
    charger_id_mapping = {orig_id: new_id + 1 
                         for new_id, orig_id in enumerate(original_charger_ids)}
    df_parking['Charger_Sequential'] = df_parking['Charger'].map(charger_id_mapping)
    
    charger_utilization = df_parking.pivot_table(
        index='Time', 
        columns='Charger_Sequential', 
        values='Value',
        aggfunc='sum',
        fill_value=0
    )
    
    # Skip if no active chargers
    if charger_utilization.empty:
        print(f"No active chargers in Parking {parking}. Skipping.")
        continue
    
    # Create complete time index
    full_index = range(1, model.HORIZON + 1)
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
    
    # Set x-axis labels
    n = max(1, len(full_index) // 10)
    xticks = [i for i in full_index if i % n == 1 or i == full_index[-1]]
    ax.set_xticks([x - 1.5 for x in xticks])
    ax.set_xticklabels(xticks)
    
    plt.title(f'Parking {parking} - Charger Utilization', fontsize=14)
    plt.xlabel('Time Period', fontsize=14)
    plt.ylabel('Charger ID', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'Parking_{parking}_Charger_Utilization.png', dpi=300, bbox_inches='tight')
     
#############% robot utilization 

# Filter parkings with actual robot utilization
valid_parkings = []
robot_utils = {}  # Cache robot utilization for each valid parking

# Create dataframe from yf dictionary
y_data = []
for (k, j, s, t), val in yf.items():  # Use yf dictionary
    y_data.append({
        'EV': k,
        'Robot': j,
        'Parking': s,
        'Time': t,
        'Value': val  # Already contains the value
    })
df_y = pd.DataFrame(y_data)

for parking in sorted(df_y['Parking'].unique()):
    df_parking = df_y[df_y['Parking'] == parking].copy()
    if df_parking.empty:
        continue

    # Map robot IDs to sequential numbers
    original_ids = sorted(df_parking['Robot'].unique())
    id_mapping = {orig: new + 1 for new, orig in enumerate(original_ids)}
    df_parking['Robot_Seq'] = df_parking['Robot'].map(id_mapping)

    # Create pivot table
    robot_util = df_parking.pivot_table(
        index='Time',
        columns='Robot_Seq',
        values='Value',
        aggfunc='sum',
        fill_value=0
    )

    # Filter for active robots
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
    fig, axes = plt.subplots(len(valid_parkings), 1, 
                           figsize=(15, 5 * len(valid_parkings)),
                           squeeze=False)
    
    plt.subplots_adjust(hspace=0.4)

    for idx, parking in enumerate(valid_parkings):
        util = robot_utils[parking]
        
        # Create heatmap
        sns.heatmap(
            util.T,
            cmap=['white', 'blue'],
            linewidths=0.5,
            linecolor='black',
            cbar=False,
            ax=axes[idx, 0]
        )

        axes[idx, 0].set_title(f'Parking {parking} - Robot Utilization', pad=12)
        axes[idx, 0].set_xlabel('Time Period')
        axes[idx, 0].set_ylabel('Robot ID')
        axes[idx, 0].set_yticks(
            ticks=np.arange(len(util.columns)),
            labels=util.columns,
            rotation=0
        )

        for spine in axes[idx, 0].spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig('Combined_Robot_Utilization_Grid.png', 
               dpi=300, 
               bbox_inches='tight')
    plt.show()

###############################
######################### SAVING VARS FOR FUTURE NEEDS #########################
