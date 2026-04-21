# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 17:22:11 2026

@author: arsalann
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 10:52:21 2025

@author: arsalann
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 12:01:57 2025

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



[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()



 

 

def build_master(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price):
    [parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

    model = pyo.ConcreteModel()

    
    EVdata = parking_data[parking_data['ParkingNo'] == s].reset_index(drop=True)
     
    # Calculate and print results
    max_overlaps = max_overlaps_per_parking(EVdata)
    print('max_overlaps=', max_overlaps)
    print('s ==',s)
    model.nCharger = max_overlaps[s]
    


    
    #Price = pd.DataFrame(Price, columns=['Price'])





    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')
    
    line_data = data['lineData']
    
    #################
    ATT = EVdata['AT']
    DTT = EVdata['DT']
    
    
    plt.subplot(2,1,1)
    plt.hist(ATT, bins=18, color='yellow', edgecolor='brown',label = 'Arrival time')
    plt.legend()
    plt.grid()
    plt.ylabel('Frequency')
    
    plt.xlim(1,SampPerH*24)
    plt.subplot(2,1,2)
    plt.hist(DTT, bins=18, color='black', edgecolor='brown', label = 'Departure time')
    plt.xlim(1,SampPerH*24)
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
    
#    if s == 1 or s == 3:
    model.nMESS = 12
#    elif s == 2:
#        model.nMESS = 15
        
        
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
    #model.x = pyo.Var(model.x_indices, within=pyo.Binary)
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
    model.Ns = pyo.Var( within=pyo.NonNegativeIntegers)
    model.Nrob = pyo.Var(within=pyo.NonNegativeIntegers)
    
    # P_btot = P_buy_total from the grid to charge the EVs and robots, P_b_EV= Purchased electricity to charge EVS, P_b_rob = Purchased electricity to charge robots
    model.P_btot = pyo.Var( model.T, within=pyo.NonNegativeReals)
    model.P_btotS = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.P_b_EV = pyo.Var(model.x_indices, within=pyo.NonNegativeReals)
    model.P_b_rob = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    model.Out1_P_b_EV = pyo.Var(model.T, within=pyo.NonNegativeReals)

    model.P_btotBar = pyo.Var(model.T, within=pyo.NonNegativeReals)
    
    # P_ch_EV= charging to EV, P_dch_EV = discharging from the EV, SOC_EV = state of charge of the EV
    model.P_ch_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.P_dch_EV = pyo.Var(model.K, model.I, model.T, within=pyo.NonNegativeReals)
    model.SOC_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
    model.P_ch_rob = pyo.Var(model.J, model.I, model.T, within=pyo.NonNegativeReals)
    model.P_dch_rob = pyo.Var(model.y_indices, within=pyo.NonNegativeReals)
    model.SOC_rob = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    
    # Capacity of the robot
    model.CapRobot = pyo.Var(model.J, model.KK, within=pyo.NonNegativeReals)
    model.PeakPower = pyo.Var(within=pyo.NonNegativeReals)  # Peak grid power
    model.robot_parking_assignment = pyo.Var(model.J, within=pyo.Binary)
    
    model.assignRobToCh = pyo.Var(model.J, model.I, within=pyo.Binary)  
    model.x_rob = pyo.Var(model.J, model.I, model.T, within=pyo.Binary)  # time occupancy of robot on charger
    
    # Decomposition variables
    model.Alpha = pyo.Var(within=pyo.Reals)  # Subproblem approximation
    model.AlphaDown = pyo.Param(initialize=-100, mutable=True)  # Lower bound
    
    voltage_at_bus = {b: 1.0 for b in model.Nodes}  # All buses start feasible
    model.x = pyo.Expression(
        model.K, model.I, model.T,
        initialize=0.0  # Default value
    )
    
    model.occupancy = pyo.Var(
        model.I, model.T,
        within=pyo.NonNegativeReals,
        bounds=(0, 1)  # Can be fractional during relaxation
    )
    model.charger_used = pyo.Var(model.I, within=pyo.Binary)  # Charger is installed
    
    
    
    if s == 2:
        for i in model.I:
            if i <= 120:
                model.z[i].fix(1)
    else: 
        for i in model.I:
            if i <= 30:
                model.z[i].fix(1)
        
    ################ Energy constraints ###########
    def PowerPurchased(model, t):
        ev_power = sum(
            model.P_b_EV[k, i, t] 
            for k in model.K 
            for i in model.I 
            if (k, i, t) in model.x_indices
        )
        
        robot_power = sum(model.P_ch_rob[j, i, t] for j in model.J for i in model.I)
        
        return model.P_btot[t] == (ev_power + robot_power)
    
    model.ConPurchasedPower = pyo.Constraint(model.T, rule=PowerPurchased) 
    
    def Out1_P_b_EV(model, t):
        ev_power = sum(
            model.P_b_EV[k, i, t] 
            for k in model.K 
            for i in model.I 
            if (k, i, t) in model.x_indices
        )
    
        return model.Out1_P_b_EV[t] == ev_power 
    model.ConOut1_P_b_EV = pyo.Constraint(model.T, rule=Out1_P_b_EV)  
   

    
#    model.ConPurchasedPower = pyo.Constraint(model.T, rule=PowerPurchased)
    
    
    def PowerPurchasedLimit(model, t):
        return model.P_btot[t] <= PgridMax
    
    
    model.ConPowerPurchasedLimit = pyo.Constraint(model.T, rule=PowerPurchasedLimit)
    
    
    ##################### Charger assignment constraints ###############
    def NoCharger(model, k, i):
        return model.assign[k, i]  <= model.z[i]   
    model.ConNoCharger1 = pyo.Constraint(model.K, model.I, rule=NoCharger)


    def NoChargerRob(model, j, i):
        return model.assignRobToCh[j,i] <= model.z[i]   
    model.NoChargerRob = pyo.Constraint(model.J, model.I, rule=NoChargerRob)    
    
    def NoCharger2(model):
        return model.Ns == sum(model.z[i] for i in model.I)   
    model.ConNoCharger2 = pyo.Constraint( rule=NoCharger2)
    
    
    def ChargingOptions(model, k):
        return sum(model.assign[k, i] for i in model.I) + sum(model.assignRobot[k, j] for j in model.J) <= 1    
    model.ConChargingOptions = pyo.Constraint(model.K, rule=ChargingOptions)
    
    
    # Each EV assigned to <= 1 charger
    def SingleChargerAssignment(model, k):
        return sum(model.assign[k, i] for i in model.I) <= 1
#    model.ConSingleAssign = pyo.Constraint(model.K,  rule=SingleChargerAssignment)
 


    def occupancy_limit(model, i, t):
            """Occupancy cannot exceed 1"""
            return model.occupancy[i, t] <= 1
        
    model.ConOccupancyLimit = pyo.Constraint(
        model.I, model.T, rule=occupancy_limit
    )


   
    def charger_installation(model, i):
        """Charger must be installed if any EV is assigned to it"""
        # Sum of all assignments to charger i
        total_assignments = sum(model.assign[k, i] for k in model.K)
        
        # If any EV is assigned, charger must be installed
        return model.z[i] >= total_assignments / M  # Big-M
        
    model.ConChargerInstallation = pyo.Constraint(
        model.I, rule=charger_installation
    )



    def charging_power_constraint(model, k, t):
            """EV charging power limited by occupancy"""
            at = EVdata['AT'][k]
            dt = EVdata['DT'][k]
            
            if t < at or t > dt:
                return model.P_ch_EV[k, t] == 0
            
            # Power is limited by whether EV is charging at time t
            # Sum occupancy over all chargers this EV might use
            total_occupancy_for_ev = sum(
                model.assign[k, i] * model.occupancy[i, t]
                for i in model.I
            )
            
            return model.P_ch_EV[k, t] <= ChargerCap * total_occupancy_for_ev
    
#    model.ConChargingPower = pyo.Constraint(
#        model.K, model.T, rule=charging_power_constraint
#    )



    def simplified_power_constraint(model, k, t):
        """Simplified linear power constraint"""
        at = EVdata['AT'][k]
        dt = EVdata['DT'][k]
        
        if t < at or t > dt:
            return model.P_ch_EV[k, t] == 0
        
        # Option A: Power limited by whether EV is assigned to ANY charger
        total_assignment = sum(model.assign[k, i] for i in model.I)
        return model.P_ch_EV[k, t] <= ChargerCap * total_assignment
    
#    model.ConSimplePower = pyo.Constraint(model.K, model.T, rule=simplified_power_constraint)


    
    def charger_occupancy_new(model, k, i):
            """If EV k is assigned to charger i, then charger must be occupied during EV's stay"""
            at = EVdata['AT'][k]
            dt = EVdata['DT'][k]
            
            if at > dt:
                return pyo.Constraint.Skip
            
            # Option A: Sum of occupancy over EV's stay must equal duration if assigned
            # This ensures the charger is fully occupied during the EV's stay
            occupancy_sum = sum(
                model.occupancy[i, t] 
                for t in range(at, dt + 1)
            )
            
            return occupancy_sum >= (dt - at + 1) * model.assign[k, i]
        
    model.ConChargerOccupancyNew = pyo.Constraint(
            model.K, model.I, rule=charger_occupancy_new
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
    
    
    #model.ConNoOverlappingAssignments = pyo.Constraint(model.K, model.K, model.I, rule=no_overlapping_assignments)
    
 
    
 
 # Only create constraints at times when something changes
    def get_critical_times(EVdata):
         """Get all arrival and departure times"""
         critical_times = set()
         for idx, ev in EVdata.iterrows():
             critical_times.add(ev['AT'])
             critical_times.add(ev['DT'])
         return sorted(critical_times)
     
    critical_times = get_critical_times(EVdata)
    model.CRITICAL_TIMES = pyo.Set(initialize=critical_times)
     
    def critical_time_constraint(model, i, ct):
         """Constraint only at critical times"""
         active_at_ct = sum(
             model.assign[k, i]
             for k in model.K
             if EVdata['AT'][k] <= ct <= EVdata['DT'][k]
         )
         return active_at_ct <= 1

    model.ConCriticalTimes = pyo.Constraint(model.I, model.CRITICAL_TIMES, rule=critical_time_constraint   )
     
 
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
    ################################
 
    
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

    
    def charging_power_limit(model, k, t):
        """EV charging power cannot exceed charger capacity"""
        return model.P_ch_EV[k, t] <= ChargerCap
#    model.ConChargingPowerLimit = pyo.Constraint(
#        model.K, model.T, rule=charging_power_limit
#    )    
    
    
    def SOC_Limit(model, k, t):
        return model.SOC_EV[k, t] <= EVdata['EVcap'][k]   
    model.ConSOC_Limit = pyo.Constraint(model.K, model.T, rule=SOC_Limit)
    
    
    ############## Robot charging constraints #####################
    
    
    def Ch_robot_limit(model, j,  t):
        return sum(model.P_ch_rob[j, i, t] for i in model.I) <= model.u_rob[j, t] *  (1)*ChargerCap
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
            return model.SOC_rob[j, t] == model.SOC_rob[j,t-1] + (1/SampPerH)*sum(model.P_ch_rob[j, i, t] for i in model.I) - (1/SampPerH)*active_discharges
    model.ConSOC_Robot = pyo.Constraint(model.J, model.T, rule=SOC_Robot)
    
    
    def SOC_Robot2(model, j, t):
        return model.SOC_rob[j, t] >= 0.2 * sum(model.CapRobot[j, kk] for kk in model.KK)
    
    
    model.ConSOC_Robot2 = pyo.Constraint(model.J, model.T, rule=SOC_Robot2)
    
    
    def SOC_Robot_limit(model, j, t):
        return model.SOC_rob[j, t] <= sum(model.CapRobot[j, kk] for kk in model.KK)
    model.ConSOC_Robot_limit = pyo.Constraint(model.J,  model.T, rule=SOC_Robot_limit)
    
    

    
    
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
    #############################################
    def symmetry_breaking(model, i):
        """Force chargers to be used in order"""
        if i > 1:
            return model.z[i] <= model.z[i-1]
        return pyo.Constraint.Skip

    #model.ConSymmetryBreaking = pyo.Constraint(model.I, rule=symmetry_breaking)   
    ################## Robot to charger
    
        
    # base version 
    def Link_x_rob(model, j, i, t):
        return model.x_rob[j, i, t] <= model.assignRobToCh[j, i]
    model.ConLink_x_rob = pyo.Constraint(model.J, model.I, model.T, rule=Link_x_rob)

    def ChargerSingleUse2_updated(model, i, t):
        """A charger can be used by either an EV OR a robot at any time"""
        
        # EV usage at charger i, time t (using occupancy)
        ev_usage = model.occupancy[i, t]
        
        # Robot usage at charger i, time t 
        # Assuming x_rob[j,i,t] is binary (robot j at charger i at time t)
        robot_usage = sum(model.x_rob[j, i, t] for j in model.J)
        
        # Combined usage cannot exceed 1
        return ev_usage + robot_usage <= 1
    
    model.ConChargerSingleUse2 = pyo.Constraint(model.I, model.T, rule=ChargerSingleUse2_updated)

    def ChargerOneRobot(model, i, t):
        return sum(model.x_rob[j, i, t] for j in model.J) <= 1
    model.ConChargerOneRobot = pyo.Constraint(model.I, model.T, rule=ChargerOneRobot)
    
    
    
    def Ch_robot_limit4(model, j, i, t):
        return model.P_ch_rob[j, i, t] <= model.x_rob[j, i, t] *  (1)*ChargerCap
    model.ConCh_robot4 = pyo.Constraint(model.J, model.I, model.T, rule=Ch_robot_limit4)
    
        
    def prevent_charger_switching(model, j, i, t):
        """
        Prevent robot from switching chargers between consecutive periods
        """
        if t == 1:
            return pyo.Constraint.Skip
        
        # Sum of robot being at ANY OTHER charger at current time
        charging_other = sum(model.x_rob[j, i_other, t] for i_other in model.I if i_other != i)
        
        # If robot was at charger i at previous time, cannot be elsewhere now
        return model.x_rob[j, i, t-1] + charging_other <= 1
    
#    model.ConNoChargerSwitching = pyo.Constraint(model.J, model.I,         [t for t in model.T if t > 1],  rule=prevent_charger_switching    )
  

    def single_charger_per_robot(model, j):
        return sum(model.assignRobToCh[j, i] for i in model.I) <= 1
#    model.ConSingleChargerPerRobot = pyo.Constraint(model.J, rule=single_charger_per_robot)


    def one_charger_per_time(model, j, t):
        return sum(model.x_rob[j,i,t] for i in model.I) <= 1
    
    model.ConOneChargerPerTime = pyo.Constraint(model.J, model.T, rule=one_charger_per_time)
    
    
    def charging_stationarity(model, j, i, t):
        if t == 1:
            return pyo.Constraint.Skip
        return model.x_rob[j,i,t] <= model.x_rob[j,i,t-1] + (1 - model.u_rob[j,t])
    model.Concharging_stationarity = pyo.Constraint(model.J, model.I, model.T, rule=charging_stationarity)

  ##################### PARKING CONSTRAINTS ##################
    


    
    
    def PeakPowerConstraint(model, t):
        return model.PeakPower >= model.P_btot[t]               
    model.ConPeakPower = pyo.Constraint( model.T, rule=PeakPowerConstraint)
    
 
    def PTotalS(model, t):
        return model.P_btotS[t] == model.P_btot[t]               
    #model.ConPTotalS = pyo.Constraint( model.S, model.T, rule=PTotalS)

    
    def AlphaFun(model):
        return model.Alpha >= model.AlphaDown
        #return model.Alpha[s] >= -100
    
    model.ConAlphaFun = pyo.Constraint( rule=AlphaFun)
    
    
    ######################### TO CONSIDER BESS ##################
    def NoCharger3(model):
        return  model.Ns >= 30
    #max_overlaps[s] - 1 
    #model.ConNoCharger3 = pyo.Constraint(rule = NoCharger3)
    
    
    def NoCharger4(model):
        return  model.Ns <= max_overlaps[s] - 1 
    #model.ConNoCharger4 = pyo.Constraint(rule = NoCharger4)
    ############## OBJECTIVE FUNCTION ##################
    
    model.obj = pyo.Objective(expr=(1) * (PFV_Charger * model.Ns * Ch_cost ) +
                                   (1) * PFV_Rob * sum(
        model.u_rob_type[j, kk] * robotCC[kk - 1] for j in model.J  for kk in model.KK) +
                                   (1/SampPerH)*sum(Price.iloc[t - 1] * (0.001) * model.P_btot[ t]  for t in model.T) +
                                   (1 / 30) * PeakPrice * model.PeakPower
                                    + 1*(model.Alpha), sense=pyo.minimize)
    
    #####################################################
    
    #####################
    x_indices = []
    for k in model.K:
        at = int(EVdata['AT'][k])
        dt = int(EVdata['DT'][k])
        for i in model.I:
            for t in range(at, min(dt, model.HORIZON) + 1):
                x_indices.append((k, i, t))
    
    model.x_sparse_indices = pyo.Set(initialize=x_indices, dimen=3)
    model.x = pyo.Expression(model.x_sparse_indices)
    
    for (k, i, t) in model.x_sparse_indices:
        model.x[k, i, t] = model.assign[k, i] * model.occupancy[i, t]
    
    print(f"Created x expression with {len(x_indices)} entries")
  
    ####
#    model.x_rob = pyo.Expression(model.J, model.I, model.T)
#    for j in model.J:
#        for i in model.I:
#            for t in model.T:
        
#                model.x_rob[j, i, t] = model.x_rob_combined[j , i, t]
    
    #print(f"Created x_rob ")    
  
    
  
    # model.cuts = pyo.Constraint(model.Nodes, rule=voltage_feasibility_cut)
    model.cuts = pyo.ConstraintList()  # THIS IS CRUCIAL
    return model
    
   

