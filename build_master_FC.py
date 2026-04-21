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
def build_masterOnlyFC(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price):
    [parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

    modelFC = pyo.ConcreteModel()

    
    EVdata = parking_data[parking_data['ParkingNo'] == s].reset_index(drop=True)
     
    # Calculate and print results
    max_overlaps = max_overlaps_per_parking(EVdata)
    print('max_overlaps=', max_overlaps)
    print('s ==',s)
    modelFC.nCharger = max_overlaps[s]
    
    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')
    
   
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
    
    plt.show(block=False)
    # -*- coding: utf-8 -*-
    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')
    M = 50 #Big M
    # present value factors
    PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
    print(len(EVdata))
    modelFC.HORIZON = SampPerH * 24
    modelFC.nEV = len(EVdata) - 1
    modelFC.Nodes = pyo.Set(initialize=range(33)) # Buses 0 to 32
    modelFC.T = pyo.Set(initialize=[x + 1 for x in range(modelFC.HORIZON)])
    modelFC.I = pyo.Set(initialize=[x + 1 for x in range(modelFC.nCharger)])
    modelFC.K = pyo.Set(initialize=[x + 1 for x in range(modelFC.nEV)])
    
    # x = binary varible for CS, z = binary variable for choosing CS, zz= number of robots
    modelFC.x_indices = pyo.Set(dimen=3, initialize=lambda modelFC: (
    (k, i, t)
    for k in modelFC.K
    for i in modelFC.I
    for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
    ))
    # I have limited x to not search the space which is not required
        
    #charger usage tracking
    modelFC.z = pyo.Var(modelFC.I,  within=pyo.Binary)
    modelFC.assign = pyo.Var(modelFC.K, modelFC.I,  within=pyo.Binary) # EV-charger assignment
    modelFC.occupancy = pyo.Var(
        modelFC.I, modelFC.T,
        within=pyo.NonNegativeReals,
        bounds=(0, 1)  
    )
    modelFC.charger_used = pyo.Var(modelFC.I, within=pyo.Binary)  # Charger is installed



    # u = binary variable to buy either from the grid or from the robots
    modelFC.u = pyo.Var(modelFC.K, within=pyo.Binary)
    modelFC.P_btotBar = pyo.Var( modelFC.T, within=pyo.NonNegativeReals)
    # Ns = number of chargers
    modelFC.Ns = pyo.Var( within=pyo.NonNegativeIntegers)
    # P_btot = P_buy_total from the grid to charge the EVs and robots, P_b_EV= Purchased electricity to charge EVS
    modelFC.P_btot = pyo.Var( modelFC.T, within=pyo.NonNegativeReals)
    modelFC.P_b_EV = pyo.Var(modelFC.K, modelFC.I,  modelFC.T, within=pyo.NonNegativeReals)
    # P_ch_EV= charging to EV, P_dch_EV = discharging from the EV, SOC_EV = state of charge of the EV
    modelFC.P_ch_EV = pyo.Var(modelFC.K, modelFC.T, within=pyo.NonNegativeReals)
    modelFC.P_dch_EV = pyo.Var(modelFC.K, modelFC.I, modelFC.T, within=pyo.NonNegativeReals)
    modelFC.SOC_EV = pyo.Var(modelFC.K, modelFC.T, within=pyo.NonNegativeReals)
    modelFC.PeakPower = pyo.Var( within=pyo.NonNegativeReals) # Peak grid power
    modelFC.Alpha = pyo.Var(within=pyo.Reals)  # Subproblem approximation
    modelFC.AlphaDown = pyo.Param(initialize=-100, mutable=True)  # Lower bound
    
    modelFC.x = pyo.Expression(
        modelFC.K, modelFC.I, modelFC.T,
        initialize=0.0  # Default value
    )
    
    ################ Energy constraints ###########
    def PowerPurchased(modelFC, t):
        return modelFC.P_btot[t] == sum(
            modelFC.P_ch_EV[k, t] 
            for k in modelFC.K 
            if EVdata['AT'][k] <= t <= EVdata['DT'][k]
        )
            
    
    modelFC.ConPurchasedPower = pyo.Constraint( modelFC.T, rule=PowerPurchased)
    
    def PowerPurchasedLimit(modelFC, t):
        return modelFC.P_btot[ t] <= PgridMax
    modelFC.ConPowerPurchasedLimit = pyo.Constraint( modelFC.T, rule=PowerPurchasedLimit)
    
    ##################### Charger assignment constraints ###############
    def NoCharger(modelFC, k, i):
        return modelFC.assign[k, i] <= modelFC.z[i]
#    modelFC.ConNoCharger1 = pyo.Constraint(modelFC.K, modelFC.I,  rule=NoCharger)
    
    
    
    
    def NoCharger2(modelFC):
        return modelFC.Ns == sum(modelFC.z[i] for i in modelFC.I)
    modelFC.ConNoCharger2 = pyo.Constraint( rule=NoCharger2)
    # Each EV assigned to <= 1 charger
    
    def SingleChargerAssignment(modelFC, k):
        return sum(modelFC.assign[k, i] for i in modelFC.I) <= 1
    modelFC.ConSingleAssign = pyo.Constraint(modelFC.K,  rule=SingleChargerAssignment)
    


    def charger_capacity_new(modelFC, i, t):
            """At most one EV can use charger i at time t"""
            # Sum of assignments for EVs active at time t
            active_assignments = sum(
                modelFC.assign[k, i]
                for k in modelFC.K
                if EVdata['AT'][k] <= t <= EVdata['DT'][k]
            )
            
            # This is the key constraint: occupancy ≤ sum of assignments
            # Since occupancy ≤ 1 (next constraint), this ensures ≤ 1 EV per charger
            return modelFC.occupancy[i, t] <= active_assignments
    
#    modelFC.ConChargerCapacityNew = pyo.Constraint(
#        modelFC.I, modelFC.T, rule=charger_capacity_new
#    )


    def occupancy_limit(modelFC, i, t):
            """Occupancy cannot exceed 1"""
            return modelFC.occupancy[i, t] <= 1
        
    modelFC.ConOccupancyLimit = pyo.Constraint(
        modelFC.I, modelFC.T, rule=occupancy_limit
    )


    def charger_installation(modelFC, i):
        """Charger must be installed if any EV is assigned to it"""
        # Sum of all assignments to charger i
        total_assignments = sum(modelFC.assign[k, i] for k in modelFC.K)
        
        # If any EV is assigned, charger must be installed
        return modelFC.z[i] >= total_assignments / M  # Big-M
        
    modelFC.ConChargerInstallation = pyo.Constraint(
        modelFC.I, rule=charger_installation
    )



    def charging_power_constraint(modelFC, k, t):
            """EV charging power limited by occupancy"""
            at = EVdata['AT'][k]
            dt = EVdata['DT'][k]
            
            if t < at or t > dt:
                return modelFC.P_ch_EV[k, t] == 0
            
            # Power is limited by whether EV is charging at time t
            # Sum occupancy over all chargers this EV might use
            total_occupancy_for_ev = sum(
                modelFC.assign[k, i] * modelFC.occupancy[i, t]
                for i in modelFC.I
            )
            
            return modelFC.P_ch_EV[k, t] <= ChargerCap * total_occupancy_for_ev
    
#    modelFC.ConChargingPower = pyo.Constraint(
#        modelFC.K, modelFC.T, rule=charging_power_constraint
#    )



    def simplified_power_constraint(modelFC, k, t):
        """Simplified linear power constraint"""
        at = EVdata['AT'][k]
        dt = EVdata['DT'][k]
        
        if t < at or t > dt:
            return modelFC.P_ch_EV[k, t] == 0
        
        # Option A: Power limited by whether EV is assigned to ANY charger
        total_assignment = sum(modelFC.assign[k, i] for i in modelFC.I)
        return modelFC.P_ch_EV[k, t] <= ChargerCap * total_assignment
    
    modelFC.ConSimplePower = pyo.Constraint(modelFC.K, modelFC.T, rule=simplified_power_constraint)





    
    def charger_occupancy_new(modelFC, k, i):
            """If EV k is assigned to charger i, then charger must be occupied during EV's stay"""
            at = EVdata['AT'][k]
            dt = EVdata['DT'][k]
            
            if at > dt:
                return pyo.Constraint.Skip
            
            # Option A: Sum of occupancy over EV's stay must equal duration if assigned
            # This ensures the charger is fully occupied during the EV's stay
            occupancy_sum = sum(
                modelFC.occupancy[i, t] 
                for t in range(at, dt + 1)
            )
            
            return occupancy_sum >= (dt - at + 1) * modelFC.assign[k, i]
        
    modelFC.ConChargerOccupancyNew = pyo.Constraint(
            modelFC.K, modelFC.I, rule=charger_occupancy_new
        )
    
#####################     
    # Constraint: No other EV can be assigned to the same charger during an occupied period
    # option A
    def no_overlapping_assignments(modelFC, k1, k2, i):
        """Prevents EV k2 from being assigned to charger i at parking s if it overlaps with EV k1's stay"""
        if k1 != k2:
            # Check if time windows overlap
            overlap_condition = (EVdata['AT'][k1] <= EVdata['DT'][k2]) and (EVdata['AT'][k2] <= EVdata['DT'][k1])
            if overlap_condition:
                return modelFC.assign[k1, i] + modelFC.assign[k2, i] <= 1
        return pyo.Constraint.Skip
    
    
#    modelFC.ConNoOverlappingAssignments = pyo.Constraint(modelFC.K, modelFC.K, modelFC.I, rule=no_overlapping_assignments)
    
        
    # option B
    def cumulative_charger_constraint(modelFC, i, t):
        """Number of EVs assigned to charger i that are active at time t ≤ 1"""
        active_evs_count = sum(
            modelFC.assign[k, i]
            for k in modelFC.K
            if EVdata['AT'][k] <= t <= EVdata['DT'][k]
        )
        return active_evs_count <= 1

#    modelFC.ConCumulativeCharger = pyo.Constraint(
#        modelFC.I, modelFC.T, rule=cumulative_charger_constraint
#        )
 

# option C
# Only create constraints at times when something changes
    def get_critical_times(EVdata):
        """Get all arrival and departure times"""
        critical_times = set()
        for idx, ev in EVdata.iterrows():
            critical_times.add(ev['AT'])
            critical_times.add(ev['DT'])
        return sorted(critical_times)
    
    critical_times = get_critical_times(EVdata)
    modelFC.CRITICAL_TIMES = pyo.Set(initialize=critical_times)
    
    def critical_time_constraint(modelFC, i, ct):
        """Constraint only at critical times"""
        active_at_ct = sum(
            modelFC.assign[k, i]
            for k in modelFC.K
            if EVdata['AT'][k] <= ct <= EVdata['DT'][k]
        )
        return active_at_ct <= 1

    modelFC.ConCriticalTimes = pyo.Constraint(
        modelFC.I, modelFC.CRITICAL_TIMES, rule=critical_time_constraint
            )


    ###################
    def symmetry_breaking(modelFC, i):
        """Force chargers to be used in order"""
        if i > 1:
            return modelFC.z[i] <= modelFC.z[i-1]
        return pyo.Constraint.Skip

    modelFC.ConSymmetryBreaking = pyo.Constraint(modelFC.I, rule=symmetry_breaking)

    def occupancy_link(modelFC, k, i, t):
        at = EVdata['AT'][k]
        dt = EVdata['DT'][k]
        
        if t < at or t > dt:
            return pyo.Constraint.Skip
        
        return modelFC.occupancy[i, t] >= modelFC.assign[k, i]
    modelFC.ConOccupancyLink = pyo.Constraint(modelFC.K, modelFC.I, modelFC.T, rule=occupancy_link)

    ############### Robot assignment constraints
    # Create a mapping from (k,s,t) to possible robots
    # First create proper index sets
    modelFC.ev_active_indices = pyo.Set(
    dimen=3,
    initialize=lambda modelFC: [
    (k, EVdata['ParkingNo'][k - 1], t)
    for k in modelFC.K
    for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
    ]
    )
    ########## charging constraints###################
    def Charging_toEV(modelFC, k, t):
        return modelFC.P_ch_EV[k,t] == sum(modelFC.P_b_EV[k, i, t] for i in modelFC.I) 
#    modelFC.ConCharging_toEV = pyo.Constraint(modelFC.K, modelFC.T, rule=Charging_toEV)
  
    def Charging_UpLimit(modelFC, k, t):
        return modelFC.P_ch_EV[k, t] <= ChargerCap
    modelFC.ConCharging_UpLimit = pyo.Constraint(modelFC.K, modelFC.T, rule=Charging_UpLimit)
    
    def SOC_EV_f1(modelFC, k, t):
        if t < EVdata['AT'][k]:
            return modelFC.SOC_EV[k, t] == 0
        elif t == EVdata['AT'][k]:
            return modelFC.SOC_EV[k, t] == EVdata['SOCin'][k] * EVdata['EVcap'][k]
        elif t > EVdata['AT'][k] and t <= EVdata['DT'][k]:
            return modelFC.SOC_EV[k, t] == modelFC.SOC_EV[k, t - 1] + (1 / SampPerH) * modelFC.P_ch_EV[k, t]
        else:
            return pyo.Constraint.Skip # nothing to enforce outside the relevant time range
    def SOC_EV_f2(modelFC, k, t):
        if t == EVdata['DT'][k]:
            return modelFC.SOC_EV[k, t] == 1 * EVdata['SOCout'][k] * EVdata['EVcap'][k]
        else:
            return pyo.Constraint.Skip # nothing to enforce outside the relevant time range
    modelFC.ConSOC_EV_f1 = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_EV_f1)
    modelFC.ConSOC_EV_f2 = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_EV_f2)
   
    def SOC_Charge_limit1(modelFC, k, i, t):
        return modelFC.P_b_EV[k, i, t] <= EVdata['EVcap'][k] * modelFC.assign[k, i]
    modelFC.ConSOC_Charge_limit1 = pyo.Constraint(modelFC.K, modelFC.I,  modelFC.T, rule=SOC_Charge_limit1)

    
    def charging_power_limit(modelFC, k, t):
        """EV charging power cannot exceed charger capacity"""
        return modelFC.P_ch_EV[k, t] <= ChargerCap

    modelFC.ConChargingPowerLimit = pyo.Constraint(
        modelFC.K, modelFC.T, rule=charging_power_limit
    )
       
    def SOC_Limit(modelFC, k, t):
        return modelFC.SOC_EV[k, t] <= EVdata['EVcap'][k]
    modelFC.ConSOC_Limit = pyo.Constraint(modelFC.K, modelFC.T, rule=SOC_Limit)
    ############## Robot charging constraints #####################
    ##################### PARKING CONSTRAINTS ##################
 
    
    def PeakPowerConstraint(modelFC, t):
        return modelFC.PeakPower >= modelFC.P_btot[t]
    modelFC.ConPeakPower = pyo.Constraint( modelFC.T, rule=PeakPowerConstraint)
    
    def AlphaFun(modelFC):
        return modelFC.Alpha >= modelFC.AlphaDown
        #return model.Alpha[s] >= -100
    
    modelFC.ConAlphaFun = pyo.Constraint( rule=AlphaFun)

    ############## OBJECTIVE FUNCTION ##################
    modelFC.obj = pyo.Objective(expr=(1) * (PFV_Charger * modelFC.Ns * Ch_cost) +
    (1/SampPerH)*sum(Price.iloc[t - 1] * (0.001) * modelFC.P_btot[t] for t in modelFC.T) +
    (1 / 30) * PeakPrice *modelFC.PeakPower + 1*(modelFC.Alpha) , sense=pyo.minimize)
    
    
    modelFC.cuts = pyo.ConstraintList()  # THIS IS CRUCIAL
    
    
    
    
    
    
    
    x_indices = []
    for k in modelFC.K:
        at = int(EVdata['AT'][k])
        dt = int(EVdata['DT'][k])
        for i in modelFC.I:
            for t in range(at, min(dt, modelFC.HORIZON) + 1):
                x_indices.append((k, i, t))
    
    modelFC.x_sparse_indices = pyo.Set(initialize=x_indices, dimen=3)
    modelFC.x = pyo.Expression(modelFC.x_sparse_indices)
    
    for (k, i, t) in modelFC.x_sparse_indices:
        modelFC.x[k, i, t] = modelFC.assign[k, i] * modelFC.occupancy[i, t]
    
    print(f"Created x expression with {len(x_indices)} entries")
        
    
    
    
    return modelFC



def build_masterBESS(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price):
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
    plt.hist(ATT, bins=18, color='skyblue', edgecolor='black',label = 'Arrival time')
    plt.legend()
    plt.grid()
    plt.ylabel('Frequency')
    
    plt.xlim(1,SampPerH*24)
    plt.subplot(2,1,2)
    plt.hist(DTT, bins=18, color='red', edgecolor='black', label = 'Departure time')
    plt.xlim(1,SampPerH*24)
    plt.grid()
    plt.legend()
    plt.xlabel('Time sample')
    plt.ylabel('Frequency')
    plt.savefig('Histogram.png', dpi = 300)
    
    plt.show(block=False)
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
    model.P_ch_rob = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
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
    
    ################ Energy constraints ###########
    def PowerPurchased(model, t):
        ev_power = sum(
            model.P_b_EV[k, i, t] 
            for k in model.K 
            for i in model.I 
            if (k, i, t) in model.x_indices
        )
        
        robot_power = sum(model.P_ch_rob[j, t] for j in model.J )
        
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
    
    
#    model.ConNoOverlappingAssignments = pyo.Constraint(model.K, model.K, model.I, rule=no_overlapping_assignments)
    
 
    
 
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

    model.ConCriticalTimes = pyo.Constraint(
         model.I, model.CRITICAL_TIMES, rule=critical_time_constraint
             )
   
 
    
 
    
 
    
 
    
 
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
        return model.P_ch_rob[j, t] <= model.u_rob[j, t] *  (1)*ChargerCap
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
            return model.SOC_rob[j, t] == model.SOC_rob[j,t-1] + (1/SampPerH)*model.P_ch_rob[j, t]  - (1/SampPerH)*active_discharges
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
    
    ################## Robot to charger
    
    

    
    
    
    '''
   #Approach 1 
    # Variables
    #model.assignRobToCh  = pyo.Var(model.J, model.I, within=pyo.Binary)
    model.x_rob_time = pyo.Var(model.J, model.T, within=pyo.Binary)
    model.x_rob_combined = pyo.Var(model.J, model.I, model.T, bounds=(0,1))
    
    # Linearization constraints for x_rob_combined = x_rob_time * assignRobToCh 
    def linearize_x_rob1(model, j, i, t):
        return model.x_rob_combined[j,i,t] <= model.x_rob_time[j,t]
    model.ConLinearizeXR1 = pyo.Constraint(model.J, model.I, model.T, rule=linearize_x_rob1)
    
    def linearize_x_rob2(model, j, i, t):
        return model.x_rob_combined[j,i,t] <= model.assignRobToCh [j,i]
    model.ConLinearizeXR2 = pyo.Constraint(model.J, model.I, model.T, rule=linearize_x_rob2)
    
    def linearize_x_rob3(model, j, i, t):
        return model.x_rob_combined[j,i,t] >= model.x_rob_time[j,t] + model.assignRobToCh [j,i] - 1
    model.ConLinearizeXR3 = pyo.Constraint(model.J, model.I, model.T, rule=linearize_x_rob3)
    
    # Modified original constraints
    def Link_x_rob(model, j, i, t):
        return model.x_rob_combined[j,i,t] <= model.assignRobToCh[j,i]
    model.ConLink_x_rob = pyo.Constraint(model.J, model.I, model.T, rule=Link_x_rob)
    
    def ChargerSingleUse2_updated(model, i, t):
        ev_usage = model.occupancy[i, t]
        robot_usage = sum(model.x_rob_combined[j, i, t] for j in model.J)
        return ev_usage + robot_usage <= 1
    model.ConChargerSingleUse2 = pyo.Constraint(model.I, model.T, rule=ChargerSingleUse2_updated)
    
    def ChargerOneRobot(model, i, t):
        return sum(model.x_rob_combined[j, i, t] for j in model.J) <= 1
    model.ConChargerOneRobot = pyo.Constraint(model.I, model.T, rule=ChargerOneRobot)
    
    def Ch_robot_limit4(model, j, i, t):
        return model.P_ch_rob[j, i, t] <= model.x_rob_combined[j, i, t] * ChargerCap
    model.ConCh_robot4 = pyo.Constraint(model.J, model.I, model.T, rule=Ch_robot_limit4)

    '''
    '''
    
     # Approach 2
    def Ch_robot_limit4(model, j, i, t):
        return model.P_ch_rob[j, i, t] <= model.assignRobToCh[j, i] * ChargerCap
    
    model.ConCh_robot4 = pyo.Constraint(model.J, model.I, model.T, rule=Ch_robot_limit4)

    def ChargerSingleUse_power(model, i, t):
        robot_power = sum(model.P_ch_rob[j, i, t] for j in model.J)
        return model.occupancy[i, t] + robot_power / ChargerCap <= 1
    
    model.ConChargerSingleUse = pyo.Constraint(model.I, model.T, rule=ChargerSingleUse_power)

    # enforcing one robot per charger at each time
    def OneRobotPerCharger(model, i, t):
        return sum(
            model.P_ch_rob[j, i, t] / ChargerCap
            for j in model.J
        ) <= 1
    
    model.ConOneRobotPerCharger = pyo.Constraint(model.I, model.T, rule=OneRobotPerCharger)
    '''

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
        return  model.Ns >= max_overlaps[s] - 1 
    model.ConNoCharger3 = pyo.Constraint(rule = NoCharger3)
    
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
    
   

