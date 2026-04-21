# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 12:02:13 2025

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

vb = 12.66
sb = 10

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





[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot] = GlobalData()



vb = 12.66
sb = 10

# Which dataset do you use?
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
    
if OmkarsData == 1:

#    EVdata = DataCuration(file_path2, SampPerH, ChargerCap)

    EVdata = DataCuration(file_path3, SampPerH, ChargerCap)
       
    # Calculate and print results
    max_overlaps = max_overlaps_per_parking(EVdata)
    for parking_no, max_overlap in max_overlaps.items():
        print(f"ParkingNo {parking_no}: Maximum overlaps = {max_overlap}")
    
    model.nCharger = 232


    
    #Price = pd.DataFrame(Price, columns=['Price'])
elif MyCase == 1:

    EVdata = pd.read_excel(file_path, sheet_name='ParkingData_ver2')

    # Calculate and print results
    max_overlaps = max_overlaps_per_parking(EVdata)
    for parking_no, max_overlap in max_overlaps.items():
        print(f"ParkingNo {parking_no}: Maximum overlaps = {max_overlap}")

    model.nCharger = 40


data = pyo.DataPortal()
data.load(filename='iee33_bus_data.dat')

line_data = data['lineData']


M = 50 #Big number
M2 = 5000 # Big number
# present value factors
PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))

print(len(EVdata))
model.HORIZON = SampPerH * 24
model.nEV = len(EVdata) - 1
model.ParkNo = 3

model.Nodes = pyo.Set(initialize=range(33))  # Buses 0 to 32
model.T = pyo.Set(initialize=[x + 1 for x in range(model.HORIZON)])
model.I = pyo.Set(initialize=[x + 1 for x in range(model.nCharger)])
model.K = pyo.Set(initialize=[x + 1 for x in range(model.nEV)])
model.S = pyo.Set(initialize=[x + 1 for x in range(model.ParkNo)])

# x = binary varible for CS, y = binary variable for robot charger, z = binary variable for choosing CS, zz= number of robots
model.x = pyo.Var(model.K, model.I, model.S, model.T, within=pyo.Binary)

model.x_indices = pyo.Set(dimen=4, initialize=lambda model: (
    (k, i, EVdata['ParkingNo'][k - 1], t)
    for k in model.K
    for i in model.I
    for t in range(EVdata['AT'][k], EVdata['DT'][k] + 1)
))



# Then create the variables using these sparse index sets
model.x = pyo.Var(model.x_indices, within=pyo.Binary)

model.z = pyo.Var(model.I, model.S, within=pyo.Binary)

model.assign = pyo.Var(model.K, model.I, model.S, within=pyo.Binary)  # EV-charger assignment

# u = binary variable to buy either from the grid or from the robots
model.u = pyo.Var(model.K, within=pyo.Binary)


model.P_btotBar = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)

# Ns = number of chargers, Nrob = number of robots
model.Ns = pyo.Var(model.S, within=pyo.Integers)

# P_btot = P_buy_total from the grid to charge the EVs and robots, P_b_EV= Purchased electricity to charge EVS, P_b_rob = Purchased electricity to charge robots
model.P_btot = pyo.Var(model.S, model.T, within=pyo.NonNegativeReals)
model.P_b_EV = pyo.Var(model.K, model.I, model.S, model.T, within=pyo.NonNegativeReals)

# P_ch_EV= charging to EV, P_dch_EV = discharging from the EV, SOC_EV = state of charge of the EV
model.P_ch_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)
model.P_dch_EV = pyo.Var(model.K, model.I, model.T, within=pyo.NonNegativeReals)
model.SOC_EV = pyo.Var(model.K, model.T, within=pyo.NonNegativeReals)

# Capacity of the robot
model.PeakPower = pyo.Var(model.S, within=pyo.NonNegativeReals)  # Peak grid power
voltage_at_bus = {b: 1.0 for b in model.Nodes}  # All buses start feasible


################ Energy constraints ###########
def PowerPurchased(model, s, t):
    return model.P_btot[s, t] == (
            sum(model.P_b_EV[k, i, s, t] for k in model.K for i in model.I) 
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



########## charging constraints###################

def Charging_toEV(model, k, t):
    return model.P_ch_EV[k,t] == sum(model.P_b_EV[k,i,s,t] for i in model.I for s in model.S) 

model.ConCharging_toEV = pyo.Constraint(model.K, model.T, rule=Charging_toEV)


def Charging_UpLimit(model, k, t):
    return model.P_ch_EV[k, t] <= ChargerCap


model.ConCharging_UpLimit = pyo.Constraint(model.K, model.T, rule=Charging_UpLimit)


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



##################### PARKING CONSTRAINTS ##################

# Additional constraints to ensure assignment variables respect parking locations
def assign_charger_parking(model, k, i, s):
    if EVdata['ParkingNo'][k - 1] != s:
        return model.assign[k, i, s] == 0
    else:
        return pyo.Constraint.Skip


model.ConAssignChargerParking = pyo.Constraint(model.K, model.I, model.S, rule=assign_charger_parking)




def PeakPowerConstraint(model, s, t):
    return model.PeakPower[s] >= model.P_btot[s, t]


model.ConPeakPower = pyo.Constraint(model.S, model.T, rule=PeakPowerConstraint)

############## OBJECTIVE FUNCTION ##################

model.obj = pyo.Objective(expr=(1) * (PFV_Charger * sum(model.Ns[s] * Ch_cost for s in model.S)) +
                               sum(Price.iloc[t - 1] * (0.001) * model.P_btot[s, t] for s in model.S for t in model.T) +
                               (1 / 30) * PeakPrice * sum(model.PeakPower[s] for s in model.S), sense=pyo.minimize)

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
    'Cuts': 3,              # Maximum cut generation
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


Ncharger_val = {(s): pyo.value(model.Ns[s]) for s in model.S}
print(Ncharger_val)
# x_val = {(k,i,t): sum( pyo.value(model.x[k,i,s,t]) for s in model.S) for k in model.K for i in model.I for t in model.T}
assign_val = {(k, i): sum(pyo.value(model.assign[k, i, s]) for s in model.S) for k in model.K for i in model.I}


Ns_val = {(s): pyo.value(model.Ns[s]) for s in model.S}

print("Non-zero chargers:")
for s, val in Ns_val.items():
    if val != 0:  # Check if the value is not zero
        print(f"Chargers in parking {s}: {val}")

print("Non-zero robot capacities:")



#for k in model.K:
#    for j in model.J:
#        for s in model.S:
#            for t in model.T:
#                 if pyo.value(model.P_dch_rob[k, j,s, t]) >= 0.000001:
#                        print(f"P_dch_rob[{k}, {j}, {s}, {t}]", pyo.value(model.P_dch_rob[k, j,s, t]))






# Prepare SOC matrix HEAT MAP
soc_matrix = [[pyo.value(model.SOC_EV[k, t]) / EVdata.loc[k - 1, 'EVcap'] * 100
               for t in model.T]
              for k in model.K]

plt.figure(figsize=(12, 8))
sns.heatmap(soc_matrix, annot=False, fmt='.0f', cmap='YlGnBu',
            xticklabels=model.T)
# , yticklabels=range(1,model.nEV+1)
plt.xlabel('Time (hours)')
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
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (kW)')
    plt.title(f'Parking {s} - Grid Charging Power')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    # --- Robot Discharging Plot (Sparse-Aware) ---

    plt.tight_layout()
    plt.savefig(f'Parking_{s}_Power_Profile.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print(f"\nParking {s} Power Summary:")
    print(f"Max Grid Charging: {max(P_b_EV_total):.2f} kW at hour {np.argmax(P_b_EV_total)+1}")
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
time = range(1, SampPerH * 24 + 1)
plt.plot(time, Price)
plt.xlabel('Time (hour)')
plt.ylabel('Price ($/MWh)')
plt.title('Electricity price')
plt.grid(True)

plt.subplot(2, 1, 2)
time = range(1, SampPerH * 24)
time_array = np.array(time)  # Convert to NumPy array
time_transposed = np.transpose(time_array)
for s in model.S:
    P_Tot_Purch = [model.P_btot[s, t].value for t in model.T]
    plt.plot(P_Tot_Purch, label=f'Parking {s}')
plt.xlabel('Time (hours)')
plt.ylabel('Power (kW)')
plt.title(f'Total purchased power')
plt.grid(True)
plt.legend()  # Show legend with labels
plt.tight_layout()
plt.savefig('InputPrice.png', dpi=600)

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



# Convert to DataFrames
df_x = pd.DataFrame(x_data)

# Create Excel writer object
with pd.ExcelWriter('charging_schedule_MultipleParkings_OnlyFC.xlsx') as writer:
    # Write x variables (station charging)
    df_x.to_excel(writer, sheet_name='Station_Charging', index=False)


    # Add summary sheets
    df_x.groupby(['Charger', 'Time'])['Value'].sum().unstack().to_excel(
        writer, sheet_name='Charger_Utilization')


print("Successfully exported to charging_schedule_MultipleParkings_OnlyFC.xlsx")

TotalChargerCost = sum(Ch_cost * pyo.value(model.Ns[s]) for s in model.S)
print("Total Charger Cost = ", TotalChargerCost)

TotalPurchaseCost = sum(Price.iloc[t - 1] * (0.001) * pyo.value(model.P_btot[s, t]) for s in model.S for t in model.T)
print("Total cost of electricity purchased  = ", TotalPurchaseCost)

TotalPeakCost = (1 / 30) * PeakPrice * sum(pyo.value(model.PeakPower[s]) for s in model.S)
print("Total cost of Peak Power  = ", TotalPeakCost)


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
plt.xlabel('Time Period', fontsize=12)
plt.ylabel('Charger ID', fontsize=12)
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
    
    # Plot heatmap
    plt.figure(figsize=(15, 6))
    sns.heatmap(
        charger_utilization.T,
        cmap=['white', 'green'],
        linewidths=0.5,
        linecolor='lightgray',
        cbar=False
    )
    
    plt.title(f'Parking {parking} - Charger Utilization', fontsize=14)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Charger ID', fontsize=12)
    
    # Save individual figure per parking
    plt.tight_layout()
    plt.savefig(f'Parking_{parking}_Charger_Utilization.png', dpi=300, bbox_inches='tight')
    
    
#############% robot utilization 
plt.show()

