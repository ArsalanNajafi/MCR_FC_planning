# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 09:54:13 2025

@author: arsalann
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:54:02 2025

@author: arsalann
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PowerFlow import PowerFlow  # Assuming you have this module
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
from build_master import build_masterBESS
#from build_masterV2 import build_master

from GlobalData import GlobalData

[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

# Initialize parameters
ParkNo = len(parking_to_bus)
sb = 10  # Scaling factor
PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
PFV_Rob = (1 / 365) * ((IR * (1 + IR) ** NYearRob) / (-1 + (1 + IR) ** NYearRob))


# Initialize storage dictionaries
results = {
    'P_btot': {}, 'P_b_EV': {}, 'P_dch_rob': {}, 'x': {}, 'y': {},
    'SOC_rob': {}, 'z': {}, 'assign': {}, 'assignRobot': {}, 'u_rob': {},
    'u_rob_type': {}, 'Ns': {}, 'P_ch_EV': {}, 'P_ch_rob': {}, 'CapRobot': {},
    'PeakPower': {}, 'Alpha': {}
}


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
Alphaf ={}
RobAssToCh_f = {}
x_rob_f = {}
P_ch_robft = {}
P_ch_robSumT = {}
P_dch_robf2 = {}

Out1_P_b_EVf = {}
# Load data
current_directory = os.path.dirname(__file__)
current_directory = os.getcwd()
file_path         = os.path.join(current_directory, 'data.xlsx')

file_path3 = os.path.join(current_directory, 'day2PublicWork.xlsx')
df = pd.read_excel(file_path3, sheet_name='Sheet1')
#parking_data = DataCuration(df, SampPerH, ChargerCap, ParkNo)  # Your data processing function
#parking_data = pd.read_excel(file_path3, sheet_name='clustered2')
parking_data = pd.read_excel(file_path3, sheet_name='clustered30min')
Price = pd.read_excel(file_path, sheet_name='electricicty_price')
Pattern = pd.read_excel(file_path, sheet_name='DemandPattern')
Pattern = np.repeat(Pattern['Pattern'], SampPerH)
Price = np.repeat(Price['Price'], SampPerH)
 

#df = pd.read_excel(file_path3, sheet_name='Sheet1')
 

# Initialize models for each parking
parking_models = {}
for s in range(1, ParkNo+1):
    parking_models[s] = build_masterBESS(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price) 

# Benders loop
converged = False
kl = 0
max_iter = 6
volt_per_node = np.ones((33, SampPerH*24))  # Initialize with feasible voltages

while True:
    kl += 1
    print(f"\n=== Iteration {kl} ===")
    
    # Step 1: Solve all parking masters
    P_btot_current = {}
    P_b_EVf = {}
    for s, model in parking_models.items():
        print(f"\nSolving parking {s} master...")
        solver = pyo.SolverFactory('gurobi')
        if s == 1 or s == 3:
            gap = 0.05
        elif s== 2: 
            gap == 0.02
        solver.options = {
            'Presolve': 2,          # Aggressive presolve
            'MIPGap': 0.03,         # Accept 5% gap early
            'Heuristics': 0.8,      # More time on heuristics
            'Cuts': 2,              # Maximum cut generation
            'Threads': 8,           # Full CPU utilization
            'NodeMethod': 2,        # Strong branching
        #    'SolutionLimit': 1      # Stop at first feasible (if applicable)
        }
        results = solver.solve(model, tee=True)
        
        # Store results
        for t in model.T:
            P_btot_current[(s,t)] = pyo.value(model.P_btot[t])
#            results['P_btot'][(s,t)] = pyo.value(model.P_btot[t])
        
        print(f"Parking {s} Alpha: {pyo.value(model.Alpha):.2f}")
        
        for t in model.T:
            P_btot_Parkings[(s, t)] = pyo.value(model.P_btot[t])

#        P_b_EVf.update({
#            (k,i,s,t): pyo.value(model.P_b_EV[k,i,t]) 
#            for (k,i,t) in model.x_indices
#        })
        for t in model.T:
            Out1_P_b_EVf[(s,t)] = pyo.value(model.Out1_P_b_EV[t])
        
        
        
        for (k, i, t) in model.x_indices:
            P_b_EVf[(s,k, i, t)] = pyo.value(model.P_b_EV[k, i, t])
        print('P_b_EVTest',s, sum(pyo.value(model.P_b_EV[k, i, t]) for k in model.K for i in model.I if (k, i, t) in model.x_indices))    

        
        
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
                assignRobotf[(k,j,s)] = pyo.value(model.assignRobot[k, j])
                    
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

        
            print(f"Parking {s} Alpha: {pyo.value(model.Alpha):.2f}")
        PeakPowerf[(s)] = pyo.value(model.PeakPower)
        Alphaf[(s)] = pyo.value(model.Alpha)
        
                
                
                
               
    # Step 2: Solve global power flow
    print("\nSolving power flow subproblem...")
    min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj = PowerFlow(
        P_btot_current, np.repeat(Pattern, SampPerH), np.repeat(Price, SampPerH))
    if np.all(volt_per_node >= Vmin - 1e-3) or kl> max_iter:
        print("Voltage constraints satisfied. Optimal solution found.")
        break
    # Step 3: Add cuts and check convergence
    #converged = True
    for s, model in parking_models.items():
        bus = parking_to_bus[s]
        #has_violation = False
        
        for t in model.T:
            if volt_per_node[bus-1, t-1] < Vmin - 1e-3:
                #converged = False
                #has_violation = True
                violation = Vmin - volt_per_node[bus-1, t-1]
                
                # Add parking-specific cut
                model.cuts.add(
                    SPobj + (
                        -(duals_DevLow.get((bus,t), 0) + 
                          duals_balance_p.get((bus,t), 0) + 
                          duals_DevUp.get((bus,t), 0)) * 100 * violation *
                        (model.P_btot[t] - 0*P_btot_current[(s,t)])/sb
                    ) <= model.Alpha
                )
                print(f"Added cut for parking {s} (bus {bus}) at t={t}")

        # Update lower bound
        current_alpha = pyo.value(model.Alpha)
        if current_alpha > model.AlphaDown.value:
            model.AlphaDown.value = current_alpha
            print(f"Updated AlphaDown[{s}] = {current_alpha:.2f}")

    print(f"\nVoltage status: Min={np.min(volt_per_node):.4f}")
    if converged:
        print("\n*** All voltages feasible - convergence achieved! ***")
        break

# Post-processing
        

# Visualization
Ncharger_val = {(s): Nsf[s] for s in range(1, ParkNo+1)}
print(Ncharger_val)
# x_val = {(k,i,t): sum( pyo.value(model.x[k,i,s,t]) for s in model.S) for k in model.K for i in model.I for t in model.T}
#assign_val = {(k, i): sum(assignf[k, i, s] for s in model.S) for k in model.K for i in model.I}
P_ch_rob_val = {(j, t): sum(P_ch_robf[j, s, t] for s in range(1, ParkNo+1)) for j in model.J for t in model.T}
P_Dch_rob_val = {(j, t): sum(P_ch_robf[j, s, t] for s in range(1, ParkNo+1)) for j in model.J for t in model.T}

CapRobot_val = {(j, s, kk): CapRobotf[j, s, kk] * u_rob_typef[j, s, kk] for j in model.J
                for s in range(1, ParkNo+1) for kk in model.KK}
CapRobot_val2 = {(j, s, kk): RobotTypes[kk - 1] * u_rob_typef[j, s, kk] for j in model.J for s in range(1, ParkNo+1) for kk in model.KK}



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


# Calculate total energy per parking
P_dch_sums = []
P_b_EV_sums = []
parking_labels = [f'Parking {s}' for s in range(1, ParkNo+1)]

for s in range(1, ParkNo+1):
    # Sum discharge energy for robots (only for current parking)
    P_dch_sum = sum(v for (k,j,sp,t), v in P_dch_robf.items() if sp == s)
    # Sum grid energy for fixed chargers (only for current parking)
    P_b_EV_sum = sum(v for (sp, t), v in Out1_P_b_EVf.items() if sp == s)
    
    P_dch_sums.append(P_dch_sum)
    P_b_EV_sums.append(P_b_EV_sum)

# Plot settings
x = np.arange(len(parking_labels))  # label locations
width = 0.35  # width of the bars
colors = {'FC': '#1f77b4', 'MCR': '#ff7f0e'}  # Consistent colors

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
rects1 = ax.bar(x - width/2, P_b_EV_sums, width, label='Fixed Chargers (FC)', color=colors['FC'])
rects2 = ax.bar(x + width/2, P_dch_sums, width, label='Mobile Robot Chargers (MCR)', color=colors['MCR'])

# Formatting
ax.set_xlabel('Parking Lot', fontsize=12)
ax.set_ylabel('Energy (kWh)', fontsize=12)
#ax.set_title('Energy Contribution by Source per Parking Lot', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(parking_labels, fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend(fontsize=11, framealpha=0.9)

# Function to add value and percentage on top of bars
def autolabel_percentage(rects, other_rects):
    for rect, other_rect in zip(rects, other_rects):
        height = rect.get_height()
        total = height + other_rect.get_height()
        pct = 100 * height / total if total != 0 else 0
        
        # Position text slightly above the bar
        y_pos = height + (0.02 * max(P_b_EV_sums + P_dch_sums))
        
        ax.text(rect.get_x() + rect.get_width()/2, y_pos,
                f'{height:.1f} kWh\n({pct:.0f}%)',
                ha='center', va='bottom',
                fontsize=10)

# Add labels to both sets of bars
autolabel_percentage(rects1, rects2)
autolabel_percentage(rects2, rects1)

# Adjust y-axis limit to accommodate labels
ax.set_ylim(0, max(P_b_EV_sums + P_dch_sums) * 1.25)

plt.tight_layout()
plt.savefig('Energy_Source_Comparison.png', dpi=300, bbox_inches='tight')
plt.show(block=False)
#########################################
for s in range(1, ParkNo+1):
    # Check discharge activity
    has_discharge = any(
        P_dch_robf.get((k,j,s,t), 0) > 0.001 
        for (k,j,t) in model.y_indices
    )
    if not has_discharge:
        continue

    # Find active robots
    active_robots = {
        j for (k,j,t) in model.y_indices 
        if P_dch_robf.get((k,j,s,t), 0) > 0.001
    }

    # Plot each active robot
    fig, axes = plt.subplots(len(active_robots), 1, figsize=(12, 2*len(active_robots)), squeeze=False)
    
    for idx, j in enumerate(active_robots):
        ax = axes[idx, 0]
        robot_capacity = sum(CapRobotf.get((j,s,kk), 0) for kk in model.KK)
        soc = [SOC_robf.get((j,s,t), 0) for t in model.T]
        charge = [P_ch_robf.get((j,s,t), 0) for t in model.T]
        
        # FIXED: Calculate discharge per robot j at time t
        discharge = [
            sum(
                P_dch_robf.get((k, j, s, t), 0) 
                for k in model.K 
                if (k, j, t) in model.y_indices
            )
            for t in model.T
        ]

        ax.plot(model.T, soc, 'b-', label='SOC (kWh)')
        ax.bar(model.T, charge, color='g', alpha=0.3, label='Charging')
        ax.bar(model.T, discharge, bottom=charge, color='r', alpha=0.3, label='Discharging')
        
        ax.set_title(f'Robot #{j} in Parking {s}')
        ax.grid(True)
        
        ax.legend()
        ax.set_xlabel('Time period')
        ax.set_ylabel('Power (kW)')

    plt.tight_layout()
    plt.savefig(f'Robot_Discharge_Parking_{s}.png', dpi=300)
# Prepare SOC matrix HEAT MAP



soc_df = pd.DataFrame({k: [pyo.value(model.SOC_EV[k, t]) for t in model.T]
                       for k in model.K})



time = range(1, SampPerH * 24 + 1)  # 1-24 hours

for s in range(1, ParkNo+1):
    plt.figure(figsize=(12, 8))  # Larger figure for better readability
    
    # --- Grid Charging Plot ---
    plt.subplot(2, 1, 1)
    # Get grid charging from P_b_EVf dictionary
    P_b_EV_total = [
        
            Out1_P_b_EVf.get((s, t), 0)  # Using .get() with default 0
        
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
    
    # Initialize discharge sum per time step
    P_dch_rob_total = [0.0] * len(model.T)
    
    # Sum discharge for ALL robots in this parking
    for t in model.T:
        for (k, j, t_idx) in model.y_indices:
            if t_idx == t:  # Ensure time matches
                P_dch_rob_total[t-1] += P_dch_robf.get((k, j, s, t), 0)
    
    plt.plot(time, P_dch_rob_total, 'r-', linewidth=2, label='Total Robot Discharge')
    plt.fill_between(time, P_dch_rob_total, color='red', alpha=0.1)
    plt.title(f'Parking {s} - Summed Robot Discharge (Max: {max(P_dch_rob_total):.1f} kW)')
    plt.xlabel('Time period')
    plt.ylabel('Power (kW)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'Parking_{s}_Summed_Discharge.png', dpi=300)
    plt.show(block=False)

    # Print statistics
    print(f"\nParking {s} Summary:")
    print(f"Max Grid Charging: {max(P_b_EV_total):.1f} kW")
    print(f"Max Summed Robot Discharge: {max(P_dch_rob_total):.1f} kW")
    print(f"Total Robot Discharge: {sum(P_dch_rob_total):.1f} kWh")
    
    
    
    
    
# Electricity price plot (unchanged as it doesn't use model variables)
plt.figure(figsize=(10, 5))
plt.plot(time, 10*Price, color='red', marker='*')
plt.xlabel('Time period', fontsize=14)
plt.ylabel('Price (SEK/MWh)', fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.savefig('InputPrice.png', dpi=600)
plt.show(block=False)




plt.figure(figsize=(10, 8))  # Slightly taller figure for two subplots

# --- Overall Purchased Electricity ---
plt.subplot(2, 1, 1)
time = range(1, SampPerH * 24 + 1)
total_power = [
    sum(P_btot_Parkings.get((s,t), 0) for s in range(1, ParkNo+1))  # Sum across all parkings
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
for s in range(1, ParkNo+1):
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
 
plt.show(block=False)

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
TotalChargerCost = sum(Ch_cost * Nsf[s] for s in range(1, ParkNo+1))
print("Total Charger Cost = ", TotalChargerCost)

TotalPurchaseCost = sum(Price.iloc[t - 1] * (0.001) * P_btot_Parkings[(s,t)] 
                   for s in range(1, ParkNo+1) for t in model.T)
print("Total cost of electricity purchased = ", TotalPurchaseCost)

TotalPeakCost = (1 / 30) * PeakPrice * sum(PeakPowerf[s] for s in range(1, ParkNo+1))
print("Total cost of Peak Power = ", TotalPeakCost)

TotalMCRcost = sum(u_rob_typef[(j,s,kk)] * robotCC[kk - 1] 
               for j in model.J for s in range(1, ParkNo+1) for kk in model.KK)
print("Total cost of MCRs = ", TotalMCRcost)

TotalCost = (TotalChargerCost + TotalMCRcost + TotalPurchaseCost + TotalPeakCost)
print("Total Cost = ", TotalCost)


ObjectiveFun = (PFV_Charger*TotalChargerCost + PFV_Rob*TotalMCRcost + (1/SampPerH)*TotalPurchaseCost + TotalPeakCost)
print("Objective function = ", ObjectiveFun)

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
    plt.show(block=False)



 

###############################
######################### SAVING VARS FOR FUTURE NEEDS #########################
MyresultsMMBESS = {
    # Power variables
    'P_b_EV': P_b_EVf,          # Dict: (k,i,s,t) → value
    'Out1_P_b_EV': Out1_P_b_EVf, # Dict: (s,t) → value
    'P_dch_rob': P_dch_robf,    # Dict: (k,j,s,t) → value
    'x': xf,                    # Dict: (k,i,s,t) → value
    'y': yf,                    # Dict: (k,j,s,t) → value
    'SOC_rob': SOC_robf,        # Dict: (j,s,t) → value
    'z': zf,                    # Dict: (i,s) → value
    'assign': assignf,          # Dict: (k,i,s) → value
    'assignRobot': assignRobotf, # Dict: (k,j,s) → value
    'u_rob': u_robf,            # Dict: (j,s,t) → value
    'u_rob_type': u_rob_typef,  # Dict: (j,s,kk) → value
    'Ns': Nsf,                  # Dict: (s) → value
    'P_btot': P_btot_Parkings,  # Dict: (s,t) → value
    'P_ch_EV': P_ch_EVf,        # Dict: (k,s,t) → value
    'P_ch_rob': P_ch_robf,      # Dict: (j,s,t) → value
    'CapRobot': CapRobotf,      # Dict: (j,s,kk) → value
    'PeakPower': PeakPowerf,    # Dict: (s) → value
    'Alpha': Alphaf  # Single value
    
}

import pickle

with open('MyresultsMMBESS.pkl', 'wb') as f:
    pickle.dump(MyresultsMMBESS, f)
    
    
    
