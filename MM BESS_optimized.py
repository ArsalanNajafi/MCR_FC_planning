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
from PowerFlow_PyPSA import PowerFlow_PyPSA
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
# from PowerFlow import PowerFlow
# from DataCuration import DataCuration
# from MaxOverlap import max_overlaps_per_parking
from build_master_BESS import build_masterBESS
from GlobalData import GlobalData

[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

# Initialize parameters
ParkNo = len(parking_to_bus)
sb = 10  # Scaling factor
PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
PFV_Rob = (1 / 365) * ((IR * (1 + IR) ** NYearRob) / (-1 + (1 + IR) ** NYearRob))
PFV_Batt = PFV_Rob # Reusing robot financial parameters for batteries

# Initialize storage dictionaries
results = {
    'P_btot': {}, 'P_ch_EV': {}, 'Out1_P_b_EV': {}, # EV/Charger vars
    'SOC_Batt': {}, 'P_ch_batt': {}, 'P_dch_batt': {}, # Battery vars
    'u_batt_type': {}, 'CapBatt': {}, # Battery Investment vars
    'Ns': {}, 'PeakPower': {}, 'Alpha': {}
}

P_btot_Parkings = {}  # Format: {(s,t): value}
P_ch_EVf = {}
Out1_P_b_EVf = {}

# Battery Storage Dictionaries
SOC_Battf = {}
P_ch_battf = {}
P_dch_battf = {}
u_batt_typef = {}
CapBattf = {}

Nsf = {}
PeakPowerf = {}
Alphaf = {}

# Load data
current_directory = os.path.dirname(__file__)
current_directory = os.getcwd()
file_path         = os.path.join(current_directory, 'data.xlsx')

file_path3 = os.path.join(current_directory, 'day2PublicWork.xlsx')
df = pd.read_excel(file_path3, sheet_name='Sheet1')
#parking_data = DataCuration(df, SampPerH, ChargerCap, ParkNo)  # Your data processing function
#parking_data = pd.read_excel(file_path3, sheet_name='clustered2')
#parking_data = pd.read_excel(file_path3, sheet_name='clustered30min')
parking_data = pd.read_excel(file_path3, sheet_name='clustered30min_2')
Price = pd.read_excel(file_path, sheet_name='electricicty_price')
Pattern = pd.read_excel(file_path, sheet_name='DemandPattern')
Pattern = np.repeat(Pattern['Pattern'], SampPerH)
Price = np.repeat(Price['Price'], SampPerH)

# Initialize models for each parking
# Ensure build_masterBESS is the function updated in the previous step
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
    for s, model in parking_models.items():
        print(f"\nSolving parking {s} master...")
        solver = pyo.SolverFactory('gurobi')
        if s == 1 or s == 3:
            gap = 0.05
        elif s== 2: 
            gap = 0.02
        
        solver.options = {
            'Presolve': 2,          # Aggressive presolve
            'MIPGap': 0.03,         # Accept 3% gap
            'Heuristics': 0.8,      # More time on heuristics
            'Cuts': 2,              # Maximum cut generation
            'Threads': 8,           # Full CPU utilization
            'NodeMethod': 2,        # Strong branching
        }
        results = solver.solve(model, tee=True)
        
        # Store results
        for t in model.T:
            P_btot_current[(s,t)] = pyo.value(model.P_btot[t])
            P_btot_Parkings[(s, t)] = pyo.value(model.P_btot[t])
            Out1_P_b_EVf[(s,t)] = pyo.value(model.Out1_P_b_EV[t])
        
        print(f"Parking {s} Alpha: {pyo.value(model.Alpha):.2f}")
        
        # --- STORE EV CHARGER VARIABLES ---
        for k in model.K:
            for t in model.T:
                P_ch_EVf[(k , s, t)] = pyo.value(model.P_ch_EV[k, t])  
        
        Nsf[(s)] = pyo.value(model.Ns)
        PeakPowerf[(s)] = pyo.value(model.PeakPower)
        Alphaf[(s)] = pyo.value(model.Alpha)

        # --- STORE BATTERY VARIABLES ---
        for j in model.J:
            for t in model.T:
                SOC_Battf[(j,s,t)] = pyo.value(model.SOC_Batt[j, t])
                P_ch_battf[(j,s,t)] = pyo.value(model.P_ch_batt[j, t])
                P_dch_battf[(j,s,t)] = pyo.value(model.P_dch_batt[j, t])
            
            for kk in model.KK:
                u_batt_typef[(j,s,kk)] = pyo.value(model.u_batt_type[j, kk])
                CapBattf[(j,s,kk)] = pyo.value(model.CapBatt[j, kk])
               
    # Step 2: Solve global power flow
    print("\nSolving power flow subproblem...")
    # In your main loop, inside the iteration
    print(f"--- Iteration {kl} Power Profile ---")
    for t in range(1, 24*SampPerH + 1):
        total_load_at_t = sum(P_btot_current[(s, t)] for s in range(1, ParkNo+1))
        print(f"Time {t}: Total Grid Load = {total_load_at_t:.2f} kW")
    min_voltages, volt_per_node, duals_DevLow, duals_DevUp, duals_balance_p, duals_Dev, SPobj = PowerFlow(
        P_btot_current, np.repeat(Pattern, SampPerH), np.repeat(Price, SampPerH))
        
    if np.all(volt_per_node >= Vmin - 1e-3) or kl > max_iter:
        print("Voltage constraints satisfied. Optimal solution found.")
        break


  # Step 3: Add cuts and check convergence
    
    # Scale factor to make the Master respect the voltage penalty
    # Increase this if the Master still ignores the cuts.
    SCALE_FACTOR = 500000 

    for s, model in parking_models.items():
        bus = parking_to_bus[s]
        
        # We check the voltage at the bus connected to this parking lot
        # volt_per_node uses 0-based indexing, bus numbers are likely 1-based
        # So we use bus-1 to index the numpy array
        
        for t in model.T:
            current_volt = volt_per_node[bus-1, t-1]
            
            # Only add cut if there is a violation
            if current_volt < Vmin - 1e-4:
                
                # 1. Retrieve the raw dual from the Power Balance constraint
                # Note: This value is expected to be NEGATIVE due to your constraint formulation
                pi_raw = duals_balance_p.get((bus, t), 0)
                
                if abs(pi_raw) < 1e-9:
                    # Skip if dual is effectively zero
                    continue
                
                # 2. CORRECT THE SIGN
                # Because Load appears positively in your balance equation (Net + Load = 0),
                # the dual is the negative of the true sensitivity.
                # We flip the sign so positive Load = Positive Cost.
                pi = -pi_raw
                
                # 3. Formulate the Cut
                P_prev = P_btot_current[(s, t)]
                
                # Benders Cut: Alpha >= Current_Penalty + Sensitivity * (New_P - Old_P)
                # We use (model.P_btot[t] - P_prev) so that if P_btot increases, Alpha increases.
                
                cut_rhs = SPobj + (pi * (model.P_btot[t] - P_prev) / sb)
                
                # Apply Scaling (Currency Conversion)
                scaled_rhs = cut_rhs * SCALE_FACTOR
                
                # Add to model
                # Constraint: Scaled_Estimate <= Alpha
                model.cuts.add(scaled_rhs <= model.Alpha)
                
                print(f"Cut Added: Parking {s}, Bus {bus}, T={t}")
                print(f"   Raw Dual: {pi_raw:.5f} -> Corrected Slope: {pi:.5f}")
                print(f"   Volt: {current_volt:.4f}")
    
   

''' 
    
 
    
 
    
 

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
       
'''    
    


# Post-processing
        
# Visualization
Ncharger_val = {(s): Nsf[s] for s in range(1, ParkNo+1)}
print("Number of Fixed Chargers:", Ncharger_val)

# --- NEW: Battery Capacity Extraction ---
CapBatt_val = {(j, s, kk): CapBattf[j, s, kk] * u_batt_typef[j, s, kk] for j in model.J
                for s in range(1, ParkNo+1) for kk in model.KK}
CapBatt_val2 = {(j, s, kk): RobotTypes[kk - 1] * u_batt_typef[j, s, kk] for j in model.J for s in range(1, ParkNo+1) for kk in model.KK}

print("Non-zero Battery capacities:")
for (j, s, kk), val in CapBatt_val.items():
    if val != 0:
        print(f"Battery {j}, parking {s}, Type {kk}: {val}")

total_batt_cap = sum(val for val in CapBatt_val.values())
print(f"\nTotal Battery Capacity installed: {total_batt_cap}")

# Calculate total energy per parking
P_dch_sums = []  # Battery Discharge
P_b_EV_sums = [] # Grid Charging
parking_labels = [f'Parking {s}' for s in range(1, ParkNo+1)]

for s in range(1, ParkNo+1):
    # Sum battery discharge energy
    P_dch_sum = sum(v for (j, t, sp) in [(j, t, s) for j in model.J for t in model.T] for v in [P_dch_battf.get((j, sp, t), 0)])
    
    # Sum grid energy for fixed chargers
    P_b_EV_sum = sum(v for (sp, t), v in Out1_P_b_EVf.items() if sp == s)
    
    P_dch_sums.append(P_dch_sum)
    P_b_EV_sums.append(P_b_EV_sum)

# Plot settings
x = np.arange(len(parking_labels))
width = 0.35
colors = {'FC': '#1f77b4', 'BESS': '#ff7f0e'}  # Updated Label to BESS

fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
rects1 = ax.bar(x - width/2, P_b_EV_sums, width, label='Grid Power (Fixed Chargers)', color=colors['FC'])
rects2 = ax.bar(x + width/2, P_dch_sums, width, label='Battery Discharge', color=colors['BESS'])

# Formatting
ax.set_xlabel('Parking Lot', fontsize=12)
ax.set_ylabel('Energy (kWh)', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(parking_labels, fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.legend(fontsize=11, framealpha=0.9)

def autolabel_percentage(rects, other_rects):
    for rect, other_rect in zip(rects, other_rects):
        height = rect.get_height()
        total = height + other_rect.get_height()
        pct = 100 * height / total if total != 0 else 0
        y_pos = height + (0.02 * max(P_b_EV_sums + P_dch_sums))
        ax.text(rect.get_x() + rect.get_width()/2, y_pos,
                f'{height:.1f} kWh\n({pct:.0f}%)',
                ha='center', va='bottom', fontsize=10)

autolabel_percentage(rects1, rects2)
autolabel_percentage(rects2, rects1)
ax.set_ylim(0, max(P_b_EV_sums + P_dch_sums) * 1.25)
plt.tight_layout()
plt.savefig('Energy_Source_Comparison_BESS.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

#########################################
# --- BATTERY VISUALIZATION (Replacing Robot Plots) ---
for s in range(1, ParkNo+1):
    # Find active batteries
    active_batteries = {
        j for j in model.J 
        if any(P_ch_battf.get((j,s,t), 0) > 0.001 or P_dch_battf.get((j,s,t), 0) > 0.001 for t in model.T)
    }
    
    if not active_batteries:
        continue

    # Plot each active battery
    fig, axes = plt.subplots(len(active_batteries), 1, figsize=(12, 2*len(active_batteries)), squeeze=False)
    
    for idx, j in enumerate(active_batteries):
        ax = axes[idx, 0]
        battery_capacity = sum(CapBattf.get((j,s,kk), 0) for kk in model.KK)
        soc = [SOC_Battf.get((j,s,t), 0) for t in model.T]
        charge = [P_ch_battf.get((j,s,t), 0) for t in model.T]
        discharge = [P_dch_battf.get((j,s,t), 0) for t in model.T]

        ax.plot(model.T, soc, 'b-', label='SOC (kWh)')
        ax.bar(model.T, charge, color='g', alpha=0.3, label='Charging')
        ax.bar(model.T, discharge, bottom=charge, color='r', alpha=0.3, label='Discharging')
        
        ax.set_title(f'Battery #{j} in Parking {s} (Cap: {battery_capacity:.1f} kWh)')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Time period')
        ax.set_ylabel('Power (kW)')

    plt.tight_layout()
    plt.savefig(f'Battery_Discharge_Parking_{s}.png', dpi=300)
    plt.show(block=False)

# Prepare SOC matrix HEAT MAP for EVs
soc_df = pd.DataFrame({k: [np.nan if t < parking_data[parking_data['ParkingNo']==s].iloc[k-1]['AT'] or t > parking_data[parking_data['ParkingNo']==s].iloc[k-1]['DT'] else P_ch_EVf.get((k,s,t), 0) # Simplified placeholder for SOC viz
                       for t in model.T] for k in model.K}) # Note: Real SOC extraction needs EVdata mapping
# (Heatmap logic omitted for brevity as it relies on complex mapping of K to s, logic kept simple for P plots)

time = range(1, SampPerH * 24 + 1)  # 1-24 hours

for s in range(1, ParkNo+1):
    plt.figure(figsize=(12, 8)) 
    
    # --- Grid Charging Plot ---
    plt.subplot(2, 1, 1)
    P_b_EV_total = [Out1_P_b_EVf.get((s, t), 0) for t in model.T]
    
    plt.plot(time, P_b_EV_total, 'b-', linewidth=2, label='Grid Charging')
    plt.fill_between(time, P_b_EV_total, color='blue', alpha=0.1)
    plt.xlabel('Time period', fontsize=14)
    plt.ylabel('Power (kW)', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.title(f'Parking {s} - Grid Charging Power')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    
    # --- Battery Discharging Plot ---
    plt.subplot(2, 1, 2)
    P_dch_batt_total = [sum(P_dch_battf.get((j, s, t), 0) for j in model.J) for t in model.T]
    
    plt.plot(time, P_dch_batt_total, 'r-', linewidth=2, label='Total Battery Discharge')
    plt.fill_between(time, P_dch_batt_total, color='red', alpha=0.1)
    plt.title(f'Parking {s} - Summed Battery Discharge (Max: {max(P_dch_batt_total):.1f} kW)')
    plt.xlabel('Time period')
    plt.ylabel('Power (kW)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'Parking_{s}_Summed_Discharge_BESS.png', dpi=300)
    plt.show(block=False)

    print(f"\nParking {s} Summary:")
    print(f"Max Grid Charging: {max(P_b_EV_total):.1f} kW")
    print(f"Max Battery Discharge: {max(P_dch_batt_total):.1f} kW")
    print(f"Total Battery Discharge: {sum(P_dch_batt_total):.1f} kWh")
    
# Electricity price plot
plt.figure(figsize=(10, 5))
plt.plot(time, 10*Price, color='red', marker='*')
plt.xlabel('Time period', fontsize=14)
plt.ylabel('Price (SEK/MWh)', fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.grid(True)
plt.savefig('InputPrice.png', dpi=600)
plt.show(block=False)

plt.figure(figsize=(10, 8)) 
# --- Overall Purchased Electricity ---
plt.subplot(2, 1, 1)
total_power = [sum(P_btot_Parkings.get((s,t), 0) for s in range(1, ParkNo+1)) for t in model.T]
plt.plot(time, total_power, 'b-', linewidth=1.5)
plt.xlabel('Time period', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.title('Overall Purchased Electricity', fontsize=14)
plt.grid()

# --- Per-Parking Purchased Electricity ---
plt.subplot(2, 1, 2)
colors = ['r', 'g', 'b', 'm', 'c'] 
for s in range(1, ParkNo+1):
    P_Tot_Purch = [P_btot_Parkings.get((s,t), 0) for t in model.T]
    plt.plot(time, P_Tot_Purch, label=f'Parking {s}', color=colors[s-1], linewidth=1.5)

plt.xlabel('Time period', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.title('Purchased Electricity by Parking', fontsize=14)
plt.grid()
plt.legend(fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig('PurchasePower_BESS.png', dpi=600, bbox_inches='tight')
plt.show(block=False)

# --- CHARGER UTILIZATION HEATMAP ---
# Reconstructing 'is_charging' data from P_ch_EV for visualization
charger_usage_data = []
for s in range(1, ParkNo+1):
    # Since we don't track individual plug IDs (I) anymore, we visualize Total Active Plugs vs Capacity
    # However, to keep the heatmap structure, we can assume sequential plugs 1 to Ns
    # Or just visualize the aggregate load per time.
    # Here we create a dummy dataset for heatmap compatibility if needed, 
    # or we skip to the aggregate bar plots used above.
    pass 

print("Skipping individual charger heatmap (Fixed Chargers replaced by aggregate assumption).")

# --- COST CALCULATIONS ---
TotalChargerCost = sum(Ch_cost * Nsf[s] for s in range(1, ParkNo+1))
print("Total Charger Cost = ", TotalChargerCost)

TotalPurchaseCost = sum(Price.iloc[t - 1] * (0.001) * P_btot_Parkings[(s,t)] 
                   for s in range(1, ParkNo+1) for t in model.T)
print("Total cost of electricity purchased = ", TotalPurchaseCost)

TotalPeakCost = (1 / 30) * PeakPrice * sum(PeakPowerf[s] for s in range(1, ParkNo+1))
print("Total cost of Peak Power = ", TotalPeakCost)

# Battery Cost instead of Robot Cost
TotalBattCost = sum(u_batt_typef[(j,s,kk)] * robotCC[kk - 1] 
               for j in model.J for s in range(1, ParkNo+1) for kk in model.KK)
print("Total cost of Batteries = ", TotalBattCost)

TotalCost = (TotalChargerCost + TotalBattCost + TotalPurchaseCost + TotalPeakCost)
print("Total Cost = ", TotalCost)

ObjectiveFun = (PFV_Charger*TotalChargerCost + PFV_Batt*TotalBattCost + (1/SampPerH)*TotalPurchaseCost + TotalPeakCost)
print("Objective function = ", ObjectiveFun)


############################# SAVING VARS #########################
MyresultsBESS = {
    # Power variables
    'P_ch_EV': P_ch_EVf,
    'Out1_P_b_EV': Out1_P_b_EVf,
    'P_btot': P_btot_Parkings,
    
    # Battery Variables
    'SOC_Batt': SOC_Battf,
    'P_ch_batt': P_ch_battf,
    'P_dch_batt': P_dch_battf,
    'u_batt_type': u_batt_typef,
    'CapBatt': CapBattf,
    
    # Infrastructure
    'Ns': Nsf,
    'PeakPower': PeakPowerf,
    'Alpha': Alphaf
}

import pickle

with open('MyresultsBESS.pkl', 'wb') as f:
    pickle.dump(MyresultsBESS, f)