# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:15:05 2026

@author: arsalann
"""

import pyomo.environ as pyo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PowerFlow import PowerFlow  # Assuming you have this module
import os
import seaborn as sns
#from build_master import build_master
from build_master_MCR_FC import build_master
#from build_masterV5_y_modified import build_master


from GlobalData import GlobalData

[parking_to_bus, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

# Initialize parameters
ParkNo = len(parking_to_bus)
sb = 10  # Scaling factor
PFV_Charger = (1 / 365) * ((IR * (1 + IR) ** NYearCh) / (-1 + (1 + IR) ** NYearCh))
PFV_Rob = (1 / 365) * ((IR * (1 + IR) ** NYearRob) / (-1 + (1 + IR) ** NYearRob))


# Initialize storage dictionaries
results = {
    'grid_charge': {}, 'assignRobot': {}, 'u_rob': {}, 'P_ch_rob': {}, 'Ns': {},
    'P_b_EV_grid': {}, 'z': {}, 'Out1_P_b_EV': {}, 'P_ch_rob_total': {}, 'P_dch_rob': {},
    'P_ch_EV': {}, 'SOC_EV': {}, 'SOC_rob': {}, 'CapRobot': {}, 'PeakPower': {}, 'Alpha': {},
    'u_rob_type': {}, 'P_b_EV_grid':{}, 'y':{}
}



P_btot_Parkings = {}  # Format: {(s,t): value}
grid_chargef = {}
assignRobotf = {}
u_robf = {}
u_rob_typef = {}
P_ch_robf = {}
Nsf = {}
P_b_EV_gridf = {}
Out1_P_b_EVf = {}
P_ch_rob_totalf = {}
P_dch_robf = {}
P_ch_EVf = {}
SOC_EVf = {}
SOC_robf = {}
CapRobotf ={}
PeakPowerf = {}
Alphaf = {}
P_b_EV_gridf = {}
yf = {}


# Load data
current_directory = os.path.dirname(__file__)
current_directory = os.getcwd()
file_path         = os.path.join(current_directory, 'data.xlsx')

file_path3 = os.path.join(current_directory, 'day2PublicWork.xlsx')
df = pd.read_excel(file_path3, sheet_name='Sheet1')
#parking_data = DataCuration(df, SampPerH, ChargerCap, ParkNo)  # Your data processing function
#parking_data = pd.read_excel(file_path3, sheet_name='clustered2')
parking_data = pd.read_excel(file_path3, sheet_name='clustered30min')
###parking_data = pd.read_excel(file_path3, sheet_name='clustered30min_2')
Price = pd.read_excel(file_path, sheet_name='electricicty_price')
Pattern = pd.read_excel(file_path, sheet_name='DemandPattern')
Pattern = np.repeat(Pattern['Pattern'], SampPerH)
Price = np.repeat(Price['Price'], SampPerH)
 

#df = pd.read_excel(file_path3, sheet_name='Sheet1')
 
   
# Ensure your data is loaded and 'all_parkings' is defined as in the previous step
all_parkings = sorted(parking_data['ParkingNo'].unique())

# Setup the figure
plt.figure(figsize=(12, 8))

# --- FIRST ROW: Arrival Times (KDE) ---
plt.subplot(2, 1, 1)
for p in all_parkings:
    subset = parking_data[parking_data['ParkingNo'] == p]
    
    # KDE Plot
    # 'bw_adjust' controls the smoothness (higher = smoother)
    # 'fill=True' adds the semi-transparent color under the curve
    sns.kdeplot(data=subset['AT'], label=f'Parking lot {p}', fill=True, alpha=0.3, bw_adjust=0.5)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylabel('Probability Density')
plt.xlim(1, SampPerH * 24)
plt.title('Arrival Time Distributions')

# --- SECOND ROW: Departure Times (KDE) ---
plt.subplot(2, 1, 2)
for p in all_parkings:
    subset = parking_data[parking_data['ParkingNo'] == p]
    
    sns.kdeplot(data=subset['DT'], label=f'Parking lot {p}', fill=True, alpha=0.3, bw_adjust=0.5)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Time sample')
plt.ylabel('Probability Density')
plt.xlim(1, SampPerH * 24)
plt.title('Departure Time Distributions')

plt.tight_layout()
plt.savefig('All_Parkings_KDE.png', dpi=300)
plt.show(block=False)

# Initialize the plot
plt.figure(figsize=(12, 8))

import seaborn as sns
import matplotlib.pyplot as plt

# ... (Ensure all_parkings and parking_data are defined) ...

all_parkings = sorted(parking_data['ParkingNo'].unique())

# Setup the figure
plt.figure(figsize=(12, 8))

# --- FIRST ROW: Arrival Times (Scaled to Count) ---
plt.subplot(2, 1, 1)
for p in all_parkings:
    subset = parking_data[parking_data['ParkingNo'] == p]
    
    # histplot with kde=True
    # stat="count": Ensures the Y-axis is the actual number of EVs
    # kde=True: Draws the smooth curve scaled to the histogram
    # alpha=0.2: Makes the bars very transparent so the curve stands out
    sns.histplot(
        data=subset['AT'], 
        kde=True, 
        stat="count", 
        label=f'Parking {p}', 
        alpha=0.2, 
        fill=True,
        line_kws={'linewidth': 3} # Makes the curve line thicker
    )

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylabel('Frequency (Count of EVs)')
plt.xlim(1, SampPerH * 24)
plt.title('Arrival Time Distributions (Frequency Scaled)')

# --- SECOND ROW: Departure Times (Scaled to Count) ---
plt.subplot(2, 1, 2)
for p in all_parkings:
    subset = parking_data[parking_data['ParkingNo'] == p]
    
    sns.histplot(
        data=subset['DT'], 
        kde=True, 
        stat="count", 
        label=f'Parking {p}', 
        alpha=0.2, 
        fill=True,
        line_kws={'linewidth': 3}
    )

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Time sample')
plt.ylabel('Frequency (Count of EVs)')
plt.xlim(1, SampPerH * 24)
plt.title('Departure Time Distributions (Frequency Scaled)')

plt.tight_layout()
plt.savefig('All_Parkings_Frequency_Curves.png', dpi=300)
plt.show(block=False)








# Setup the figure
plt.figure(figsize=(12, 8))
my_colors = ['yellow', 'green', 'red']
# --- FIRST ROW: Arrival Times (Side-by-Side Bars) ---
plt.subplot(2, 1, 1)

# multiple="dodge" creates the side-by-side grouping
# shrink=0.8 adds a little gap between bars within a group so they are distinct
sns.histplot(
    data=parking_data, 
    x='AT', 
    hue='ParkingNo', 
    multiple='dodge', 
    bins=SampPerH * 24, 
    shrink=0.8,
    edgecolor = 'black',
    palette = my_colors,
    label=f'Parking lot {p}', 
    linewidth=0.5
)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylabel('Frequency (Count)')
plt.xlim(1, SampPerH * 24)
plt.title('Arrival Times')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# --- SECOND ROW: Departure Times (Side-by-Side Bars) ---
plt.subplot(2, 1, 2)

sns.histplot(
    data=parking_data, 
    x='DT', 
    hue='ParkingNo', 
    multiple='dodge', 
    bins=SampPerH * 24, 
    shrink=0.8,
    edgecolor= 'black',
    palette=my_colors,
    label=f'Parking lot {p}', 
    linewidth=0.5
)

plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Time sample', fontsize=14)
plt.ylabel('Frequency (Count)', fontsize=14)
plt.xlim(1, SampPerH * 24)
plt.title('Departure Times')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig('Side_by_Side_Histogram.png', dpi=300)
plt.show(block=False)
# Initialize models for each parking
parking_models = {}
for s in range(1, ParkNo+1):
    
    
    parking_models[s] = build_master(s, parking_data, parking_to_bus, SampPerH, Ch_cost, robotCC, Pattern, Price) 

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
        gapp = 0.05
        if s == 2:
            gapp = 0.05
        print(f"\nSolving parking {s} master...")
        solver = pyo.SolverFactory('gurobi')
        solver.options = {
#            'Presolve': 2,          # Aggressive presolve
            'MIPGap': gapp,         # Accept 5% gap early
            'Heuristics': 0.3,      # time on heuristics
#            'Cuts': 2,              # Maximum cut generation
            'NodeLimit': 500,
            'CutPasses': 10,
#            'MIRCuts': 2,
            'Threads': 8,           # Full CPU utilization
            'MIPFocus':2
#            'NodeMethod': 2,        # Strong branching 2
            #'MIPFocus': 1,          # # prioritize feasible solutions    
            #'Symmetry': 1,
            #'Aggregate': 2,         # eliminate redundant constraints
            #'RINS': 20              # Helps improve incumbent solutions quickly
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


        for t in model.T:
            Out1_P_b_EVf[(s,t)] = pyo.value(model.Out1_P_b_EV[t])




        # Add this after solving:
        for k in model.K:
            for t in model.T:
                val = pyo.value(model.P_b_EV_grid[k, t])
                if val > 1e-6:
                    P_b_EV_gridf[(k, s, t)] = val



        for k in model.K:
            grid_chargef[(k,s)] = pyo.value(model.grid_charge[k]) 
        
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
                
        yf.update( {
            (k, j , s, t):  pyo.value(model.y_link[k, j, t]) 
            for (k, j, t) in model.y_indices  # Uses predefined sparse indices
        }) 
                
        for k in model.K:
            for t in model.T:
                 P_ch_EVf[(k , s, t)] =   pyo.value(model.P_ch_EV[k, t])  
                 
        for k in model.K:
            for t in model.T:
                SOC_EVf[(k , s, t)] =   pyo.value(model.SOC_EV[k, t])          
            
        for j in model.J:
             for kk in model.KK:
                 CapRobotf[(j,s,kk)] =   pyo.value(model.CapRobot[j, kk])    
            
        for j in model.J:
            for t in model.T:
                SOC_robf[(j,s,t)] =   pyo.value(model.SOC_rob[j, t]) 
            
     
        for j in model.J:
            for t in model.T:
                P_ch_robf[(j,s,t)] =   pyo.value(model.P_ch_rob_total[j, t])  
         
        for j in model.J:
            for t in model.T:
                P_ch_rob_totalf[(j,s,t)] =   pyo.value(model.P_ch_rob_total[j, t])    
         
                        
          
                    
        P_dch_robf.update({
              (k, j, s, t): pyo.value(model.P_dch_rob[k, j, t])
              for (k, j, t) in model.y_indices  # Uses predefined sparse indices
          })  
                 
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
    



# =====================================================================
# POST-PROCESSING: Reconstruct physical x, y, x_rob for visualization
# =====================================================================
print("\nReconstructing physical charger assignments for visualization...")

xf = {}

zf = {}
x_rob_f = {}
assignf = {}
P_ch_robft = {}
P_ch_robSumT = {}
P_dch_robf2 = {}

df_x_data = []
df_y_data = []

for s in range(1, ParkNo+1):
    model = parking_models[s]
    EVdata_s = parking_data[parking_data['ParkingNo'] == s].reset_index(drop=True)
    Ns_val = int(Nsf[s])
    
    # Reconstruct z (which chargers are installed)
    for i in range(1, Ns_val + 1):
        zf[(i, s)] = 1.0
        
    # Reconstruct assign (arbitrary sequential mapping for plotting)
    ev_counter = 1
    for k in model.K:
        if grid_chargef.get((k, s), 0) > 0.5:
            assignf[(k, ev_counter, s)] = 1.0
            ev_counter += 1
            
    #Reconstruct x (EV physically on charger i at time t)
    for t in model.T:
        available_chargers = list(range(1, Ns_val + 1))
        for k in model.K:
            if grid_chargef.get((k, s), 0) > 0.5:
                if EVdata_s['AT'][k] <= t <= EVdata_s['DT'][k]:
                    if available_chargers:
                        i_assign = available_chargers.pop(0)
                        xf[(k, i_assign, s, t)] = 1.0
                        df_x_data.append({'EV': k, 'Charger': i_assign, 'Parking': s, 'Time': t, 'Value': 1.0})


    for t in model.T:
        # Remove chargers already used by EVs at this time to prevent visual overlap
        ev_used_chargers = set([key[1] for key in xf.keys() if len(key)==4 and key[3] == t and key[2] == s])
        available_chargers_rob = [i for i in range(1, Ns_val + 1) if i not in ev_used_chargers]
        
        for j in model.J:
            if u_robf.get((j, s, t), 0) > 0.5:
                if available_chargers_rob:
                    i_assign = available_chargers_rob.pop(0)
                    x_rob_f[(j, i_assign, t, s)] = 1.0
                    # Map total robot power to this specific assigned charger
                    P_ch_robft[(j, i_assign, t, s)] = P_ch_rob_totalf.get((j, s, t), 0)
                    
                    # For the Robot to Charger heatmap sum
                    if (j, i_assign, s) not in P_ch_robSumT:
                        P_ch_robSumT[(j, i_assign, s)] = 0.0
                    P_ch_robSumT[(j, i_assign, s)] += P_ch_rob_totalf.get((j, s, t), 0)

    # 6. Reconstruct P_dch_robf2 (Summed discharge per robot per time)
    for t in model.T:
        for j in model.J:
            total_dch = sum(P_dch_robf.get((k, j, s, t), 0) for k in model.K)
            if total_dch > 0.001:
                P_dch_robf2[(j, t, s)] = total_dch

# Create the DataFrames expected by your heatmap plotting code
df_x = pd.DataFrame(df_x_data)
df_y = pd.DataFrame(df_y_data)







# Visualization
Ncharger_val = {(s): Nsf[s] for s in range(1, ParkNo+1)}
print(Ncharger_val)

P_ch_rob_val = {(j, t): sum(P_ch_robf[j, s, t] for s in range(1, ParkNo+1)) for j in model.J for t in model.T}
P_Dch_rob_val = {
    (j, t): sum(
        P_dch_robf.get((k, j, s, t), 0) 
        for k in model.K 
        for s in range(1, ParkNo+1)
    ) 
    for j in model.J 
    for t in model.T
}
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
    # Check discharge activity using the stored dictionary
    has_discharge = any(
        val > 0.001 
        for (k, j, s_key, t), val in P_dch_robf.items() 
        if s_key == s
    )
    
    if not has_discharge:
        print(f"No discharge activity in Parking {s}")
        continue

    # Find active robots in this parking
    active_robots = set()
    for (k, j, s_key, t), val in P_dch_robf.items():
        if s_key == s and val > 0.001:
            active_robots.add(j)
    
    active_robots = sorted(active_robots)  # Sort for consistent ordering
    
    if not active_robots:
        continue
    
    print(f"Parking {s}: Active robots {active_robots}")

    # Plot each active robot
    fig, axes = plt.subplots(len(active_robots), 1, figsize=(12, 2*len(active_robots)), squeeze=False)
    
    for idx, j in enumerate(active_robots):
        ax = axes[idx, 0]
        
        # Robot capacity (sum over all robot types)
        robot_capacity = sum(CapRobotf.get((j, s, kk), 0) for kk in range(1, len(robotCC)+1))
        
        # SOC over time
        soc = [SOC_robf.get((j, s, t), 0) for t in range(1, SampPerH*24 + 1)]
        
        # Charging power from grid
        charge = [P_ch_rob_totalf.get((j, s, t), 0) for t in range(1, SampPerH*24 + 1)]
        
        # Discharge power to EVs (sum over all EVs)
        discharge = []
        for t in range(1, SampPerH*24 + 1):
            total_discharge = sum(
                val for (k, j_key, s_key, t_key), val in P_dch_robf.items()
                if j_key == j and s_key == s and t_key == t
            )
            discharge.append(total_discharge)
        
        # Plot
        time = range(1, SampPerH*24 + 1)
        
        ax.plot(time, soc, 'b-', linewidth=2, label='SOC (kWh)')
        ax.bar(time, charge, color='green', alpha=0.4, label='Charging from Grid', width=0.8)
        ax.bar(time, discharge, bottom=charge, color='red', alpha=0.4, label='Discharging to EVs', width=0.8)
        
        # Add capacity line
        ax.axhline(y=robot_capacity, color='b', linestyle='--', alpha=0.5, label=f'Capacity: {robot_capacity:.1f} kWh')
        
        ax.set_title(f'Robot #{j} in Parking {s}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xlabel('Time period (30-min intervals)', fontsize=10)
        ax.set_ylabel('Power (kW) / SOC (kWh)', fontsize=10)
        
        # Add some statistics
        total_charged = sum(charge)
        total_discharged = sum(discharge)
        ax.text(0.02, 0.95, f'Charged: {total_charged:.1f} kWh | Discharged: {total_discharged:.1f} kWh',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'Robot_Discharge_Parking_{s}.png', dpi=300, bbox_inches='tight')
# Prepare SOC matrix HEAT MAP
#plt.show(block=False)


#soc_df = pd.DataFrame({k: [pyo.value(model.SOC_EV[k, t]) for t in model.T]
#                       for k in model.K})



time = range(1, SampPerH * 24 + 1)  # 1-24 hours

for s in range(1, ParkNo+1):
    plt.figure(figsize=(12, 8))  # Larger figure for better readability
    
    time = range(1, SampPerH * 24 + 1)
    
    # --- Grid Charging Plot ---
    plt.subplot(2, 1, 1)
    # Get grid charging from Out1_P_b_EVf dictionary
    P_b_EV_total = [
        Out1_P_b_EVf.get((s, t), 0)  # Using .get() with default 0
        for t in time
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
    P_dch_rob_total = [0.0] * len(time)
    
    # Sum discharge for ALL robots in this parking
    # Iterate directly over P_dch_robf dictionary
    for (k, j, s_key, t), val in P_dch_robf.items():
        if s_key == s:  # Only this parking
            if 1 <= t <= len(time):  # Ensure time is in range
                P_dch_rob_total[t-1] += val
    
    plt.plot(time, P_dch_rob_total, 'r-', linewidth=2, label='Total Robot Discharge')
    plt.fill_between(time, P_dch_rob_total, color='red', alpha=0.1)
    
    max_discharge = max(P_dch_rob_total) if P_dch_rob_total else 0
    plt.title(f'Parking {s} - Summed Robot Discharge (Max: {max_discharge:.1f} kW)')
    plt.xlabel('Time period', fontsize=14)
    plt.ylabel('Power (kW)', fontsize=14)
    plt.tick_params(axis='both', labelsize=14)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()

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
plt.bar(time, 10*Price, color='black')
plt.xlabel('Time sample', fontsize=14)
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
plt.plot(time, total_power, 'b-', linewidth=1.5, marker='+')
plt.xlabel('Time sample', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.title('Total Purchased Electricity', fontsize=14)
plt.grid()

# --- Per-Parking Purchased Electricity ---
plt.subplot(2, 1, 2)
colors = ['m', 'g', 'b', 'm', 'c']  # Different colors for each parking
markers = ['^', '*', '+']
for s in range(1, ParkNo+1):
    P_Tot_Purch = [
        P_btot_Parkings.get((s,t), 0)  # Get from saved results
        for t in model.T
    ]
    plt.plot(time, P_Tot_Purch, 
             label=f'Parking {s}',
             color=colors[s-1],  # Use different color for each parking
             linewidth=1.5, marker= markers[s-1])

plt.xlabel('Time sample', fontsize=12)
plt.ylabel('Power (kW)', fontsize=12)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
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
plt.xlabel('Time sample', fontsize=14)
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
    plt.xlabel('Time sample', fontsize=14)
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
            cmap=['white', 'green'],
            linewidths=0.5,
            linecolor='red',
            cbar=False,
            ax=axes[idx, 0]
        )

        axes[idx, 0].set_title(f'Parking {parking} - Robot Utilization', pad=12)
        axes[idx, 0].set_xlabel('Time sample')
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
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.savefig('Combined_Robot_Utilization_Grid.png', 
               dpi=300, 
               bbox_inches='tight')
    plt.show(block=False)


if not valid_parkings:
    print("No robot connected to chargers.")
else:    
    fig, axes = plt.subplots(len(valid_parkings), 1, 
                           figsize=(15, 5 * len(valid_parkings)),
                           squeeze=False)
    
    plt.subplots_adjust(hspace=0.4)
    
    # Create heatmap for each parking
    for idx, parking in enumerate(valid_parkings):
            # Extract all unique robot IDs and charger IDs for this parking
            robots = sorted(set(j for (j, i, s) in P_ch_robSumT.keys() if s == parking))
            chargers = sorted(set(i for (j, i, s) in P_ch_robSumT.keys() if s == parking))
            
            # Create empty matrix (robots x chargers)
            heatmap_matrix = np.zeros((len(robots), len(chargers)))
            
            # Fill the matrix with values from dictionary
            for j_idx, robot in enumerate(robots):
                for i_idx, charger in enumerate(chargers):
                    # Get value from dictionary, default to 0 if not found
                    heatmap_matrix[j_idx, i_idx] = P_ch_robSumT.get((robot, charger, parking), 0)
            
            sns.heatmap(
                        heatmap_matrix,
                        cmap='YlOrRd',  # Color gradient (yellow to orange to red)
                        linewidths=0.5,
                        linecolor='black',
                        cbar=True,  # Show color bar for values
                        ax=axes[idx, 0],
                        annot=False,  # Show values in cells
                        fmt='.2f',   # Format numbers with 2 decimals
                        center=None,  # No center for continuous data
                        vmin=0,      # Minimum value for color scale
                        vmax=np.max(heatmap_matrix) if np.max(heatmap_matrix) > 0 else 1  # Auto-scale max
                    )
            axes[idx, 0].set_title(f'Parking {parking} - Robot to Charger', pad=12)
            axes[idx, 0].set_xlabel('Charger ID')
            axes[idx, 0].set_ylabel('Robot ID')
            
            # Set tick labels with actual IDs
            axes[idx, 0].set_xticks(np.arange(len(chargers)) + 0.5)
            axes[idx, 0].set_xticklabels(chargers, rotation=0)
            axes[idx, 0].set_yticks(np.arange(len(robots)) + 0.5)
            axes[idx, 0].set_yticklabels(robots, rotation=0)
    
            for spine in axes[idx, 0].spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(1.5)
    plt.tight_layout()
    plt.savefig('Robot_To_ChargerMap.png', 
               dpi=300, 
               bbox_inches='tight')
    plt.show(block=False) 
# Create heatmap for each parking
      
                
                
###############################
######################### SAVING VARS FOR FUTURE NEEDS #########################


MyresultsMMv2 = {
    # Power variables
    
    'P_btot_Parkings': P_btot_Parkings,        # Dict: (s,t) → value
    'grid_charge': grid_chargef,               # Dict: (k,s) → value
    'assignRobot' :assignRobotf,               # Dict: (k,j,s) → value
    'u_rob' : u_robf,                          # Dict: (j,s,t) → value  
    'u_rob_type': u_rob_typef,                 # Dict: (j,s,kk) → value
    'P_ch_rob': P_ch_robf,                     # Dict: (j,s,t) → value
    'Ns': Nsf,                                 # Dict: (s) → value  
    'P_b_EV_grid' : P_b_EV_gridf,              # Dict: (k, s,t) → value
    'Out1_P_b_EV': Out1_P_b_EVf,               # Dict: (s,t) → value
    'P_ch_rob_total' :  P_ch_rob_totalf,       # Dict: (j,s,t) → value
    'P_dch_rob' : P_dch_robf,                  # Dict: (k,j,s,t) → value
    'P_ch_EV' : P_ch_EVf,                      # Dict: (k,s,t) → value
    'SOC_EV': SOC_EVf,                         # Dict: (k,s,t) → value
    'SOC_rob': SOC_robf,                       # Dict: (j,s,t) → value    
    'CapRobot': CapRobotf,                     # Dict: (j,s,kk) → value
    'PeakPower': PeakPowerf,                   # Dict: (s) → value
    'Alpha': Alphaf,                           # Dict: (s) → value
    'x_data': x_data,                          # Dict: (k,i,s,t) → value 
    'y_data': y_data ,                          # Dict: (k,j,s,t) → value 
    'P_ch_robft': P_ch_robft,
    'y': yf
        
}

import pickle

with open('pyomo_resultsMMv2.pkl', 'wb') as f:
    pickle.dump(MyresultsMMv2, f)
    
  
    
##################################################
"""
Enhanced visualization for Robot 1, Parking 3
Shows: Robot discharge to EVs + Robot charging from fixed chargers
"""

"""
Enhanced visualization for Robot 1, Parking 3
Shows: Robot discharge to EVs + Robot charging from fixed chargers
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Load results
with open('pyomo_resultsMMv2.pkl', 'rb') as f:
    results = pickle.load(f)

robot_id = 6
parking_id = 3

# ============================================================================
# 1. EXTRACT DATA
# ============================================================================
discharge_events = []  # (t, EV_id, power)
for (k, j, s, t), val in results['P_dch_rob'].items():
    if j == robot_id and s == parking_id and val > 0.001:
        discharge_events.append((t, k, val))

charging_events = []  # (t, charger_id, power)
for (j, i, t, s), val in results['P_ch_robft'].items():
    if j == robot_id and s == parking_id and val > 0.001:
        charging_events.append((t, i, val))

# SOC trajectory
soc_vals = [results['SOC_rob'].get((robot_id, parking_id, t), 0) for t in range(1, 49)]

# Get unique EVs and chargers
unique_evs = sorted(set([ev for (_, ev, _) in discharge_events]))
unique_chargers = sorted(set([ch for (_, ch, _) in charging_events]))

# Color maps
ev_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_evs)))
charger_colors = plt.cm.Set2(np.linspace(0, 1, len(unique_chargers)))

# ============================================================================
# 2. PRE-PROCESS DATA INTO MATRICES
# ============================================================================
time = np.arange(1, 49)

# Matrix for Discharging Power
discharge_matrix = np.zeros((len(unique_evs), 48))
for i, ev in enumerate(unique_evs):
    for t in range(48):
        val = next((v for (t_ev, e, v) in discharge_events if t_ev == t+1 and e == ev), 0)
        discharge_matrix[i, t] = val

# Matrix for Charging Power (Used for Row 1 and Text Summary)
charging_matrix = np.zeros((len(unique_chargers), 48))
for i, ch in enumerate(unique_chargers):
    for t in range(48):
        val = next((v for (t_ch, c, v) in charging_events if t_ch == t+1 and c == ch), 0)
        charging_matrix[i, t] = val

# Calculate Energy Matrices (Power * 0.5 hours)
discharge_energy_matrix = discharge_matrix * 0.5
charging_energy_matrix = charging_matrix * 0.5

# ============================================================================
# 3. TEXT SUMMARY
# ============================================================================
print(f"\n{'='*70}")
print(f"ROBOT {robot_id} IN PARKING {parking_id} - VISUALIZATION SUMMARY")
print(f"{'='*70}")
print(f"\n📊 Discharges to {len(unique_evs)} EVs:")
total_dch_energy = np.sum(discharge_energy_matrix)
for i, ev in enumerate(unique_evs):
    total = np.sum(discharge_energy_matrix[i, :])
    print(f"   • EV {ev}: {total:.1f} kWh")
print(f"\n🔋 Charges from {len(unique_chargers)} fixed chargers:")
total_ch_energy = np.sum(charging_energy_matrix)
for i, ch in enumerate(unique_chargers):
    total = np.sum(charging_energy_matrix[i, :])
    print(f"   • Charger {ch}: {total:.1f} kWh")

# ============================================================================
# FIGURE 1: Comprehensive Timeline (Row 3 Removed)
# ============================================================================
# Changed nrows from 3 to 2, adjusted height
fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

# --- Panel 1: Robot SOC and Power Balance (BARS) ---
ax1 = axes[0]
ax1.plot(time, soc_vals, 'b-o', linewidth=2, markersize=3, label='SOC', color='darkblue')
ax1.fill_between(time, 0, soc_vals, alpha=0.15, color='blue')
ax1.set_ylabel('SOC (kWh)', fontsize=14, color='darkblue')
ax1.tick_params(axis='y', labelcolor='darkblue')
ax1.set_title(f'Robot {robot_id} in Parking {parking_id}: Energy Balance', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--', axis='y')
# Add charging/discharging as BARS on twin axis
ax1_twin = ax1.twinx()

# Calculate total power per time step
total_p_charge = np.sum(charging_matrix, axis=0)
total_p_discharge = -np.sum(discharge_matrix, axis=0)

# Plot bars
ax1_twin.bar(time, total_p_charge, color='green', alpha=0.6, width=0.8, label='Charging (+)')
ax1_twin.bar(time, total_p_discharge, color='red', alpha=0.6, width=0.8, label='Discharging (-)')

ax1_twin.set_ylabel('Power (kW)', fontsize=12)
ax1_twin.legend(loc='upper left', fontsize=10)
ax1_twin.set_ylim(bottom=min(0, np.min(total_p_discharge))*1.2, top=max(0, np.max(total_p_charge))*1.2)
ax1_twin.grid(False)
ax1.tick_params(axis='y', labelsize=14)
ax1_twin.tick_params(axis='y', labelsize=14)
# --- Panel 2: Discharge Energy (Stacked) ---
ax2 = axes[1]
bottom = np.zeros(len(time))
for i, ev in enumerate(unique_evs):
    # Plotting Energy values
    ax2.bar(time, discharge_energy_matrix[i, :], bottom=bottom, width=0.8, 
            label=f'EV {ev}', color=ev_colors[i], edgecolor='black', linewidth=0.5)
    bottom += discharge_energy_matrix[i, :]

ax2.set_ylabel('Energy Delivered (kWh)', fontsize=14)
ax2.set_title(f'Robot {robot_id} Discharging to EVs', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9, ncol=min(3, len(unique_evs)))
ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
ax2.tick_params(axis='y', labelsize=14)
# Common X-axis formatting
hour_positions = list(range(1, 50, 4))
hour_labels = ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', 
               '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '00:00']

for ax in axes:
    ax.set_xticks(hour_positions)
    ax.set_xticklabels(hour_labels, rotation=45, fontsize = 14)
    ax.axvline(x=24.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlim(0, 49)

plt.tight_layout()
plt.savefig(f'Robot_{robot_id}_P{parking_id}_Updated_Timeline.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# ============================================================================
# FIGURE 2: Gantt-style activity chart
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 6))
ax.set_facecolor('#f0f0f0')

# Prepare Gantt data
gantt_rows = []
row_counter = 0

# Discharge events
for t, ev, power in discharge_events:
    gantt_rows.append({
        'label': f'Discharge to EV {ev} ({power:.1f}kW)',
        'start': t - 0.4,
        'end': t + 0.4,
        'row': row_counter,
        'color': ev_colors[unique_evs.index(ev)],
        'type': 'discharge'
    })
row_counter += 1

# Charging events
for t, charger, power in charging_events:
    gantt_rows.append({
        'label': f'Charge from Ch {charger} ({power:.1f}kW)',
        'start': t - 0.4,
        'end': t + 0.4,
        'row': row_counter,
        'color': charger_colors[unique_chargers.index(charger)],
        'type': 'charge'
    })
row_counter += 1

# Draw Gantt Bars
for event in gantt_rows:
    y = event['row']
    rect = Rectangle((event['start'], y-0.35), event['end']-event['start'], 0.7,
                     facecolor=event['color'], alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(event['start'] + (event['end']-event['start'])/2, y, 
            event['label'], ha='center', va='center', fontsize=9, fontweight='bold')

# Format
ax.set_xlim(0, 49)
ax.set_ylim(-0.5, row_counter - 0.5)
ax.set_yticks([0, 1])
ax.set_yticklabels(['📤 Discharging (to EVs)', '🔌 Charging (from fixed chargers)'], fontsize=12)
ax.set_xlabel('Time Period (30-min intervals)', fontsize=12)
ax.set_xticks(hour_positions)
ax.set_xticklabels(hour_labels, rotation=45)
ax.axvline(x=24.5, color='red', linestyle='--', alpha=0.7, linewidth=2)

# SOC overlay
ax2 = ax.twinx()
soc_norm = np.array(soc_vals) / max(soc_vals) if max(soc_vals) > 0 else soc_vals
ax2.plot(time, soc_norm, 'b-', linewidth=2, marker='s', markersize=3, label='SOC (normalized)')
ax2.set_ylabel('Normalized SOC', fontsize=12, color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim(0, 1.1)

ax.set_title(f'Robot {robot_id} in Parking {parking_id}: Activity Timeline + SOC', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, axis='x', alpha=0.3, linestyle='--')
ax.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(f'Robot_{robot_id}_P{parking_id}_Gantt_Timeline.png', dpi=300, bbox_inches='tight')
plt.show(block=False)

# ============================================================================
# Print summary table
# ============================================================================
print(f"\n{'='*70}")
print(f"TIMELINE SUMMARY - Robot {robot_id} in Parking {parking_id}")
print(f"{'='*70}")
print(f"\n{'Time':<12} {'Action':<25} {'EV/Charger':<12} {'Power':<10}")
print(f"{'-'*70}")

all_events = [(t, 'DISCHARGE', f'EV {ev}', power) for (t, ev, power) in discharge_events] + \
             [(t, 'CHARGE', f'Ch {ch}', power) for (t, ch, power) in charging_events]

for t, action, target, power in sorted(all_events, key=lambda x: x[0]):
    hour = (t-1)//2
    minute = 30 if (t-1)%2 == 1 else 0
    time_str = f"{hour:02d}:{minute:02d}"
    print(f"{time_str:<12} {action:<25} {target:<12} {power:<10.1f}kW")

# Calculate and show robot utilization
active_periods = len(set([t for (t,_,_) in discharge_events] + [t for (t,_,_) in charging_events]))
print(f"\n📊 Robot Utilization: {active_periods}/48 periods ({100*active_periods/48:.1f}% of time active)")
print(f"⚡ Total Energy Discharged: {total_dch_energy:.1f} kWh")
print(f"🔋 Total Energy Charged: {total_ch_energy:.1f} kWh")
if total_ch_energy > 0:
    print(f"🔄 Round-trip Efficiency: {100 * total_dch_energy / total_ch_energy:.1f}%")
    
    
    
    
    
    
    
    
    
    

# ============================================================================
# 1. SETTINGS
# ============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle
import matplotlib.patches as mpatches

# ============================================================================
# 1. SETTINGS
# ============================================================================
parking_id = 1
snapshot_time = None  # None = Auto-find busiest time

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

def find_busiest_time(results, parking_id):
    """
    Scans all time steps to find the one with the maximum number of active 
    charging events (Robot->EV or Fixed->EV) in the specific parking.
    """
    activity_counts = {}
    
    # Count Robot -> EV
    if 'P_dch_rob' in results:
        for (k, j, s, t), val in results['P_dch_rob'].items():
            if s == parking_id and val > 0.001:
                activity_counts[t] = activity_counts.get(t, 0) + 1
                
    # Count Fixed -> EV
    if 'P_ch_evft' in results:
        for (i, k, s, t), val in results['P_ch_evft'].items():
            if s == parking_id and val > 0.001:
                activity_counts[t] = activity_counts.get(t, 0) + 1
    
    if not activity_counts:
        print("⚠️ WARNING: No activity found in Parking {}.".format(parking_id))
        return 10
        
    # Return the time with the highest count
    best_time = max(activity_counts, key=activity_counts.get)
    print(f"Found busiest time: t={best_time} (with {activity_counts[best_time]} active connections)")
    return best_time

if snapshot_time is None:
    snapshot_time = find_busiest_time(results, parking_id)
else:
    print(f"Using manually specified snapshot time: t={snapshot_time}")

# Extract Connections
connections = []

# 1. Robot -> EV (P_dch_rob)
# Indices: (k=EV, j=Robot, s=Parking, t=Time)
if 'P_dch_rob' in results:
    for (ev, rob, park, t), val in results['P_dch_rob'].items():
        if park == parking_id and t == snapshot_time and val > 0.001:
            connections.append({
                'type': 'Robot',
                'source_id': rob,
                'target_id': ev,
                'power': val
            })

# 2. Fixed Charger -> EV (P_ch_evft)
# Indices: (i=Charger, k=EV, s=Parking, t=Time)
if 'P_ch_evft' in results:
    for (ch, ev, park, t), val in results['P_ch_evft'].items():
        if park == parking_id and t == snapshot_time and val > 0.001:
            connections.append({
                'type': 'Fixed',
                'source_id': ch,
                'target_id': ev,
                'power': val
            })
else:
    print("⚠️ WARNING: 'P_ch_evft' key not found in results. Cannot plot Fixed Charger -> EV connections.")
    print("   (If your model is Fixed -> Robot -> EV, you will only see Robots here).")

# Get Unique Entities
robots = sorted(list(set([c['source_id'] for c in connections if c['type'] == 'Robot'])))
fixed_chargers = sorted(list(set([c['source_id'] for c in connections if c['type'] == 'Fixed'])))
evs = sorted(list(set([c['target_id'] for c in connections])))

# Get SOC
ev_socs = {}
if 'SOC_ev' in results:
    for (ev, park, t), val in results['SOC_ev'].items():
        if park == parking_id and t == snapshot_time:
            ev_socs[ev] = val

# ============================================================================
# 3. VISUALIZATION
# ============================================================================

if not connections:
    print(f"No connections found to plot.")
else:
    print(f"Plotting {len(connections)} connections...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Layout Coordinates
    x_sources = 0.2
    x_evs = 0.8
    node_radius = 0.03

    # Layout logic: Distribute nodes vertically
    def get_y_positions(count, start_y, spacing):
        return [start_y - (i * spacing) for i in range(count)]

    y_robots = get_y_positions(len(robots), 0.85, 0.08)
    y_fixed = get_y_positions(len(fixed_chargers), 0.35, 0.08)
    y_evs = get_y_positions(len(evs), 0.85, 0.08)

    # Maps for quick lookup
    robot_y_map = {r: y for r, y in zip(robots, y_robots)}
    fixed_y_map = {f: y for f, y in zip(fixed_chargers, y_fixed)}
    ev_y_map = {e: y for e, y in zip(evs, y_evs)}

    # Helper to draw lines
    def draw_connection(ax, x_src, y_src, x_dst, y_dst, power, color):
        lw = power * 1.2 # Scale width by power
        if lw < 1: lw = 1
        
        # Use arc3 for curved lines
        arrow = FancyArrowPatch(
            posA=(x_src + node_radius, y_src),
            posB=(x_dst - node_radius, y_dst),
            connectionstyle=f"arc3,rad=0.15",
            arrowstyle="-", 
            color=color,
            linewidth=lw,
            alpha=0.6,
            zorder=1
        )
        ax.add_patch(arrow)
        
        # Label Power
        mid_x = (x_src + x_dst) / 2
        mid_y = (y_src + y_dst) / 2
        ax.text(mid_x, mid_y, f"{power:.1f}kW", color='black', fontsize=8, 
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

    # --- DRAW SOURCES ---
    
    # Robots
    if robots:
        ax.text(x_sources, 0.92, "ROBOTS (Discharging)", ha='center', fontsize=12, fontweight='bold', color='royalblue')
        for r in robots:
            y = robot_y_map[r]
            circle = Circle((x_sources, y), node_radius, color='royalblue', ec='black', zorder=10)
            ax.add_patch(circle)
            ax.text(x_sources - 0.06, y, f"R{r}", ha='right', va='center', fontsize=9, fontweight='bold')

    # Fixed Chargers
    if fixed_chargers:
        ax.text(x_sources, 0.42, "FIXED CHARGERS", ha='center', fontsize=12, fontweight='bold', color='limegreen')
        for ch in fixed_chargers:
            y = fixed_y_map[ch]
            rect = Rectangle((x_sources - node_radius, y - node_radius), node_radius*2, node_radius*2, 
                             color='limegreen', ec='black', zorder=10)
            ax.add_patch(rect)
            ax.text(x_sources - 0.06, y, f"Ch{ch}", ha='right', va='center', fontsize=9, fontweight='bold')

    # --- DRAW TARGETS (EVs) ---
    for ev in evs:
        y = ev_y_map[ev]
        soc = ev_socs.get(ev, 0)
        soc_color = plt.cm.RdYlGn(soc / 100)
        
        circle = Circle((x_evs, y), node_radius, color=soc_color, ec='black', zorder=10, linewidth=1.5)
        ax.add_patch(circle)
        
        ax.text(x_evs + 0.06, y, f"EV {ev}", ha='left', va='center', fontsize=9, fontweight='bold')
        ax.text(x_evs, y, f"{soc:.0f}%", ha='center', va='center', fontsize=7, color='black', fontweight='bold')

    # --- DRAW CONNECTIONS ---
    for c in connections:
        src_id = c['source_id']
        tgt_id = c['target_id']
        pwr = c['power']
        
        if c['type'] == 'Robot':
            y_src = robot_y_map[src_id]
            color = 'royalblue'
        else:
            y_src = fixed_y_map[src_id]
            color = 'limegreen'
            
        y_tgt = ev_y_map[tgt_id]
        draw_connection(ax, x_sources, y_src, x_evs, y_tgt, pwr, color)

    # Title
    hour = (snapshot_time-1)//2
    minute = 30 if (snapshot_time-1)%2 == 1 else 0
    time_str = f"{hour:02d}:{minute:02d}"
    
    ax.set_title(f"Parking {parking_id} Network Snapshot at {time_str} (Busiest Time)", 
                 fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(f'Parking_{parking_id}_Snapshot_Busiest.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)