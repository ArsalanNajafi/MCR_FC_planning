import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GlobalData import GlobalData
import pyomo.environ as pyo
import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GlobalData import GlobalData

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from GlobalData import GlobalData
import pyomo.environ as pyo

[parking_to_node, ChargerCap, SampPerH, Vmin, Ch_cost, nChmax, PgridMax, NYearCh, NYearRob, RobotTypes, robotCC, IR, DCchargerCap, PeakPrice, MaxRobot, NevSame] = GlobalData()

def PowerFlow_PyPSA(ParkingDemand, Pattern, Price):
    """
    Robust version: Loop over BUSES to ensure proper load creation/update.
    Solves OPF with Gurobi QCQP (SOCP physics).
    """
    
    # 1. Initialize Network
    n = pypsa.Network()
    
    # Define Snapshots
    time_steps = SampPerH * 24
    n.set_snapshots(pd.RangeIndex(time_steps, name="time"))
    
    # 2. Load Data
    data = pyo.DataPortal()
    data.load(filename='iee33_bus_data.dat')
    
    vb = 12.66
    sb = 10
    
    # --- 3. ADD BUSES ---
    print("Building Buses...")
    for i in range(33):
        n.add("Bus", name=str(i), v_nom=vb, carrier="AC")
    
    # --- 4. ADD LINES ---
    print("Building Lines...")
    for l, from_bus, to_bus in data['lineCon']:
        n.add("Line", 
               name=f"L{l}",
               bus0=str(from_bus),
               bus1=str(to_bus),
               x=data['X'][l],       
               r=data['R'][l],       
               s_nom=10.0            
        )
        
    # --- 5. ADD GENERATOR (Slack Bus) ---
    n.add("Generator", 
          name="Slack", 
          bus="0",                
          p_nom=100,             
          control="Slack", 
          marginal_cost=0
    )

    # --- 6. PREPARE PARKING LOADS (Time Series) ---
    # Create a DataFrame to hold parking power (Snapshots x Buses)
    parking_load_series = pd.DataFrame(0.0, index=n.snapshots, columns=range(33))
    
    # Fill the series from your dictionary
    for t_idx, t in enumerate(n.snapshots):
        my_time = t_idx + 1 
        
        for p, bus_node in parking_to_node.items():
            if (p, my_time) in ParkingDemand:
                parking_load_series.at[t, bus_node] += ParkingDemand[p, my_time]
    
    # --- 7. ADD / UPDATE LOADS (Robust Method) ---
    # Instead of adding base loads then trying to merge, we loop over BUSES.
    # For each bus, we calculate Total Load = Base + Parking, then create/update.
    print("Building and Applying Loads...")
    
    for node in range(33):
        # A. Get Base Demand (Constant)
        base_p = data['bus_Pd'][node] if node in data['bus_Pd'] else 0
        base_q = data['bus_Qd'][node] if node in data['bus_Qd'] else 0
        
        # B. Get Parking Demand (Time Series)
        # parking_load_series is a DataFrame. Selecting column 'node' gives a Series.
        node_parking_p = parking_load_series[node]
        # EVs usually have PF ~1, so Q ~ 0. We keep base Q + 0.
        node_parking_q = node_parking_p * 0.1 # Small reactive load if needed, or 0
        
        # C. Total Demand
        total_p_series = base_p + node_parking_p
        total_q_series = base_q + node_parking_q # Simplification
        
        # D. Add or Update Load Object
        load_name = f"Load_Base_{node}"
        
        if load_name in n.loads:
            # Update existing
            n.loads_t.p_set[load_name] = total_p_series
            n.loads_t.q_set[load_name] = total_q_series
        else:
            # Create new (Only if there is actual demand to add)
            if total_p_series.sum() > 0 or total_q_series.sum() > 0:
                n.add("Load", 
                      name=load_name, 
                      bus=str(node),
                      p_set=total_p_series,
                      q_set=total_q_series
                )

    # --- 8. SOLVE ---
    print("Solving with PyPSA (Gurobi QCQP)...")
    solver_options = {
        'NonConvex': 2,   
        'NumericFocus': 2,
        'OutputFlag': 1,    # 1 to see Gurobi logs
        'TimeLimit': 100000
    }
    
    try:
        n.lopf(
            snapshots=n.snapshots,
            solver_name="gurobi",
            solver_options=solver_options,
            extra_functionality="voltage_constraints"
        )
        print("Solver Successful.")
    except Exception as e:
        print(f"Solver Error: {e}")
        return None, None, None, None, None, None, None

    # --- 9. EXTRACT RESULTS ---
    
    # Voltages
    volt_per_node_pu = n.buses_t.v_pu.T 
    min_voltages = volt_per_node_pu.min(axis=1)
    
    # Plotting
    plt.figure(figsize=(8, 4))
    plt.plot(range(33), min_voltages, 'r-o', label='PyPSA Voltage')
    plt.axhline(y=Vmin, color='b', linestyle='--', label='Vmin')
    plt.title('Voltage Profile (PyPSA - Robust Version)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("MinVoltage_PyPSA_Robust.png", dpi=600)
    plt.show(block=False)

    # --- 10. EXTRACT DUALS (LMPs) ---
    lmp_values = n.buses_t.marginal_price.values.T 
    
    duals_balance_p = {}
    for node in range(33):
        for t_idx, t in enumerate(n.snapshots):
            duals_balance_p[(node, t_idx+1)] = lmp_values[node, t_idx]

    return min_voltages, volt_per_node_pu, None, None, duals_balance_p, None, 0