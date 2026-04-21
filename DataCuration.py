# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:04:24 2025

@author: arsalann
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
from datetime import datetime, time

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans

#### FOR datetime.time FORMAT
#def time_to_seconds(t):
#    return t.hour * 3600 + t.minute * 60 + t.second


#def time_to_slot(t, SampPerH):
#    total_minutes = t.hour * 60 + t.minute + t.second / 60        
#    slot = int(round(total_minutes / (60/SampPerH)))  # round to nearest e.g. 10 min
#    slot = max(1, min(slot, SampPerH*24))  # ensure within 1..T
#    return slot


#### FOR str FORMAT

def time_to_seconds(t_str):
    hh, mm, ss = map(int, t_str.split(':'))
    return hh * 3600 + mm * 60 + ss


def time_to_slot(t, SampPerH):
    hh, mm, ss = map(int, t.split(':'))
    total_minutes = hh * 60 + mm + ss / 60    
    slot = int(round(total_minutes / (60/SampPerH)))  # round to nearest e.g. 10 min
    slot = max(1, min(slot, SampPerH*24))  # ensure within 1..T
    return slot
#######################################################


current_directory = os.path.dirname(__file__)
current_directory = os.getcwd()
#StartTime = pd.read_excel(file_path, sheet_name='startTime_24h')
#EndTime = pd.read_excel(file_path, sheet_name='endTime_24h')

#X = pd.read_excel(file_path, sheet_name='xCoord')
#Y = pd.read_excel(file_path, sheet_name='yCoord')

def DataCuration(df, SampPerH, ChargerCap, ParkNo):
    
    #ParkNo = 3
    
#    df = pd.read_excel(file_path, sheet_name='filtered_charging_events2_Updat')

    df['start_sec'] = df['startTime_24h'].apply(time_to_seconds)
    df['end_sec'] = df['endTime_24h'].apply(time_to_seconds)
    
    # Step 2: Filter rows where charging time is at least 60 seconds
    
    
    
    df['EVcap'] = df['endSoc_kWh'] / df['endSoc']
    
    df['SOCout'] = df['endSoc']
    df['SOCin'] = df['startSoc']
    
    
    #######################
    ### for parking number
    
    coords = df[['xCoord', 'yCoord']]
    kmeans = KMeans(n_clusters = ParkNo, random_state=42, n_init=10)
    
    df['ParkingNo'] = kmeans.fit_predict(coords)
    
    # Optional: convert cluster numbers from 0,1,2 to 1,2,3
    df['ParkingNo'] = df['ParkingNo'] + 1
    
    #############################
    ## for AT and DT
    

 

    df['AT'] = df['startTime_24h'].apply(lambda t: time_to_slot(t, SampPerH))  # Correct
    df['DT'] = df['endTime_24h'].apply(lambda t: time_to_slot(t, SampPerH))    # Correct
    
    # Ensure duration is at least 1 slot (10 min)
    df['DT'] = np.where(df['DT'] <= df['AT'],
                                 df['AT'] + 1,
                                 df['DT'])
    if SampPerH == 1:
        df['DT'] = df['DT'] + 4
#        df['AT'] = df['AT'] - 2
    elif  SampPerH == 2:
        df['DT'] = df['DT'] + 6
    else:
        df['DT'] = df['DT'] + 8


    # Ensure DT does not go beyond 144 (wrap if needed)
    df['DT'] = np.where(df['DT'] > 24*SampPerH, 24*SampPerH, df['DT'])
    df['AT'] = np.where(df['AT'] > 24*SampPerH -2, 24*SampPerH -2, df['AT'])

    df['Duration'] = df['DT'] - df['AT']



   #########################
    for k in range(0,len(df['DT']) ):
        while ( (df['Duration'][k] - 1)*ChargerCap)/SampPerH < df['transmittedEnergy_kWh'][k]:
           
           if ( (df['Duration'][k] - 1)*ChargerCap)/SampPerH < df['transmittedEnergy_kWh'][k]:
               if df['DT'][k]<24*SampPerH:
                  df['DT'][k] = df['DT'][k] + 1 
               else:
                   df['AT'][k] = df['AT'][k] - 1
           df['Duration'][k] = df['DT'][k] - df['AT'][k]    
               
#           if ( (df['Duration'][k] - 1)*ChargerCap)/SampPerH >= df['transmittedEnergy_kWh'][k]:
#               break 
    
    
    
    
    #####################################
    
    df.to_excel('clustered_output.xlsx', index=False)
    # Plotting
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['xCoord'], df['yCoord'], c=df['ParkingNo'], cmap='viridis', s=50)
    
    # Add legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.gca().add_artist(legend1)
    
    plt.xlabel('xCoord')
    plt.ylabel('yCoord')
    plt.title('Geographical Clustering (KMeans)')
    plt.grid(True)
    plt.show()
    
    return df
 
    
    
