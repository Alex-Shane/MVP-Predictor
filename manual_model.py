#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:06:19 2023

@author: Alex
"""

import pandas as pd
import warnings


# Define the NL weights for each statistic (customize these based on your preferences)
NL_BAW = 2.38888889    # Weight for Batting Average
NL_HRW = 2.346978557   # Weight for Home Runs
NL_RBIW = 2.211202938  # Weight for RBIs
NL_OPSW = 4.864646464  # Weight for OPS
NL_WARW = 6.86039886   # Weight for WAR
NL_SBW = 0.5597396559  # Weight for Stolen Bases
NL_ROBAW = 4.864646464  # Weight for rOBA
NL_DEFW = 0.1955815464  # Weight for rTot(defense)
NL_TEAMW = 4.053872054  # Weight for team ranking (by win%)

# Define the AL weights for each statistic (customize these based on your preferences)
AL_BAW = 2.77       # Weight for Batting Average
AL_HRW = 2.248      # Weight for HRs
AL_RBIW = 2.206     # Weight for RBIs
AL_OPSW = 6.499     # Weight for OPS
AL_WARW = 9.929     # Weight for WAR
AL_SBW = 0.863      # Weight for SBs
AL_ROBAW = 6.744    # Weight for rOBA
AL_DEFW = 0.168     # Weight for defense
AL_TEAMW = 4.703    # Weight for team performance 

#Define standardized weights (average of both leagues) for each stat
BAW = 2.579444444444444    # Weight for Batting Average
HRW = 2.29748928   # Weight for Home Runs
RBIW = 2.20860146  # Weight for RBIs
OPSW = 5.6818232  # Weight for OPS
WARW = 8.39469943   # Weight for WAR
SBW = 0.71136828  # Weight for Stolen Bases
ROBAW = 5.8043232  # Weight for rOBA
DEFW = 0.18179077  # Weight for rTot(defense)
TEAMW = 4.37843602 # Weight for team ranking (by win%)


def inSampleTesting():
    for x in range(2004, 2023):
        if x == 2020:
            continue
        data = pd.read_csv(
            f'./training_data/full_season_data/{x}_composite.csv')
        al_data = data[data['Lg'] == 'AL']
        nl_data = data[data['Lg'] == 'NL']

        # Iterate through each statistic column and calculate the rank
        for stat in ['BA', 'HR', 'RBI', 'SB', 'WAR', 'OPS', 'rOBA', 'Rtot', 'WinPercentage']:
            column_name = stat + '_Rank'
            al_data.loc[:, column_name] = al_data[stat].rank(
                ascending=False, method='dense')
            nl_data.loc[:, column_name] = nl_data[stat].rank(
                ascending=False, method='dense')

        # Calculate MVP Predictor score for each player
        al_data['MVP_Predictor'] = (
            BAW * al_data['BA_Rank'] +
            HRW * al_data['HR_Rank'] +
            RBIW * al_data['RBI_Rank'] +
            SBW * al_data['SB_Rank'] +
            WARW * al_data['WAR_Rank'] +
            OPSW * al_data['OPS_Rank'] +
            ROBAW * al_data['rOBA_Rank'] +
            DEFW * al_data['Rtot_Rank'] +
            TEAMW * al_data['WinPercentage_Rank']
        )

        nl_data['MVP_Predictor'] = (
            BAW * nl_data['BA_Rank'] +
            HRW * nl_data['HR_Rank'] +
            RBIW * nl_data['RBI_Rank'] +
            SBW * nl_data['SB_Rank'] +
            WARW * nl_data['WAR_Rank'] +
            OPSW * nl_data['OPS_Rank'] +
            ROBAW * nl_data['rOBA_Rank'] +
            DEFW * nl_data['Rtot_Rank'] +
            TEAMW * nl_data['WinPercentage_Rank']
        )

        # Sort the players by MVP Predictor score in ascending order
        al_data = al_data.sort_values(by='MVP_Predictor', ascending=True)
        nl_data = nl_data.sort_values(by='MVP_Predictor', ascending=True)

        # get the predicted MVP(s)
        al_mvp = al_data[['Last Name', 'MVP_Predictor']].head(1)
        nl_mvp = nl_data[['Last Name', 'MVP_Predictor']].head(1)

        # print the predicted MVPs
        print(f'Predicted {x} AL MVP: {al_mvp}')
        print(f'Predicted {x} NL MVP: {nl_mvp}')

warnings.filterwarnings("ignore")
#inSampleTesting()
# AL: got Morneau wrong (said Berkman), got Pedroia wrong (said Rodriguez), got Cabera wrong (said Trout), got Donaldson wrong (said Trout), got Altuve wrong (said Judge), got Betts wrong (said Trout),
    # accuracy = 10/16 = 63% with seperate weights
    # accuracy = 10/16 = 63% with standardized weights
# NL: got Howard wrong (said Pujols), got Rollins wrong (said Pujols), got Votto wrong (said Pujols), got Posey wrong (said Braun), got McCutchen wrong (said Goldschmidt), got Bryant wrong (said freeman), got Stanton wrong (said Votto), got Bellinger wrong (said Rendon), got Harper wrong (said Soto)
    # accuracy = 8/17 = 47% with seperate weights
    # accuracy = 9/16 = 56% with standardized weights (got McCutchen right)
    
    
# Load the CSV file containing player statistics
data = pd.read_csv('./training_data/full_season_data/2006_composite.csv')
al_data = data[data['Lg'] == 'AL']
nl_data = data[data['Lg'] == 'NL']

# Iterate through each statistic column and calculate the rank
for stat in ['BA', 'HR', 'RBI', 'SB', 'WAR', 'OPS', 'rOBA', 'Rtot', 'WinPercentage']:
    column_name = stat + '_Rank'
    al_data.loc[:, column_name] = al_data[stat].rank(
        ascending=False, method='dense')
    nl_data.loc[:, column_name] = nl_data[stat].rank(
        ascending=False, method='dense')

# Calculate MVP Predictor score for each player
al_data['MVP_Predictor'] = (
    BAW * al_data['BA_Rank'] +
    HRW * al_data['HR_Rank'] +
    RBIW * al_data['RBI_Rank'] +
    SBW * al_data['SB_Rank'] +
    WARW * al_data['WAR_Rank'] +
    OPSW * al_data['OPS_Rank'] +
    ROBAW * al_data['rOBA_Rank'] +
    DEFW * al_data['Rtot_Rank'] +
    TEAMW * al_data['WinPercentage_Rank']
)

nl_data['MVP_Predictor'] = (
    BAW * nl_data['BA_Rank'] +
    HRW * nl_data['HR_Rank'] +
    RBIW * nl_data['RBI_Rank'] +
    SBW * nl_data['SB_Rank'] +
    WARW * nl_data['WAR_Rank'] +
    OPSW * nl_data['OPS_Rank'] +
    ROBAW * nl_data['rOBA_Rank'] +
    DEFW * nl_data['Rtot_Rank'] +
    TEAMW * nl_data['WinPercentage_Rank']
)

# Sort the players by MVP Predictor score in ascending order
al_data = al_data.sort_values(by='MVP_Predictor', ascending=True)
nl_data = nl_data.sort_values(by='MVP_Predictor', ascending=True)


# Print the predicted MVP(s)
print(al_data[['Last Name', 'MVP_Predictor']].head(5))
print(nl_data[['Last Name', 'MVP_Predictor']].head(5))
