#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:06:19 2023

@author: Alex
"""

import pandas as pd
import warnings

#Define standardized weights (average of both leagues) for each stat
BAW = 3.999444444  # Weight for Batting Average
HRW = 3.39748928  # Weight for Home Runs
RBIW = 2.20860146  # Weight for RBIs
OPSW = 5.6818232  # Weight for Stolen Bases
WARW = 10.1       # WEight for WAR
SBW = 0.71136828  # Weigth for SB
ROBAW = 5.8043232 # Weigth for rOBA
DEFW = 0.18179077 # weight for DEF
TEAMW = 6.37843602 # weight for win%


def inSampleTesting():
    for x in range(2004, 2023):
        if x == 2020:
            continue
        data = pd.read_csv(
            f'./training_data/full_season_data/{x}_composite.csv')
        al_data = data[data['Lg'] == 'AL']
        nl_data = data[data['Lg'] == 'NL']

        # rank each player by each stat
        for stat in ['BA', 'HR', 'RBI', 'SB', 'WAR', 'OPS', 'rOBA', 'Rtot', 'WinPercentage']:
            column_name = stat + '_Rank'
            al_data.loc[:, column_name] = al_data[stat].rank(
                ascending=False, method='dense')
            nl_data.loc[:, column_name] = nl_data[stat].rank(
                ascending=False, method='dense')

        # calculate MVP Predictor score for each player
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
        
        # check for triple crown winner
        al_triple_crown_winner = al_data[(al_data['BA_Rank'] == 1) & (al_data['HR_Rank'] == 1) & (al_data['RBI_Rank'] == 1)]
        if not al_triple_crown_winner.empty:
            al_data.loc[al_triple_crown_winner.index[0], 'MVP_Predictor'] -= 100
    
        nl_triple_crown_winner = nl_data[(nl_data['BA_Rank'] == 1) & (nl_data['HR_Rank'] == 1) & (nl_data['RBI_Rank'] == 1)]
        if not nl_triple_crown_winner.empty:
            nl_data.loc[nl_triple_crown_winner.index[0], 'MVP_Predictor'] -= 100

        # sort the players by MVP Predictor score in ascending order
        al_data = al_data.sort_values(by='MVP_Predictor', ascending=True)
        nl_data = nl_data.sort_values(by='MVP_Predictor', ascending=True)

        # get the predicted MVP(s)
        al_mvp = al_data[['Last Name', 'MVP_Predictor']].head(5)
        nl_mvp = nl_data[['Last Name', 'MVP_Predictor']].head(5)

        # print the predicted MVPs
        print(f'Predicted {x} AL MVP: {al_mvp}')
        print(f'Predicted {x} NL MVP: {nl_mvp}')

warnings.filterwarnings("ignore")
    
    
# Load the CSV file containing player statistics
data = pd.read_csv('./testing_data/2023_composite.csv')
al_data = data[data['Lg'] == 'AL']
nl_data = data[data['Lg'] == 'NL']

# rank each player by each stat
for stat in ['BA', 'HR', 'RBI', 'SB', 'WAR', 'OPS', 'rOBA', 'Rtot', 'WinPercentage']:
    column_name = stat + '_Rank'
    al_data.loc[:, column_name] = al_data[stat].rank(
        ascending=False, method='dense')
    nl_data.loc[:, column_name] = nl_data[stat].rank(
        ascending=False, method='dense')

# calculate MVP Predictor score for each player
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


# adjust for triple crown
al_triple_crown_winner = al_data[(al_data['BA_Rank'] == 1) & (al_data['HR_Rank'] == 1) & (al_data['RBI_Rank'] == 1)]
if not al_triple_crown_winner.empty:
    al_data.loc[al_triple_crown_winner.index[0], 'MVP_Predictor'] -= 100

nl_triple_crown_winner = nl_data[(nl_data['BA_Rank'] == 1) & (nl_data['HR_Rank'] == 1) & (nl_data['RBI_Rank'] == 1)]
if not nl_triple_crown_winner.empty:
    nl_data.loc[nl_triple_crown_winner.index[0], 'MVP_Predictor'] -= 100

# Sort the players by MVP Predictor score in ascending order
al_data = al_data.sort_values(by='MVP_Predictor', ascending=True)
nl_data = nl_data.sort_values(by='MVP_Predictor', ascending=True)

# Print the predicted MVP(s)
print(al_data[['Last Name', 'MVP_Predictor']].head(5))
print(nl_data[['Last Name', 'MVP_Predictor']].head(5))
