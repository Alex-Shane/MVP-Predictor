import pandas as pd

def format_fielding(filename):
    # cut off top 4 lines (whitespace) and remove all pitchers
    df_fielding = pd.read_excel(filename, header = 4)
    df_fielding = df_fielding[df_fielding['Pos'] != 'P']

    # check if any pitchers left
    test = df_fielding[df_fielding['Pos'] == 'P']
    assert(not test.shape[0])

    df_fielding=df_fielding.iloc[:, 1:]

    df_fielding = filter_firstname_lastname(df_fielding)

    df_fielding = df_fielding[df_fielding['G'] >= 120]

    return df_fielding

def format_hitting(filename):
    df_hitters = pd.read_excel(filename, header = 4)

    # Create a mask to identify rows where 'Lg' is 'MLB'
    mlb_mask = df_hitters['Tm'] == 'TOT'

    # Create a mask to identify duplicate names within 'Name' column
    duplicate_names_mask = df_hitters.duplicated(subset='Name', keep=False)

    # Filter the DataFrame to keep rows where 'Lg' is 'MLB' or not duplicate names
    df_hitters = df_hitters[(mlb_mask & duplicate_names_mask) | ~duplicate_names_mask]

    df_hitters = filter_firstname_lastname(df_hitters)

    df_hitters = df_hitters[df_hitters['PA'] >= 500]

    # chop off *'s and #'s
    i = 0
    for lastname in df_hitters["Last Name"]:
        if ('*' in lastname) or ('#' in lastname):
            df_hitters.loc[i, 'Last Name'] = lastname[:-1]
        i+=1

    return df_hitters

def clean_name(name):
    if '*' in name or '#' in name:
        return name[:-1]
    return name

def filter_firstname_lastname(df):
    # Split the "Name" column into "Firstname" and "Lastname"
    df[['First Name', 'Last Name']] = df['Name'].str.split(n=1, expand=True)

    # Sort the DataFrame by the "Lastname" column
    df.sort_values(by='Last Name', inplace=True)

    # Drop the intermediate "Name" column
    df = df.drop(columns='Name')

    # Sort the DataFrame by "Last Name" and then by "First Name"
    df = df.sort_values(by=['Last Name', 'First Name']).reset_index(drop=True)

    # Reorder the columns
    new_column_order = ['Last Name', 'First Name'] + [col for col in df.columns if col not in ['Lastname', 'Firstname']]
    df = df[new_column_order]

    # Drop the last two columns
    df = df.iloc[:, :-2]

    df["Last Name"] = df["Last Name"].apply(clean_name)

    return df



def align_names(df1, df2):

    # Merge dataframes on 'Last Name' and 'First Name'
    merged_df = pd.merge(df1, df2, on=['Last Name', 'First Name'], how='inner')

    # Separate the dataframes for players present in both
    df1_filtered = df1[df1.apply(lambda row: (row['Last Name'], row['First Name']) in set(zip(merged_df['Last Name'], merged_df['First Name'])), axis=1)]
    df2_filtered = df2[df2.apply(lambda row: (row['Last Name'], row['First Name']) in set(zip(merged_df['Last Name'], merged_df['First Name'])), axis=1)]

    return df1_filtered, df2_filtered

def add_roba_column(df1, df2):
    rOBA_vals = list()
    df1['Name'] = df1['First Name'] + chr(160) + df1['Last Name']
    
    for name in df1["Name"]:
        rOBA_vals.append(df2.loc[df2["Name"] == name, "rOBA"].values[0])
    
    df1["rOBA"] = rOBA_vals
    df1.drop(columns=['Name'], inplace=True)

    return df1 

def add_WAR_column(df1, df2):
    WAR_vals = list()
    df1['Name'] = df1['First Name'] + chr(160) + df1['Last Name']
    
    for name in df1["Name"]:
        WAR_vals.append(df2.loc[df2["Name"] == name, "WAR"].values[0])
    
    df1["WAR"] = WAR_vals
    df1.drop(columns=["Name"], inplace=True)
    
    return df1

def clean_all_folders():
    for x in range(2006, 2007):
        if x == 2020:
            continue
        print(x)
        fielding_filename = f'./training_data/{x}/fielding.xlsx'
        hitting_filename = f'./training_data/{x}/basic_hitting.xlsx'
        df_hitters = format_hitting(hitting_filename)
        df_position_players = format_fielding(fielding_filename)
        df_hitters, df_position_players = align_names(df_hitters, df_position_players)
        
        defense_columns = ['G', 'GS', 'Rtot', 'Pos']
        df_def = df_position_players[defense_columns]
        df_def.to_csv(f'./training_data/full_season_data/{x}_defense.csv', index = False)
        
        roba_file = f'./training_data/{x}/rOBA.xlsx'
        roba_df = pd.read_excel(roba_file, header = 5)
        df_final = add_roba_column(df_hitters, roba_df)
        
        war_file = f'./training_data/{x}/WAR.xlsx'
        war_df = pd.read_excel(war_file, header = 4)
        war_df = war_df.dropna(how='any')
        war_df.loc[:, "Name"] = war_df["Name"].apply(clean_name)
        df_final = add_WAR_column(df_final, war_df)
        df_final.to_csv(f'./training_data/full_season_data/{x}_hitting.csv', index = False)
        print(f'{x} cleaned!')

    
def clean_standings():
    for x in range(2005,2023):
        df = pd.read_excel('./training_data/{x}/standings.xlsx', header = 4, index_col = False)
        teams = dict()
        if x < 2008:
            teams = {
            "St. Louis Cardinals": "STL",
            "Chicago White Sox": "CWS",
            "New York Yankees": "NYY",
            "Los Angeles Angels of Anaheim": "LAA",
            "Boston Red Sox": "BOS",
            "Cleveland Indians": "CLE",
            "Atlanta Braves": "ATL",
            "Houston Astros": "HOU",
            "Philadelphia Phillies": "PHI",
            "Oakland Athletics": "OAK",
            "Florida Marlins": "FLA",
            "New York Mets": "NYM",
            "Minnesota Twins": "MIN",
            "San Diego Padres": "SDP",
            "Milwaukee Brewers": "MIL",
            "Washington Nationals": "WSN",
            "Toronto Blue Jays": "TOR",
            "Texas Rangers": "TEX",
            "Chicago Cubs": "CHC",
            "Arizona Diamondbacks": "ARI",
            "San Francisco Giants": "SFG",
            "Baltimore Orioles": "BAL",
            "Cincinnati Reds": "CIN",
            "Los Angeles Dodgers": "LAD",
            "Detroit Tigers": "DET",
            "Seattle Mariners": "SEA",
            "Colorado Rockies": "COL",
            "Tampa Bay Devil Rays": "TBD",
            "Pittsburgh Pirates": "PIT",
            "Kansas City Royals": "KCR"
            }
        elif x < 2012:
            teams = {"Los Angeles Angels of Anaheim": "LAA",
                     "Chicago Cubs": "CHC",
                     "Tampa Bay Rays": "TBR",
                     "Boston Red Sox": "BOS",
                     "Philadelphia Phillies": "PHI",
                     "Milwaukee Brewers": "MIL",
                     "New York Mets": "NYM",
                     "New York Yankees": "NYY",
                     "Chicago White Sox": "CWS",
                     "Minnesota Twins": "MIN",
                     "Houston Astros": "HOU",
                     "St. Louis Cardinals": "STL",
                     "Toronto Blue Jays": "TOR",
                     "Florida Marlins": "FLA",
                     "Los Angeles Dodgers": "LAD",
                     "Arizona Diamondbacks": "ARI",
                     "Cleveland Indians": "CLE",
                     "Texas Rangers": "TEX",
                     "Oakland Athletics": "OAK",
                     "Kansas City Royals": "KCR",
                     "Colorado Rockies": "COL",
                     "Detroit Tigers": "DET",
                     "Cincinnati Reds": "CIN",
                     "Atlanta Braves": "ATL",
                     "San Francisco Giants": "SFG",
                     "Baltimore Orioles": "BAL",
                     "Pittsburgh Pirates": "PIT",
                     "San Diego Padres": "SDP",
                     "Seattle Mariners": "SEA",
                     "Washington Nationals": "WSN"
                     }
        elif x < 2022:
            teams = {
                "Washington Nationals": "WSN",
                "Cincinnati Reds": "CIN",
                "New York Yankees": "NYY",
                "San Francisco Giants": "SFG",
                "Oakland Athletics": "OAK",
                "Atlanta Braves": "ATL",
                "Baltimore Orioles": "BAL",
                "Texas Rangers": "TEX",
                "Tampa Bay Rays": "TBR",
                "Los Angeles Angels of Anaheim": "LAA",
                "Detroit Tigers": "DET",
                "St. Louis Cardinals": "STL",
                "Los Angeles Dodgers": "LAD",
                "Chicago White Sox": "CWS",
                "Milwaukee Brewers": "MIL",
                "Arizona Diamondbacks": "ARI",
                "Philadelphia Phillies": "PHI",
                "Pittsburgh Pirates": "PIT",
                "San Diego Padres": "SDP",
                "Seattle Mariners": "SEA",
                "New York Mets": "NYM",
                "Toronto Blue Jays": "TOR",
                "Kansas City Royals": "KCR",
                "Miami Marlins": "MIA",
                "Boston Red Sox": "BOS",
                "Cleveland Indians": "CLE",
                "Minnesota Twins": "MIN",
                "Colorado Rockies": "COL",
                "Chicago Cubs": "CHC",
                "Houston Astros": "HOU"
                }
        else:
            teams = {"Los Angeles Dodgers": "LAD",
                     "Houston Astros": "HOU",
                     "Atlanta Braves": "ATL",
                     "New York Mets": "NYM",
                     "New York Yankees": "NYY",
                     "St. Louis Cardinals": "STL",
                     "Cleveland Guardians": "CLE",
                     "Toronto Blue Jays": "TOR",
                     "Seattle Mariners": "SEA",
                     "San Diego Padres": "SDP",
                     "Philadelphia Phillies": "PHI",
                     "Milwaukee Brewers": "MIL",
                     "Tampa Bay Rays": "TBR",
                     "Baltimore Orioles": "BAL",
                     "Chicago White Sox": "CWS",
                     "San Francisco Giants": "SFG",
                     "Minnesota Twins": "MIN",
                     "Boston Red Sox": "BOS",
                     "Chicago Cubs": "CHC",
                     "Arizona Diamondbacks": "ARI",
                     "Los Angeles Angels": "LAA",
                     "Miami Marlins": "MIA",
                     "Texas Rangers": "TEX",
                     "Colorado Rockies": "COL",
                     "Detroit Tigers": "DET",
                     "Kansas City Royals": "KCR",
                     "Pittsburgh Pirates": "PIT",
                     "Cincinnati Reds": "CIN",
                     "Oakland Athletics": "OAK",
                     "Washington Nationals": "WSN"}
        df["Tm"] = list(teams.values())
        df.to_csv("./training_data/{x}/standings.csv", index = False)
        print (f'{x} standings cleaned!')
            
            
if __name__ == "__main__":
    clean_standings()
    #clean_all_folders()

    """fielding_filename = "./training_data/2004/2004_fielding.xlsx"
    hitting_filename = "./training_data/2004/2004_basic_hitting.xlsx"

    df_hitters = format_hitting(hitting_filename)
    df_position_players = format_fielding(fielding_filename)

    # df_hitters.to_csv("hitterstest_pre.csv", index = False)
    # df_position_players.to_csv("fielderstest_pre.csv", index = False)

    df_hitters, df_position_players = align_names(df_hitters, df_position_players)
    
    # df_hitters.to_csv("hitterstest_post.csv", index = False)
    # df_position_players.to_csv("fielderstest_post.csv", index = False)

    #df_final = merge(df_position_players, df_hitters)
    
    roba_file = "./training_data/2004/2004_rOBA.xlsx"
    roba_df = pd.read_excel(roba_file, header = 4)
    df_final = add_roba_column(df_hitters, roba_df)
    
    war_file = "./training_data/2004/2004_WAR.xlsx"
    war_df = pd.read_excel(war_file, header = 4)
    war_df = war_df.dropna(how='any')
    war_df.loc[:, "Name"] = war_df["Name"].apply(clean_name)
    df_final = add_WAR_column(df_final, war_df)
    df_final.to_csv('finaltest_alex.csv')
    #df_final = add_roba_column(df_final, roba_df)
    #df_final.to_csv('finaltest_alex.csv')"""
    
    
    

