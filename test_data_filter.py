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

if __name__ == "__main__":
    clean_all_folders()

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
    

