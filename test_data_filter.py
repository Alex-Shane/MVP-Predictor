import pandas as pd

def format_fielding(filename):
    # cut off top 4 lines (whitespace) and remove all pitchers
    df_fielding = pd.read_excel(filename, header = 4)
    df_fielding = df_fielding[df_fielding['Pos Summary'] != 'P']

    # check if any pitchers left
    test = df_fielding[df_fielding['Pos Summary'] == 'P']
    assert(not test.shape[0])

    df_fielding=df_fielding.iloc[:, 1:]

    df_fielding = filter_firstname_lastname(df_fielding)

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

    return df_hitters

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

    return df


if __name__ == "__main__":

    fielding_filename = "./training_data/2004/2004_fielding.xlsx"
    hitting_filename = "./training_data/2004/2004_basic_hitting.xlsx"

    df_hitters = format_hitting(hitting_filename)
    df_position_players = format_fielding(fielding_filename)

    # output dataframes to csv's
    df_hitters.to_csv("hitterstest.csv", index = False)
    df_position_players.to_csv("fielderstest.csv", index = False)





