import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def task_1(data_prefix_str):
    """
    This function merges all csv files with a given prefix that are in the current working directory into a single dataframe, and sorts by date.

    Parameters
    ----------
    data_prefix_str : String
        The prefix of the csv files that you want to merge. For example; ERCOT.

    Returns
    -------
    merged_ercot_df : dataframe
        The merged dataframe, sorted by datetime value.

    """
    current_directory_str = os.getcwd() #gets current dictory
    price_csv_files_list = [f for f in os.listdir(current_directory_str) if f.startswith(data_prefix_str)] #List of files with that prefix
    price_filepaths_list = [os.path.join(current_directory_str, f) for f in price_csv_files_list] #List of filepaths for each csv file to be merged
    merged_ercot_df = pd.concat(map(pd.read_csv, price_filepaths_list), ignore_index=True) #Merges all csv files in the filepath list
    merged_ercot_df['Date'] = pd.to_datetime(merged_ercot_df['Date'])

    merged_ercot_df=merged_ercot_df.sort_values(by="Date").reset_index(drop=True) #sort by date and resets index.

    return merged_ercot_df


def task_2_3_bonus(ercot_df):
    """
    Does Task 2,3 and the bonus. The Input is the ERCOT dataframe from task 1.

    Parameters
    ----------
    ercot_df : TYPE
        DESCRIPTION.

    Returns
    -------
    average_settle_year_month_df : TYPE
        DESCRIPTION.

    """
    #Task 2 Creates the dataframes for average price (across hours) for every Year-month pair
    # Create 'Year' and 'Month' columns
    ercot_df['Year'] = ercot_df['Date'].dt.year
    ercot_df['Month'] = ercot_df['Date'].dt.month
    
    #Group by settlement,year and month and print to csv
    settlement_group=ercot_df.groupby(["SettlementPoint","Year","Month"])
    average_settle_year_month_df=settlement_group["Price"].mean().reset_index()
    average_settle_year_month_df.to_csv("AveragePriceByMonth.csv", index=False)
    
    ##Bonus Plot the average price by hub/loadzone. Saved in current working directory.
    average_settle_year_month_df["Date"] = pd.to_datetime(average_settle_year_month_df[["Year", "Month"]].assign(DAY=1))
    hub_df = average_settle_year_month_df[average_settle_year_month_df["SettlementPoint"].str.startswith("HB_")]
    lz_df = average_settle_year_month_df[average_settle_year_month_df["SettlementPoint"].str.startswith("LZ_")]
    
    plt.figure(figsize=(12, 6))

    #loops through hubs and plots
    for point in hub_df["SettlementPoint"].unique():
        subset = hub_df[hub_df["SettlementPoint"] == point]
        plt.plot(subset["Date"], subset["Price"], label=point)
    
    plt.title("Monthly Average Price by Settlement Hub")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("SettlementHubAveragePriceByMonth.png")

    plt.figure(figsize=(12, 6))
    #loops through load zones and plots

    for point in lz_df["SettlementPoint"].unique():
        subset = lz_df[lz_df["SettlementPoint"] == point]
        plt.plot(subset["Date"], subset["Price"], label=point)
    
    plt.title("Monthly Average Price by Load Zone")
    plt.xlabel("Date")
    plt.ylabel("Average Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("LoadZoneAveragePriceByMonth.png")
    print("Tasks 2 and 3 completed")
    return average_settle_year_month_df


def task_4_5_6_bonus(ercot_df):
    """
    Takes the ercot_df as an input and computes volatility in the log returns of prices. Plots them as a bonus.

    Parameters
    ----------
    ercot_df : TYPE
        DESCRIPTION.

    Returns
    -------
    volatility_df : TYPE
        DESCRIPTION.
    max_vol_df : TYPE
        DESCRIPTION.

    """
    #Keeo only hubs
    hub_df = ercot_df[ercot_df["SettlementPoint"].str.startswith("HB_")].copy()

    # Create a year column
    hub_df["Date"] = pd.to_datetime(hub_df["Date"])
    hub_df["Year"] = hub_df["Date"].dt.year

    # Filter out non-positive prices
    hub_df = hub_df[hub_df["Price"] > 0].copy()

    # Sort values by SettlementPoint and Date
    hub_df.sort_values(by=["SettlementPoint", "Date"], inplace=True)

    # Compute log returns
    hub_df["LogReturn"] = np.log(hub_df["Price"] / hub_df.groupby("SettlementPoint")["Price"].shift(1))

    # Drop NaNs caused by shift
    hub_df = hub_df.dropna(subset=["LogReturn"])

    # Compute volatility (std dev of log returns) by SettlementPoint and Year
    volatility_df = (hub_df.groupby(["SettlementPoint", "Year"])["LogReturn"].std())
    volatility_df=volatility_df.reset_index()
    volatility_df=volatility_df.rename(columns={"LogReturn":"HourlyVolatility"})
    

    ###Task 5 Writes to csv
    volatility_df.to_csv("HourlyVolatilityByYear.csv", index=False)
    
    ##Task 6 Extracts max volatility for each year settlmentmeent point
    max_mask_df=volatility_df.groupby("Year")["HourlyVolatility"].idxmax()
    max_vol_df=volatility_df.loc[max_mask_df].reset_index(drop=True)              
    max_vol_df.to_csv("MaxVolatilityByYear.csv", index=False)
    
    
    #Bonus
    plt.figure(figsize=(12, 6))
    
    for hub in volatility_df["SettlementPoint"].unique():
        subset = volatility_df[volatility_df["SettlementPoint"] == hub].copy()
        subset["Date"] = pd.to_datetime(subset["Year"].astype(str) + "-01-01")
        plt.plot(subset["Date"], subset["HourlyVolatility"], label=hub)
    
    plt.title("Hourly Volatility by Hub and Year")
    plt.xlabel("Year")
    plt.ylabel("Hourly Volatility (Std. Dev. of Log Returns)")
    
    # Format x-axis for yearly ticks
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Hourly_volatility.png")
    print("task 4,5 and 6 completed.")
    return volatility_df,max_vol_df


def task_7(ercot_df):
    """
    Splits the historical price data into separate DataFrames for each SettlementPoint,
    keeping only SettlementPoint, Date, and Price columns.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with columns 'SettlementPoint', 'Date', 'Price'.

    Returns:
        dict: Dictionary where keys are SettlementPoint names and values are DataFrames.
    """

    
    # Keep only neccessary columns
    ercot_df = ercot_df[["SettlementPoint", "Date", "Price"]]
    
    # Create dictionary of DataFrames
    settlement_dic = {point: group.copy() 
                      for point, group in ercot_df.groupby("SettlementPoint")}
    


    # Loop over each SettlementPoint and save
    
    
    current_directory_str = os.getcwd() #gets current dictory
    formatted_spot_dir_str=os.path.join(current_directory_str, "formattedSpotHistory")
    os.makedirs(formatted_spot_dir_str, exist_ok=True)

    for point_name, df in settlement_dic.items():
        filename = f"spot_{point_name}.csv"
        filepath = os.path.join(formatted_spot_dir_str, filename)
        df.to_csv(filepath, index=False)
        
    print("Task 7 completed Files have been saved")
    
    return None

def shape_profile_bonus(ercot_df, settlement_point):
    """
    For a given SettlementPoint, create 24 lists (0-23 hours) of prices quoted at that hour.

    Parameters:
        df (pd.DataFrame): DataFrame with columns 'SettlementPoint', 'Date', 'Price'
        settlement_point (str): The settlement point to filter

    Returns:
        dict: Keys = hour (0-23), Values = list of prices for that hour
    """
    # Ensure Date is datetime
    ercot_df = ercot_df.copy()
    ercot_df["Date"] = pd.to_datetime(ercot_df["Date"])
    
    # Filter for the given settlement point
    sp_df = ercot_df[ercot_df["SettlementPoint"] == settlement_point]

    # Extract hour
    sp_df["Hour"] = sp_df["Date"].dt.hour

    # Create dictionary: hour -> list of prices
    hourly_prices_dict = {hour: sp_df.loc[sp_df["Hour"] == hour, "Price"].tolist()
                     for hour in range(24)}
    
    hourly_avg_dict = {hour: np.mean(prices) for hour, prices in hourly_prices_dict.items()}

    total_float = sum(hourly_avg_dict.values())
    normalized_dict= {hour: val / total_float for hour, val in hourly_avg_dict.items()}
    
    
        ### Plotting
    current_directory_str = os.getcwd()  # gets current directory
    formatted_shape_dir_str = os.path.join(current_directory_str, "hourlyShapeProfiles")
    os.makedirs(formatted_shape_dir_str, exist_ok=True)
    
    # File name for the saved plot
    filename = f"profile_{settlement_point}.png"
    filepath = os.path.join(formatted_shape_dir_str, filename)
    
    # Prepare data
    hours = list(normalized_dict.keys())
    values = list(normalized_dict.values())
    
    # Create plot
    plt.figure(figsize=(10, 5))
    plt.plot(hours, values, marker='o', linestyle='-', color='b')
    
    plt.title(f"Normalized Shape Price Across Hours - {settlement_point}")
    plt.xlabel("Hour of Day")
    plt.ylabel("Normalized Price")
    plt.grid(True)
    plt.xticks(range(24))
    plt.tight_layout()
    
    # Save plot to file
    plt.savefig(filepath, dpi=300)
    plt.close()  # closes the figure so it doesn't stay in memory
    
    print(f"Saved plot to {filepath}")

    return None


#task 1
data_prefix_str="ERCOT"
ercot_df=task_1(data_prefix_str)

#task 2,3
average_dfs=task_2_3_bonus(ercot_df)

#task 4,5,6
task_4_5_6_bonus(ercot_df)
settlments_list=list(ercot_df["SettlementPoint"].unique())

##task 7
task_7(ercot_df)

#bonus
for settlement_point in settlments_list:
    shape_profile_bonus(ercot_df,settlement_point)
