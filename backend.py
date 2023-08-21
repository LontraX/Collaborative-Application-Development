import pandas as pd
from xgboost import XGBRegressor
import numpy as np

def get_all_unique_events(df):
    # Drop duplicate rows based on 'EventId'
    unique_events = df.drop_duplicates(subset=['EventId'])

    # Create a dictionary of EventId to EventName
    unique_event_dict = {}
    for index, row in unique_events.iterrows():
        unique_event_dict[row['EventId']] = row['EventName']

    return unique_event_dict

def get_event_by_id(event_id, df):
    # Filter the dataframe to get the specific event using loc
    specific_event = df.loc[df['EventId'] == event_id]
    return specific_event


# this method calculates the number of weekly tickets sold for each event,and also includes the week's start and end date. 
def calculate_weekly_ticket_sales_per_event_with_date(org_df):
    # Convert 'StartDate' and 'StatusCreatedDate' to datetime format
    org_df['StartDate'] = pd.to_datetime(org_df['StartDate'])
    org_df['StatusCreatedDate'] = pd.to_datetime(org_df['StatusCreatedDate'])

    # Create a new DataFrame to store the weekly ticket sales with dates
    weekly_ticket_sales_withdate = []

    # Iterate through each unique event
    for event_id, event_data in org_df.groupby('EventId'):
        event_name = event_data['EventName'].values[0]

        # Get the earliest status created date and the event start date
        start_date = event_data['StatusCreatedDate'].min()
        end_date = event_data['StartDate'].values[0]

        # Calculate the number of weeks between start and end dates
        weeks_between = (end_date - start_date).days // 7 + 1

        # Calculate the number of tickets sold for each week
        for week_number in range(1, weeks_between + 1):
            week_start = start_date + pd.DateOffset(weeks=week_number - 1)
            week_end = start_date + pd.DateOffset(weeks=week_number) - pd.DateOffset(days=1)
            tickets_sold = event_data[
                (event_data['StatusCreatedDate'] >= week_start) &
                (event_data['StatusCreatedDate'] <= week_end)
            ]['GroupSize'].sum()

            # Append the weekly ticket sales to the DataFrame with dates
            weekly_ticket_sales_withdate.append({
                'EventId': event_id,
                'EventName': event_name,
                'Week': week_number,
                'TicketsSold': tickets_sold,
                'WeekStartDate': week_start,  # Add the start date for the week
                'WeekEndDate': week_end  # Add the end date for the week
            })

    # Create the final DataFrame for weekly ticket sales with dates
    weekly_ticket_sales_withdate_df = pd.DataFrame(weekly_ticket_sales_withdate)
    return weekly_ticket_sales_withdate_df

def calculate_weekly_ticket_sales_per_event(org_df):
    # Convert 'StartDate' and 'StatusCreatedDate' to datetime format
    org_df['StartDate'] = pd.to_datetime(org_df['StartDate'])
    org_df['StatusCreatedDate'] = pd.to_datetime(org_df['StatusCreatedDate'])

    # Create a new DataFrame to store the weekly ticket sales
    weekly_ticket_sales = []
    # Iterate through each unique event
    for event_id, event_data in org_df.groupby('EventId'):
        event_name = event_data['EventName'].values[0]

        # Get the earliest status created date and the event start date
        start_date = event_data['StatusCreatedDate'].min()
        end_date = event_data['StartDate'].values[0]

        # Calculate the number of weeks between start and end dates
        weeks_between = (end_date - start_date).days // 7 + 1

        # Calculate the number of tickets sold for each week
        for week_number in range(1, weeks_between + 1):
            week_start = start_date + pd.DateOffset(weeks=week_number - 1)
            week_end = start_date + pd.DateOffset(weeks=week_number) - pd.DateOffset(days=1)
            tickets_sold = event_data[
                (event_data['StatusCreatedDate'] >= week_start) &
                (event_data['StatusCreatedDate'] <= week_end)
            ]['GroupSize'].sum()

            # Append the weekly ticket sales to the DataFrame
            weekly_ticket_sales.append({
            'EventId': event_id,
            'EventName': event_name,
            'Week': week_number,
            'TicketsSold': tickets_sold
            })

    # Create the final DataFrame for weekly ticket sales
    weekly_ticket_sales_df = pd.DataFrame(weekly_ticket_sales)
    return weekly_ticket_sales_df


def calculate_total_tickets_sold_per_event(org_df):
    # Create a new DataFrame to store the total ticket sales for each event
    total_ticket_sales = []
    # Iterate through each unique event
    for event_id, event_data in org_df.groupby('EventId'):
        event_name = event_data['EventName'].values[0]

        # Calculate the total tickets sold for the event
        total_tickets_sold = event_data['GroupSize'].sum()

        # Append the total ticket sales to the DataFrame
        total_ticket_sales.append({
            'EventId': event_id,
            'EventName': event_name,
            'TotalTicketsSold': total_tickets_sold
            })
        # Create the final DataFrame for total ticket sales
    total_ticket_sales_df = pd.DataFrame(total_ticket_sales)
    return total_ticket_sales_df


def merge_datasets(main_org_df, weekly_ticket_sales_withdate_df):
    # Extract the relevant columns from event_details_df
    event_details_subset = main_org_df[['EventId',  'EventType', 'StartDate']]

    # Merge weekly_sales_df with the extracted columns from event_details_df
    merged_df = pd.merge(weekly_ticket_sales_withdate_df, event_details_subset, on=['EventId'], how='left')
    return merged_df

def engineer_features(merged_df):
    # Feature Engineering
    merged_df['EventType'] = merged_df['EventType'].astype('category').cat.codes
    #merged_df['TicketType'] = merged_df['TicketType'].astype('category').cat.codes
    merged_df['WeekOfYear'] = merged_df['StartDate'].dt.isocalendar().week
    merged_df['DayOfWeek'] = merged_df['StartDate'].dt.dayofweek
    merged_df['Month'] = merged_df['StartDate'].dt.month

    # Calculate lag features for previous weeks' ticket sales
    lag_weeks = [1, 2, 3]
    for lag in lag_weeks:
        merged_df[f'TicketsSold_t-{lag}'] = merged_df.groupby('EventId')['TicketsSold'].shift(lag)

    # Calculate moving averages
    rolling_window = 3
    merged_df['MovingAverage'] = merged_df.groupby('EventId')['TicketsSold'].rolling(rolling_window).mean().reset_index(0, drop=True)
    
    # Convert day of year to sine and cosine components
    merged_df['DayOfYear'] = merged_df['StartDate'].dt.dayofyear
    merged_df['DayOfYearSin'] = np.sin(2 * np.pi * merged_df['DayOfYear'] / 365)
    merged_df['DayOfYearCos'] = np.cos(2 * np.pi * merged_df['DayOfYear'] / 365)

    # Drop columns not needed for modeling
    columns_to_drop = [ 'StartDate', 'DayOfYear']
    merged_df = merged_df.drop(columns=columns_to_drop)
    return merged_df

def make_prediction(event_df):
    # Convert WeekOfYear to int type
    event_df['WeekOfYear'] = event_df['WeekOfYear'].astype(int)

    # Define features and target
    features = ['EventType', 'WeekOfYear', 'DayOfWeek', 'Month', 'TicketsSold_t-3', 'MovingAverage',  'DayOfYearSin', 'DayOfYearCos']
    target = 'TicketsSold'

    # Define hyperparameters for the XGBoost model
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }



