import streamlit as st
import pandas as pd
import backend as _bk
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

st.title("Ticket sales prediction")

st.write("Upload an Excel file with ticket details to predict the ticket sales.")
# Define the required columns
required_columns = ['EventType', 'StartDate', 'StatusCreatedDate', 'GroupSize']
# Provide information about the required columns
st.write("The uploaded file must contain the following columns (at least):")
st.write(required_columns)



template_df = pd.DataFrame(columns=required_columns)
csv = convert_df(template_df)

st.download_button(
    label="Download template as CSV",
    data=csv,
    file_name='events_template.csv',
    mime='text/csv',
)


st.header('OR')

# Create a file uploader widget
uploaded_file = st.file_uploader("Upload a file directly", type=["xlsx", "xls", "csv"])
# Check if a file is uploaded
if uploaded_file is not None:
    # Determine the file type (Excel or CSV) based on the file extension
    file_extension = uploaded_file.name.split(".")[-1]

    if file_extension in ["xlsx", "xls"]:
        # Load the uploaded Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)
    elif file_extension == "csv":
        # Load the uploaded CSV file into a DataFrame
        df = pd.read_csv(uploaded_file)

    # Check if required columns are present in the uploaded file
    missing_columns = [col for col in required_columns if col not in df.columns]
    if len(missing_columns) == 0:
        st.success("Required columns found in the uploaded file.")
        unique_events = _bk.get_all_unique_events(df)
        # Convert the dictionary to a list of tuples for display in the selectbox
        event_options = list(unique_events.items())
        place_holder = [(0,"Select an event from the drop down")]
        event_options = place_holder + event_options
        st.subheader(f":blue[{len(event_options)}] _unique event(s)_ detected")
        #st.write("event(s) detected")
        selected_event = st.selectbox(
        'pick an option',
        event_options,label_visibility="hidden")

        if selected_event[0] != 0:
            event_id = selected_event[0]
            event_name = selected_event[1]
            total_tickets_sold_df = _bk.calculate_total_tickets_sold_per_event(df)
            event = total_tickets_sold_df[total_tickets_sold_df['EventId'] == event_id]
            total_tickets_sold =  event['TotalTicketsSold'].values[0]


            specific_org_df = _bk.get_event_by_id(event_id=event_id,df=df)
            weekly_ticket_sales_withdate = _bk.calculate_weekly_ticket_sales_per_event_with_date(specific_org_df)
            merged_df = _bk.merge_datasets(specific_org_df,weekly_ticket_sales_withdate)
            engineered_df = _bk.engineer_features(merged_df=merged_df)
            print(engineered_df.columns)
            # Convert WeekOfYear to int type
            engineered_df['WeekOfYear'] = engineered_df['WeekOfYear'].astype(int)

            # Define features and target
            features = ['EventType', 'Week', 'WeekOfYear', 'DayOfWeek', 'Month', 'TicketsSold_t-3', 'MovingAverage', 'DayOfYearSin', 'DayOfYearCos']
            target = 'TicketsSold'

            # Define hyperparameters for the XGBoost model
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }

            # Train and evaluate models for the selected events
            #for event_id in unique_event_ids:
            #event_data = engineered_df[engineered_df['EventId'] == event_id]

                # Sort the data chronologically based on the 'WeekStartDate'
            event_data = engineered_df.sort_values(by='WeekStartDate')

                # Calculate the index to split the data
            split_index = int(len(event_data) * 0.8)  # 80% train, 20% test

            train_data = event_data[:split_index]
            test_data = event_data[split_index:]

            X_train = train_data[features]
            y_train = train_data[target]
            X_test = test_data[features]
            y_test = test_data[target]

                # Define and train the XGBoost model
            xgb_model = XGBRegressor(**xgb_params)
            xgb_model.fit(X_train, y_train)

                # Make predictions
            y_pred = xgb_model.predict(X_test)

                # Evaluate the model
            mae = mean_absolute_error(y_test, y_pred)
            print(f"Event {event_id} - Mean Absolute Error: {mae}")

                

            plt.figure(figsize=(10, 6))
            # # Convert WeekEndDate to the desired format (MM/DD)
            # week_end_dates = event_data['WeekEndDate'].dt.strftime('%m/%d')

            # plt.plot(week_end_dates, event_data['TicketsSold'], label='Actual')
            # plt.plot(week_end_dates.iloc[-len(y_pred):], y_pred, label='Predicted')  # Use iloc to match prediction length
            # plt.xlabel('Week End Date')
            # plt.ylabel('Number of Tickets Sold')
            # plt.title(f'Event {event_name} - Actual vs Predicted Tickets Sold')
            # plt.legend()

            # # Rotate x-axis labels for better readability
            # plt.xticks(rotation=45)

            # # Display the plot using st.pyplot()
            # st.pyplot(plt)


            #import matplotlib.pyplot as plt

            # plt.figure(figsize=(10, 6))
            # plt.bar(event_data['WeekEndDate'].dt.strftime('%m/%d'), event_data['TicketsSold'], label='Actual')
            # plt.plot(test_data['WeekEndDate'].dt.strftime('%m/%d'), y_pred, marker='o', label='Predicted', color='orange')
            # plt.xlabel('Week End Date')
            # plt.ylabel('Number of Tickets Sold')
            # plt.title(f'Event {event_id} - Actual vs Predicted Tickets Sold')
            # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            # plt.legend()
            # plt.tight_layout()
            # plt.show()
            # st.pyplot(plt)





            plt.plot(event_data['Week'], event_data['TicketsSold'], label='Actual')
            plt.plot(test_data['Week'], y_pred, label='Predicted')
            plt.xlabel('Week')
            plt.ylabel('Number of Tickets Sold')
            plt.title(f'Event {event_id} - Actual vs Predicted Tickets Sold')
            plt.legend()
            # Display the plot using st.pyplot()
            st.pyplot(plt)

            # Calculate event metadata
            week_tickets_sold = event_data.groupby('WeekEndDate')['TicketsSold'].sum().reset_index()
            # total_tickets_sold = week_tickets_sold['TicketsSold'].sum()
            num_weeks = event_data['Week'].max()
            quartile_tickets = event_data['TicketsSold'].quantile([0.25, 0.5, 0.75]).tolist()


            # Display event metadata
            st.write(f"Total Tickets Sold: {total_tickets_sold}")
            st.write(f"Number of Weeks ticket sales ran for: {num_weeks}")
            
    else:
        st.error(f"The uploaded file is missing the following required columns: {missing_columns}")
    
        


