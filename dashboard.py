import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import global_data

def main():
    st.title("Dashboard")
    
    # # Get global data
    # df_sales = global_data.get_data()
    
    # # Header
    # st.header("Welcome to the Dashboard")
    
    # # Summary statistics
    # st.subheader("Summary Statistics")
    # data = {
    #     'Metric': ['Sales', 'Revenue', 'Profit', 'Customers'],
    #     'Value': [df_sales['Sales'].sum(), df_sales['Sales'].sum() * 10, df_sales['Sales'].sum() * 0.2, len(df_sales)]
    # }
    # df_summary = pd.DataFrame(data)
    
    # # Using columns for summary statistics
    # col1, col2, col3, col4 = st.columns(4)
    # col1.metric("Sales", df_summary.loc[0, 'Value'])
    # col2.metric("Revenue", df_summary.loc[1, 'Value'])
    # col3.metric("Profit", df_summary.loc[2, 'Value'])
    # col4.metric("Customers", df_summary.loc[3, 'Value'])
    
    # # Divider
    # st.markdown("---")
    
    # # Charts with Expander
    # with st.expander("Sales Over Time"):
    #     st.subheader("Sales Over Time")
        
    #     fig, ax = plt.subplots()
    #     ax.plot(df_sales['Date'], df_sales['Sales'], marker='o')
    #     ax.set_title('Sales Over Time')
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('Sales')
    #     st.pyplot(fig)
    
    # # Data Table with Expander
    # with st.expander("Example Data Table"):
    #     st.subheader("Example Data Table")
    #     st.dataframe(df_sales)
    
    # # Additional plots and filters
    # with st.expander("Additional Analysis"):
    #     st.subheader("Filter by Date")
    #     start_date = st.date_input("Start date", pd.to_datetime('2021-01-01'))
    #     end_date = st.date_input("End date", pd.to_datetime('2021-04-10'))
        
    #     # Ensure start_date and end_date are in datetime64[ns] format
    #     start_date = pd.to_datetime(start_date)
    #     end_date = pd.to_datetime(end_date)
        
    #     filtered_data = df_sales[(df_sales['Date'] >= start_date) & (df_sales['Date'] <= end_date)]
    #     st.write(f"Data from {start_date.date()} to {end_date.date()}")
    #     st.dataframe(filtered_data)

    #     # Bar chart example
    #     st.subheader("Sales Distribution")
    #     fig, ax = plt.subplots()
    #     ax.hist(df_sales['Sales'], bins=20)
    #     ax.set_title('Sales Distribution')
    #     ax.set_xlabel('Sales')
    #     ax.set_ylabel('Frequency')
    #     st.pyplot(fig)

if __name__ == "__main__":
    main()
