import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import datetime
import yfinance as yf

st.set_page_config(page_title= 'Bankruptcy Prediction 📈', page_icon = '💰', layout= 'wide',
                   initial_sidebar_state= 'expanded',
                   menu_items= {'Get help': 'https://streamlit.io/gallery?category=finance-business'})


PATH = 'H:\Excelr\Project\Stock_Prediction\csv_files\Tatamotors_5years.csv'


# Function to retrive data from y_finance

@st.cache
def fetch_data(symbol, interval='1d', data_of_years=1):
    ''':param
    - symbol: Pass symbol on respective company
    - interval: pass the string of interval of data that you want to
                extract from [2m,5m,15m,30m, 60m, 1d,]
    - data_of_years: for how many of years of data you wants
    '''

    if interval in ['2m', '5m', '15m', '30m']:
        print('Sorry, but only 2 month data can be extracted for given interval')
        current_time = datetime.datetime.now()
        month_value = current_time.month - 2
        """
        Following can be used to get 2 month previous date
        datetime.datetime.now()-datetime.timedelta(days=60)
        """
        starting_date = current_time.replace(month= month_value)
    else:
        year_value = datetime.datetime.now().year-data_of_years
        ending_date = datetime.datetime.now()
        starting_date = ending_date.replace(year=year_value)
    data = yf.Ticker(symbol)
    data = data.history(interval=interval, start= starting_date)
    return data

# def save_to_csv(data, file_name,):
#     file_name = location+file_name
#     data.to_csv(file_name)

def gap_calculate(data):
    dt_all_hr = pd.date_range(start=data.index[0],end=data.index[-1])# retrieve the dates that ARE in the original datset
    dt_obs_hr = [d.strftime("%Y-%m-%d %H:%M:%S") for d in pd.to_datetime(data.index)]# define dates with missing values
    dt_breaks_hr = [d for d in dt_all_hr.strftime("%Y-%m-%d %H:%M:%S").tolist() if not d in dt_obs_hr]
    return dt_breaks_hr

def candle_plot_fun(data):

    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    fig.update_layout(xaxis_rangeslider_visible=False, hovermode='x unified')
    fig.update_xaxes(rangebreaks = [dict(values = gap_calculate(data))])
    st.plotly_chart(fig, use_container_width= True)
    return fig


def main():
    st.markdown("<h1 style= 'text-align:center'> Stock Forecasting and Analysis </h1> ", unsafe_allow_html=True)

    
    tatamotors_5year= fetch_data(symbol='TATAMOTORS.NS', data_of_years= 1)
    st.dataframe(tatamotors_5year, width = 1000)

    fig =candle_plot_fun(tatamotors_5year)

    st.write('Grkgd')


if __name__ == '__main__':
    main()