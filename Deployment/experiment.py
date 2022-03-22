import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import datetime
import yfinance as yf
from PIL import Image

st.set_page_config(page_title= 'Stock Forecasting 📈', page_icon = '📈', layout= 'wide',
                   initial_sidebar_state= 'expanded',
                   menu_items= {'Get help': 'https://streamlit.io/gallery?category=finance-business'})

companies = pd.read_csv('EQUITY_L.csv')


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
        # print('Sorry, but only 2 month data can be extracted for given interval')
        current_time = datetime.datetime.now()
        month_value = current_time.month - 2
        # """
        # Following can be used to get 2 month previous date
        # datetime.datetime.now()-datetime.timedelta(days=60)
        # """
        starting_date = current_time.replace(month= month_value)
    elif interval == '1m':
        starting_date = datetime.datetime.now()-datetime.timedelta(days=6)

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

def candle_plot_fun(data, interval):
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    fig.update_layout(xaxis_rangeslider_visible=False, hovermode='x unified')
    if 'm' in interval or 'h' in interval:
        fig.update_xaxes(rangebreaks = [dict(values = gap_calculate(data)), dict(pattern='hour', bounds=[15.5, 9.24])])
    else:
        fig.update_xaxes(rangebreaks=[dict(values=gap_calculate(data))])
    st.plotly_chart(fig, use_container_width= True)
    return fig

def volume_plot(data, interval):
    area = px.area(data_frame=data,
                   x=data.index,
                   y='Volume', markers=True,
                   hover_data=['High', 'Low'])
    area.update_traces(line_color='Blue')
    area.update_layout(xaxis_rangeslider_visible=False, hovermode='x unified')
    if 'm' in interval or 'h' in interval:
        area.update_xaxes(rangebreaks=[dict(values=gap_calculate(data)), dict(pattern='hour', bounds=[15.5, 9.24])])
    else:
        area.update_xaxes(rangebreaks=[dict(values=gap_calculate(data))])
    st.plotly_chart(area, use_container_width=True)
    return area

    st.plotly_chart(area)
    
def weeks_high_low(data):
    date_52week = data.index.max()-datetime.timedelta(weeks=52)

def get_symbol(selection):
    index_no = companies.index[companies['NAME OF COMPANY'] == selection].values
    SYMBOL = companies.loc[index_no, 'SYMBOL'].values
    SYMBOL = SYMBOL[0]+'.NS'
    return SYMBOL


def model_dataset(data):
    pass

def forecasting(model, data):
    pass



def main():
    # st.title('Stock Forecasting')
    st.markdown("<h1 style= 'text-align:center'> Stock Forecasting and Analysis </h1> ", unsafe_allow_html=True)
    _, col,_ = st.columns([1,3,1])
    share_image = Image.open('business-growth.jpg')
    col.image(share_image)

    col1, interval_col, period_col = st.columns([3,1,1])
    with col1:
        selection = st.selectbox(
            label= 'Select Company',
            options = companies['NAME OF COMPANY']
        )
    with interval_col:
        interval = st.selectbox(
            label = 'Select interval of stocks',
            options = ['1m' ,'2m','5m','15m','30m','1h','1d','5d','1wk','1mo','3mo'],
            index = 6
        )

    with period_col:
        period = st.number_input(
            label= 'Select period of data (in years)',
            min_value = 1,
            max_value = 10,
            value = 1,
            step = 1
        )
        period = int(period)
    SYMBOL = get_symbol(selection)

    stock_data= fetch_data(symbol=SYMBOL, interval= interval, data_of_years=period )
    st.dataframe(stock_data, width = 1200)

    candle_col, volume_col = st.columns(2)
    with candle_col:
        st.subheader('Candle plot')
        candle_plot_fun(stock_data, interval)

    with volume_col:
        st.subheader('Volume plot')
        volume_plot(stock_data, interval)


    data_for_model = fetch_data(symbol=SYMBOL, data_of_years = 5)



if __name__ == '__main__':
    main()