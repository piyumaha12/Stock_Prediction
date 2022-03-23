import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import datetime
import yfinance as yf
from PIL import Image

st.set_page_config(page_title='Stock Forecasting 📈', page_icon='📈', layout='wide',
                   initial_sidebar_state= 'expanded',
                   menu_items={'Get help': 'https://streamlit.io/gallery?category=finance-business'})

companies = pd.read_csv('EQUITY_L.csv')
data_for_model = None


# Function to retrive data from y_finance

@st.cache
def fetch_data(symbol, interval='1d', data_of_years=1):
    ''':param
    - symbol: Pass symbol on respective company
    - interval: pass the string of interval of data that you want to
                extract from [2m,5m,15m,30m, 60m, 1d,]
    - data_of_years: for how many of years of data you wants
    '''
    global data_for_model
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
    data_for_model = data
    return data

# def save_to_csv(data, file_name,):
#     file_name = location+file_name
#     data.to_csv(file_name)

@st.cache
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


@st.cache
def weeks_high_low(data):
    date_52week = data.index.max()-datetime.timedelta(weeks=52)
    high, low = data['High'][date_52week:].max(), data['Low'][date_52week:].min()
    closest_date = data.index.max()
    pre_open = None
    pre_close = None
    i = 1
    while pre_open is None:
        previous_day = closest_date - datetime.timedelta(days = i)
        try:
            pre_open = data['Open'][previous_day]
            pre_close = data['Close'][previous_day]
        except:
            i +=1
            continue;

    open = data['Open'][closest_date]
    close = data['Close'][closest_date]
    high = round(high, 4)
    open = round(open, 4)
    low = round(low, 4)
    close = round(close, 4)
    open_change = round((open - pre_open), 4)
    close_change = round((close - open), 4)
    return high, low, open, close, open_change, close_change

@st.cache
def get_symbol(selection):
    index_no = companies.index[companies['NAME OF COMPANY'] == selection].values
    SYMBOL = companies.loc[index_no, 'SYMBOL'].values
    SYMBOL = SYMBOL[0]+'.NS'
    return SYMBOL

@st.cache
def model_dataset(data):
    pass

@st.cache
def forecasting(model, data):
    pass

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


def main():

    selection_side = st.sidebar.selectbox('SELECT FROM BELOW',
                                     (
                                         'HOME', 'ANALYSE',
                                         'FORECASTING'
                                     ))

    if selection_side == 'HOME':
        global data_for_model
        # st.title('Stock Forecasting')
        st.markdown("<h1 style= 'text-align:center'> Stock Forecasting and Analysis </h1> ", unsafe_allow_html=True)
        _, col,_ = st.columns([0.2,5,0.2])
        share_image = Image.open('bull_bear.jpg')
        col.image(share_image)


        company_col, interval_col, period_col = st.columns([3,1,1])
        with company_col:
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
        st.write('')


        open_col, close_col, high_col, low_col, download_col = st.columns([1,1,1,1,4])
        high, low, open, close, open_change, close_change = weeks_high_low(stock_data)
        with open_col:
            st.metric('Open', open, open_change)
        with close_col:
            st.metric('Close', close, close_change)
        with high_col:
            st.metric('52weeks High', high)
        with low_col:
            st.metric('52weeks Low', low)

        dataframe_col, col = st.columns([4,1])
        dataframe_col.dataframe(stock_data, width = 1400)

        st.download_button(
            label="Download data as CSV",
            data=convert_df(stock_data),
            file_name= selection+'.csv',
            mime='text/csv',
        )

        # candle_col, volume_col = st.columns(2)
        # with candle_col:
        st.subheader('Candle plot')
        candle_plot_fun(stock_data, interval)

        # with volume_col:
        st.subheader('Volume plot')
        volume_plot(stock_data, interval)
        data_for_model = fetch_data(symbol=SYMBOL, data_of_years = 6)

    elif selection_side == 'ANALYSE':
        global data_for_model
        st.header('Stock Analysis using Moving Average')
        data_for_model['MA2'] = data_for_model.Close.rolling(2).mean()
        fig = px.line(data_frame=data_for_model.iloc[:100, :],
                      x=data_for_model.index[:100],
                      y='MA2')
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig)


if __name__ == '__main__':
    main()