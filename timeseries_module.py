#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 11:14:51 2019

@author: bryanbutler
"""

# file to create all of the time series functions

# import basic libraries
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame


# stats libraries
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.tools.eval_measures import mse, rmse
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller




def adf_test(series,title=''):
    """
    Pass in a time series and an optional title, returns an ADF (Agumented Dickey-Fuller) report for stationarity

    Input: series selected in a time series format

    Output: returns the ADF stat and evaluation on stationarity

    """

    print(f'Augmented Dickey-Fuller Test: {title}')

    # .dropna() handles differenced data
    result = adfuller(series.dropna(),autolag='AIC')

    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val

    # to_string() removes the line "dtype: float64"
    print(out.to_string())

    # set the p value at .05
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")



# create the Durbin-Watson statistic
def durbin_watson_stat(data_frame):
    """
    Take in a data frame use OLS to build the residuals

    Returns the Durbin-Watson Statistic, best value = 2.00

    Pass in a residual series, retrieve DW statistic

    DW theory:

    DW = 2 * (1 - rho)

    DW = 2, rho = 0, not autocorrelation in errors

    DW between 0 and 2: positive first order autocorrelation, rho between 0 and 1

    DW between 2 and 4: negative first order autocorrelation, rho between 0 and -1



    """

    ols_res = OLS(data_frame, np.ones(len(data_frame))).fit()
    return durbin_watson(ols_res.resid)




# apply dw to a model
def get_dw(model):
    """
    Pass in a model, get the D-W statistic on the residuals

    """

    resid = model.resid
    return durbin_watson_stat(resid)



# make the model
def build_model(series, p, d, q, S, exog_data, P=None, D=None, Q=None):
    """
    Function to build SARIMAX model

    inputs:

        series = name of the series in the dataframe; should be specified in the following
        df['series_name'], series = 'series_name'

        p,d,q for arima modeling

        S: seasonal lag

        P,D,Q for seasonal modeling

        p,P: autoregressive components

        d,D: differencing components

        q,Q: moving average of error term components

        exog_data = matrix of exogenous variables can also be set to None

    default mode sets seasonal P, D, Q = p,d,Q

    Output;

    SARIMAX model results
    """
    if P is None:
        P = p
    if D is None:
        D = d
    if Q is None:
        Q = q
    model = SARIMAX(series, order=(p,d,q),
                    seasonal_order=(P,D,Q,S),
                    exog=exog_data,
                    enforce_invertibility=True)
    results = model.fit()
    return results


def backtest_model(model_results, exog_data, train, end, start=1, name='Backtest_Model'):
        """
        Create backtest values for the model to test against historical actual data using the training data set

        Inputs:
            model_results = used for the name of the model from build model

            exog_data: matrix of exogenous variables

            start: starting lag to model against, usually 1, not zero

            train: name of the training set to get the lengths from

            end: default to the length of the training set - 1 (len(train)-1)

            """
        results = model_results.predict(start=1,
                                        end=end,
                                        exog=exog_data,
                                        type='levels').rename(name)
        return results


def make_predictions(model_results, model_name, start, end, exog_data):
    """
    Predict model results given a start end end

    Inputs:
        model_result: name of model variable

        model_name: name to be used for model results

        start: where to start sequence at (integer)

        end: where to end predictions (integer)

        exog_data: matrix of exogenous variables
    """
    predictions = model_results.predict(start=start,
                                        end=end,
                                        exog=exog_data).rename(model_name)
    return predictions


def plot_fit(series, backtest_model, train, decimal = False, exogenous=False):
    """
    Make a plot that compares the model to the actual series via backtest

    Inputs:
        series: name of series of actual results

        backtest_model: the name of the results of the backtest model;
        usually backtest or xxx_backtest

        train: name of the training dataframe

        decimal: Include 2 digits in the decimal for the y axis plots; default is false
        which sets the float formatting to 0, 2 is the alternate to show x.xx values

        holiday is an exogenous series to include duummy variables that indicate holidays which can be important in time series
    """
    # series is training data
    ax = series.plot(figsize=(16,8), legend=True)
    ax1 = backtest_model.plot(legend=True)
    ax.legend(loc='upper left')


    # adjust the digits to be shown based on the decimal T/F value
    # applies to the y axis only

    if decimal:
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))

    else:
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))


    # add the exogenous indicators in
    if exogenous:
        for day in train.query('exogenous==1').index:
            # add in a vertical line there
            ax.axvline(x=day, color='red', linestyle='--', alpha=.25);


def prediction_plot(series, predictions, df, exogenous_df, decimal=False):
    """
    Plot the predictions vs the actual series

    Inputs:
        series: the name of the actual data series, usually the test data

        predictions: name of predictions data series

        exogenous_df: dataframe of exogenous dummy variables called exogenous

        decimal: Include 2 digits in the decimal, default is false which sets the float
        formatting to 0, 2 is the alternate to show x.xx values

    Output:
        returns a plot of both series

    """

    ax = series.plot(figsize=(16,8), legend=True)
    ax1 = predictions.plot(legend=True)


    # adjust the digits to be shown based on the decimal T/F value
    # applies to the y axis only

    if decimal:
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))

    else:
        ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))



    for day in exogenous_df.query('exogenous==1').index:
        # add in a vertical line there
        ax.axvline(x=day, color='red', linestyle='--', alpha=.25);



# make a dataframe to compare predictions
def compare_results(test_data, predictions, wdf):
    """
    Return a new dataframe of test results and predictions

    Inputs:
        test_data: the test data set (series)

        predictions: series of prdictions

        df: weekly dataframe of all values,

    Output:

        dataframe containing the two series for plotting
    """
    new_df = pd.concat([test_data, predictions], axis = 1 )

    end_date = df.index[-1]

    new_df = new_df[:end_date]

    return new_df



# calculate periodic error
def error_calcs(df, modeled, actual):
    """
    Calculate the error at each prediction point

    Input:
        df: name of the dataframe created from the compare results

    Output:
        Error: column calculating the actual error between the prediction and actual;
        calculation is based on actual - predicted

        Percent: converts the error to a percentage of the actual

    Returns a dataframe styled with clean formatting, and 2 decimal points for percent
    """
    df['Error'] = actual - modeled
    df['Percent'] = df['Error']/actual* 100

    list_cols = df.columns

    # adjust the format for the columns
    return df.style.format({list_cols[0]: "{:,.0f}".format,
                            list_cols[1]: "{:,.0f}".format,
                            list_cols[2]: "{:,.0f}".format,
                            list_cols[3]: "{:,.2f}".format})




# get a roll up of the total error
def calculate_total_error(actual, predictions, df):
    """
    Calculate root mean square error (RMSE), mean and error as a percentage of mean

    Inputs:
        actual: values of actual data series

        predictions: values of prediction data series

        df: dataframe of all values

    Outputs:
        root mean squared error of the two series

        mean of the actual series

        percent: percentage of rmse of the actual mean


    Means and errors are formatted as integers

    Percent is formatted as one decimal point
    """

    end_date = df.index[-1]


    actual = actual[:end_date]
    predictions = predictions[:end_date]



    error = rmse(actual, predictions)
    print(f'{error:.0f}', 'RMSE')

    CancMean = actual.mean()
    print(f'{CancMean:.0f}', 'Mean')

    percent = error/CancMean*100
    print(f'{percent:.1f}', '% Error')



# create the confidence intervals around the forecasts
def get_conf_interval(model, actual, steps_ahead,
                      predictions, exog_data, alpha = 0.05):
    """
    Create upper and lower confidence intervals around the predictions

    Inputs:
        model: the model to get the forecasts

        actual: actual data series

        steps_ahead: number of steps ahead that model is forecasting (integer)

        predictions: prediction series

        exog_data: matrix of exogenous data, default is normally test[['holiday']]

        alpha: amount in the tails, at 0.20, 80% confidence intervals

    Output:
        Dataframe with actual, predictions, lower confidence interval of predictions and
        upper confidenc interval of predictions

    """
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data, alpha = alpha)

    conf_df = pd.concat([actual,predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    end_date = df.index[-1]

    conf_df = conf_df.loc[:end_date]


    return conf_df.style.format("{:,.0f}")




#### OUT OF SAMPLE FUNCTIONS ###########
# Functions to be used with out of series forecasting when testi data is not available.


def get_oos_conf_interval(model, steps_ahead, exog_data, alpha=0.05):
    """
    Get confidence intervals for out of sample forecasts

    This differs from the other function in that it

    Inputs:
        model: name of model

        steps_ahead: nupber of steps ahead for forecasting (integer)

        exog_data: matrix of out of series exogenous data for oos;
        generall oos[['holiday']]

        alpha: amount in the tails, at 0.05, 95% confidence intervals


    Outputs:

        Returns a style object with predictions, lower confidence interval, upper confidence interval

    """
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data, alpha = alpha)


    conf_df = pd.concat([predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    # return conf_df.style.format("{:,.0f}")
    return conf_df

# create a dataframe for oos plotting
def make_oos_plot_df(model, steps_ahead, exog_data):
    """
    Create a dataframe of values to be used in plotting the out of sequence values and intervals

    Inputs:
        model: name of model used to develop forecasts

        steps_ahead: number of steps ahead for forecasting

        exog_data: matrix of exogenous variables, usually oos_exog or related matrix

    Output:
        returns a dataframe of predictions, lower confidence interval and upper confidence interval;
        since this is out of series (oos), no actuals are available
    """
    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data)
    conf_df = pd.concat([predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    # convert it to a dataFrame
    conf_df = pd.DataFrame(conf_df)

    return conf_df




def plot_oos(conf_df, df, series, backtest, fut_exog, start_date, decimal = False):
    """
    Make a plot of the series, out of sample forecasts and shaded confidence intervals

    Inputs:
        conf_df: name of the dataframe created using the make_oos_plot_df

        df: name of the dataframe that holds the series of actual values

        series: name of the series of actual values

        backtest: name of the series holding the backtest resultsu

        fut_exog: matrix of future dates and exogenous variables, in this case the dataset should have a variable called exogenous or change the name

        start_date: starting date to show the window of actual values; it is in a format of 'YYYY-MM-DD'

    Output:
        returns a plot with the actual series, the modeled series, confidence interval and shaded
        region within the conidence intervals

    """
    # make a plot of model fit
    # color = 'skyblue'

    fig = plt.figure(figsize = (16,8))
    ax = fig.add_subplot(111)

    # set the x value to the index
    x = conf_df.index.values

    # get the confidence interval series
    upper = conf_df['upper ' + series]
    lower = conf_df['lower ' + series]

    # add the actual data starting at the start_date
    ax = df[series].loc[start_date:].plot(figsize=(16,8),
                                          legend=True,
                                          label = 'Actual ' + series,
                                          linewidth=2)

    # add in the backtest series
    ax0 = backtest.loc[start_date:].plot(color = 'orange', label = 'Model backtest')

    # plot the predictions
    ax1 = conf_df['Predictions'].plot(color = 'red',label = 'Predicted ' + series )

    # plot the uper and lower confidence bounds
    ax2 = upper.plot(color = 'grey', label = 'Upper CI')
    ax3 = lower.plot(color = 'grey', label = 'Lower CI')

    # plot the legend for the first plot
    plt.legend(loc = 'lower left', fontsize = 12)


    # fill between the conf intervals
    plt.fill_between(x, lower, upper, color='grey', alpha='0.2')

    # add the exogenous indicators in for the actuals
    # if the df contains another flag such as weekend, then
    # set is as 'weekend==1'
    for day in df.query('exogenous==1').index:
        # add in a vertical line there
        ax.axvline(x=day, color='red', alpha=.2)

    # add the holidays in for the ooos
    for day in fut_exog.query('exogenous==1').index:
        # add in a vertical line there
        ax.axvline(x=day, color='red', alpha=.2)

    # adjust the digits to be shown based on the decimal T/F value
    # applies to the y axis only

    if decimal:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))

    else:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))

    plt.show();


def create_final_oos_df(model, steps_ahead, exog_data, series, add_date=False):
    """
    Create a dataframe for the oos predictions values to be imported and used later

    Inputs:

        model: name of model for forecasting

        steps_ahead: number of steps ahead for weekly forecast

        exog_data: matrix of time series with exogenous data

        add_date: boolean value to add the current date as a column for date of forecast

    Output:
        returns a formatted data frame of out of series predictions, lower confidence interval,
        upper confidence interval, and date of fcast if add_date set to True
    """
    from datetime import datetime as dt

    predictions_int = model.get_forecast(steps=steps_ahead,exog=exog_data)
    conf_df = pd.concat([predictions_int.predicted_mean, predictions_int.conf_int()], axis = 1)
    # rename the columns
    conf_df = conf_df.rename(columns={0: 'Predictions', 1: 'Lower CI', 2: 'Upper CI'})

    # convert it to a dataFrame
    conf_df = pd.DataFrame(conf_df)

    # rename the columns
    conf_df.columns = [series, 'lower_' + series, 'upper_' + series]

    # add date of forecast if add_date = true
    if add_date:
        conf_df['fcast_date'] = dt.now().date()

    return conf_df











