# TimeSeriesModule
Functions to perform SARIMAX time series modeling in Python.

The functions consist of the following:

1. adf: testing for Augmented Dickey-Fuller testing for stationarity
2. durbin_watson_stat: create the Durbin-Watson statistic to evaluate the fit of the model to the series and check the amount of autocorrelation to be removed
3. get_dw: returns the Durbin-Watson statistic
4. build_model: builds the SARIMAX model and outputs the summary
5. backtest_model: rerun the model in backtest mode to create values to check goodness of fit
6. make_predictions: make preditions from the model
7. plot_fit: plot the backtest model against the actual series
8. prediction_plot: plot the predictions against the test series
9. compare_results: make a data frame to compare the predictions to the testing data
10. error_calcs: calculate the actual error difference and the percent difference at each prediction interval
11. calculate_total_error: get the overall RMSE
12. get_conf_interval: get confidence intervals around the predictions
13. get_oos_conf_interval: creates a confidence interval for out of series(OOS) forecasting, no comparison for testing
14. make_oos_plot_df: creates a specific dataframe to handle OOS forecasts for future plotting
15. plot_oos: make a plot of the OOS series, forecasts, confidence intervals
16. create_final_oos_df: creates a final date frame of the OOS data and confidence intervals and places a date forecast stamp so this can be used later or loaded into a data warehouse


