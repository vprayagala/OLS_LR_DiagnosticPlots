# OLS_LR_DiagnosticPlots
Class for generating diagnostic plots for OLS Linear Regression in Python similar to R plots

OLS :
OLS is commonly used regression method and simple method to understand relationship between dependent and independent attributes. This method can be treated as first step for studying correlations, p-values, t-statistic , coefficients and significance of attributes. Though this is simple method which makes certain assumptions, yet its most used method to understand the affect of independent attributes on dependent. Lets understand the assumptions that are made for linear regression method,

1. Association is Linear 
2. Error Terms are independent
3. Error Terms have Constant Variance (Homoscedasticity) 
4. Error Terms are Normally Distributed 

In R provides a plot function for linear regression models that gives all the 4 plots. The equivalent plots in python is easy , but tricky. It is easy to get these plots if we build OLS model using stats models than sklearn linear model. 
Class was built to generate these plots. The object can be created by passing x and y to this class. Fit method of the class can be used for OLS model. The diagnostic method can be used after that to generate the plots and json summary response.

The diagnostic plots can be used to validate the if the assumptions are valid. Approaches like transformation of features, fitting polynomial regression etc to be used if assumption fails or use more flexible algorithm to fit the data.


Usage:

1) Create an object using x, y - linear_plot = Plot.LinearRegressionResidualPlot(x_train.values, y_train.values)
2) Use the class fit method for OLS 
3) Pass this model to diagnostic_plots method to generate the plots and summary 
    ex, lm = linear_plot.fit()
    summary, diag_res = linear_plot.diagnostic_plots(lm)
    
    
