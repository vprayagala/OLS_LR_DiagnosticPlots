#  imports
# import pandas as pd
from sklearn.model_selection import train_test_split
from DataHandler import PrepareData
from ModelDiagnostics import Plot
#
DATA_DIR = "/Users/Saatwik/Documents/INSOFE/CSE 7202C Models Regressions /Day1/Batch 14_CSE 7202c_LinearReg_03012016/CustomerData.csv"


#  Functions
def main():
    raw_df = PrepareData.load_data_local(DATA_DIR)
    print("Data Shape:{}".format(raw_df.shape))
    print("Data Types:{}".format(raw_df.dtypes))

    drop_columns = ['CustomerID']

    num_columns = ['NoOfChildren', 'MinAgeOfChild', 'MaxAgeOfChild', 'Tenure', 'FrquncyOfPurchase',
                   'NoOfUnitsPurchased', 'FrequencyOFPlay', 'NoOfGamesPlayed', 'NoOfGamesBought']
    cat_columns = ['FavoriteChannelOfTransaction', 'FavoriteGame']

    processed_df = PrepareData.pre_process_data(raw_df, drop_columns, num_columns, cat_columns)

    print("Data Pre-Processing Done, Data Shape:{}".format(processed_df.shape))
    x = processed_df.drop(['TotalRevenueGenerated'], axis=1)
    y = processed_df.loc[:, 'TotalRevenueGenerated']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    print("Train Feature Shape:{}".format(x_train.shape))
    print("Train Label Shape:{}".format(y_train.shape))
    print("Test Feature Shape:{}".format(x_test.shape))
    print("Test Label Shape:{}".format(y_test.shape))

    linear_plot = Plot.LinearRegressionResidualPlot(x_train.values, y_train.values)
    lm = linear_plot.fit()
    summary, diag_res = linear_plot.diagnostic_plots(lm)
    print("Summary of Regression\n:{}".format(summary))
    print("Diagnostic Tests of Regression\n:{}".format(diag_res))

#  Main function


if __name__ == '__main__':
    main()
