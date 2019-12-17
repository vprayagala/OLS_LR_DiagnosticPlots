# imports
import os
import logging
import pandas as pd
from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger()


#  Functions


def load_data_local(path):
    if os.path.exists(path):
        raw_data = pd.read_csv(path)
        logger.info("Data Loaded from local path:{}".format(path))
        return raw_data
    else:
        print("Unable to Read ..Path:{} Does not Exists".format(path))
        return None


def pre_process_data(data,
                     drop_columns=None,
                     num_columns=None,
                     cat_columns=None
                     ):
    processed = data.copy()

    # Drop columns
    processed.drop(drop_columns, axis=1, inplace=True)

    #  Standardize numeric columns

    for col in num_columns:
        print("Transforming Column:{}".format(col))
        num_scaler = RobustScaler()
        temp = num_scaler.fit_transform(processed.loc[:, col].values.reshape(-1, 1))
        processed = pd.concat([processed, pd.DataFrame(temp)], axis=1)
    #  One hot encode categorical columns
    for cat_col in cat_columns:
        print("Transforming Column:{}".format(cat_col))
        # cat_encoder = OneHotEncoder(handle_unknown='ignore')
        # processed[cat_col] = processed[cat_col].apply(lambda x : 'Not Specified' if x == 'NaN' else x)
        # temp_cat = cat_encoder.fit_transform(processed.loc[:,cat_col].values.reshape(-1,1))

        df_dummies = pd.get_dummies(processed[cat_col], prefix='category')
        processed = pd.concat([processed, df_dummies], axis=1)

    processed.drop(num_columns, axis=1, inplace=True)
    processed.drop(cat_columns, axis=1, inplace=True)

    return processed
