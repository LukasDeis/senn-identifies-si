import numpy as np
import pandas as pd
import tensorflow as tf


class DataSorter:
    def __init__(self):
        self.headers_loc = 'C:/Users/Lukas.Deis/Documents/dataset/headers.csv'
        
    def sort(self, dataframe, target_heading):
        # To preprocess the input into usable data, we need to know which column contains what kind of data.
        # Data can either be:
        #     - a scalar value (a fee one has to pay)
        #     - a numeric value that should be interpreted as categorical (age in groups)
        #     - a string that should be interpreted as categorical ( very,a bit, not really, no)
        # Usually not everything is encoded that nicely, 
        # in this dataset there are some dates that can not easily be converted.
        #     - columns with type date need to be converted to a value of age (in years) 
        #       and then sorted into categories, before further processing as categorical, numeric values
    
        # read different types for columns to treat them accordingly
        # TODO read the headers from headers.csv here and remove the old hardcoded keys
        numerical_features = []
        categorical_int_features = []
        categorical_cols = []
        date_cols = []
        year_cols = []
        to_be_removed = []
       
        headers = pd.read_csv(self.headers_loc)  # TODO we could use a JSON file for this
    
        headers.pop("notes")
        type_sorting = {  # todo, this should be stored in a JSON file instead of code
            "date": date_cols,
            "year": year_cols,
            "skalar": numerical_features,
            "categorical_int": categorical_int_features,
            "categorical_string": categorical_cols,
            "remove": to_be_removed
            # cols like id's and the date of the test that don't actually carry information for the prediction
        }
    
        for row in headers.itertuples(): 
            heading, col_type = row.heading, row.type
            # the target variable, should not be sorted as it is removed from the dataset earlier.
            if heading == target_heading:
                print(row)
                pass
            right_column_list = type_sorting.get(col_type, lambda: tf.print("invalid type:", col_type, " for column:",
                                                                            heading))  # find the type of this key 
            # TODO fix that instead of a lambda, the error message is give, 
            #  otherwise this leads to: "AttributeError: 'function' object has no attribute 'append'" later on
            try:
                right_column_list.append(heading)  # append the key to the right col
            except:
                tf.print("ERROR: Something went wrong while reading the headers.csv file")
                tf.print("       Could it be that '", col_type,
                         "' is not actually a valid type for a heading? It is used in row: ", heading)
    
        # age and date columns are easiest to work with if they are in time-format.
        # Thus the columns are converted to time since [date / year] 
        # and added to the categorical_int_features as ages in years
        for date_col in date_cols:
            # convert to years
            dataframe[date_col] = dataframe[date_col].replace(' ', np.nan,
                                                              regex=True)  # replace empty strings with parsable NaN
            dataframe[date_col] = pd.to_datetime(dataframe[date_col], format='%m/%d/%Y')  # convert to date format
            dataframe[date_col] = pd.DatetimeIndex(dataframe[date_col]).year  # take only year of date
            # add to year cols
            year_cols.append(date_col)
    
        for year_col in year_cols:
            current_year = 2020
            dataframe[year_col] = dataframe[year_col].replace(' ', np.nan,
                                                              regex=True)  # replace empty strings with parsable NaN
            dataframe[year_col] = current_year - dataframe[year_col].astype(float)
            # remove NaN values as they would break further computations
            # this bears the risk of seeing factors in the wrong way, 0 can mean: always, never, don't know...
            dataframe[year_col] = dataframe[year_col].replace(np.nan, 0, regex=True)
            # add to categorical_int features because that is what ages are
            categorical_int_features.append(year_col)
            print(dataframe[year_col])
    
        # categorical_int_features are all ages
        # while age in general is important, small differences in age max cause more confusion than clarity
        # 20 VS 40 should be considered by the network, 20 VS 21 not so much
        # age should be considered in categories of age-groups, but the network can figure all that out for itself
        # for simplicity, age will considered as a normal numeric value, any complex relationship to the outcome should be learned
    
        # It happens that all categorical_ints in this set are ages and none are left afterwards
        # If that was different, one would need to do this differently
        # TODO make a different tag "age" that can be used or just mark ages as skalars
        numerical_features.extend(categorical_int_features)
        categorical_int_features = []

        # make numerical columns readable as floats
        for feature in numerical_features:
            dataframe[feature] = dataframe[feature].replace(' ', 0,
                                                            regex=True)  # used to be np.NaN, but that does not work too well
            dataframe[feature] = dataframe[feature].astype(float)
    
        # remove cells that should not be considered (generalizable (long texts) or do not carry value (ID's))
        dataframe = dataframe.drop(columns=to_be_removed)
        return dataframe
