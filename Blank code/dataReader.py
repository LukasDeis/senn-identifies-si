data_file = 'C:/Users/Lukas.Deis/Documents/dataset/MIND_Set_Data_exported.csv'
# convert to csv
dataframe = pd.read_csv(data_file)
#dataframe = dataframe.replace(np.nan,"UNKNOWN")

print('-----------')
#for typ in dataframe.dtypes:
    #print(typ.?)
    #get column name of each non object column and convert to string?
print('-----------')

#Alternative way to read things, less intuitive, other customization-options
#dataframe, meta = pyreadstat.read_sav(data_file,user_missing=True, apply_value_formats=False)
#print(meta.missing_ranges["AQ50q1"])

#If I use the second method with apply_value_formats=False all numerical ones should be fine and I have to only interpret the two dates and strings manually

dataframe.head()