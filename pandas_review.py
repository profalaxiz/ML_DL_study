import pandas as pd

# pandas is a convenient tool for dealing with tabular datasets

# pandas series is like a column in a table = 1d array 
# a = [12393, True, "Guthrie"]
# myvar = pd.Series(a)
# ages = pd.Series([20, 30, 25])
# my_var_2 = pd.Series(a, index = ["ID", "Is Student: ", "Name: "])
# print(myvar[2])
# print(my_var_2)  # or more specific my_var_2["ID"]
# print(ages)

# dataframes is a table = 2D array
# data = {
#     "age": [20, 30, 25],
#     "salary": [3000, 5000, 4000],
#     "city": ["Beijing", "Shanghai", "Beijing"]
# }
# df = pd.DataFrame(data)     # can also use index=[...] to name each row
# # print(df)
# print(f"data of the first row: \n{df.loc[0]}")  # returns a pandas series
# print(df.loc[[0, 1]])  # using list of indexes, returns a pandas dataframe
# print(f"salary of first person: {df.loc[0, "salary"]}")    # loc uses labels
# print(f"salary of first person: {df.iloc[0, 1]}")    # iloc uses numerical positions

# csv (comma separated values) -> use it when data naturally fits into rows and cols
# df = pd.read_csv('data.csv')
# print(df.to_string())
# print(df) # only prints first and last 5 rows
# print(pd.options.display.max_rows) 

# json (javascript object notation) -> use it when data has nested or varying structure (APIs, web applications, configuration files,etc)
# df = pd.read_json('data.json')  
# # print(df.to_string())
# # print(df)
# # print(df.head(10))  
# # print(df.tail())
# print(df.info()) # shows technical structural overview (memory usage, data types, row/col counts)
# print(df.describe())  # shows statistical summary of data distribution, returns a dataframe

# data cleaning
# df = pd.read_csv('data.csv')
# # print(df.info())     
# new_df = df.dropna()
# print(new_df.info())
# # df.dropna(inplace=True) -> change the original table inplace
# print(df.head())   
# df.fillna({"Calories": 150}, inplace=True)
# df.fillna({"Maxpulse": 140}, inplace=True)
# df.fillna({"Duration": 60}, inplace=True) 
# print(df.info())
# x = df["Calories"].mean()   # avg value (sum of all values / number of values)
# y = df["Calories"].median() # value in the middle, after all values are sorted ascending
# z = df["Calories"].mode()   # value that appears most frequently
# print(z)
# print(df.describe())
# print(df.dtypes)
# print(df.columns)
# print(df.isna().sum())  # isna() -> boolean mask where missing values map to True, and valid data maps to 0, sum() adds those ones along the column axis by default
# df.isna().sum(axis=1)    # count missing values per row
# print(f"total missing values in the whole dataset: {df.isna().sum().sum()}")
# print(new_df.head())
# print(new_df.tail()) 
# high_pulse = df[
#     (df["Pulse"] > 120) &
#     (df["Maxpulse"] > 150)]
# print(high_pulse.sort_values(by="Pulse"))  # to sort by col values
# print(high_pulse.sort_index())  # to sort by row index, axis=1 to sort cols alphabetically by their names
# new_df["High_Pulse"] = new_df["Pulse"] > 120  #creating new col
# print(new_df.head(5))

# data = {
#     'Department': ['Sales', 'Tech', 'Sales', 'Tech', 'HR'],
#     'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
#     'Salary': [50000, 85000, 62000, 90000, 55000]
# }
# df = pd.DataFrame(data)
# avg_salary = df.groupby('Department')['Salary'].mean()
# print(avg_salary)
# salary_info = df.groupby('Department').agg({'Salary': ['min', 'max', 'mean']})
# print(salary_info)
# pandas -> numpy 
# X = df[['Employee', 'Department','Salary']].to_numpy()
# print(X)

#practice
data = {
    "name": ["A", "B", "C", "D", "E"],
    "age": [20, 25, None, 35, 40],
    "salary": [3000, 4500, 5000, None, 8000],
    "department": ["IT", "HR", "IT", "HR", "IT"]
}
# #1
df = pd.DataFrame(data)
# #2
# print(f"types:\n{df.dtypes}")
# print(f"shape: {df.shape}")
# #3
# print(df.isna().sum())
# #4
# df.fillna({"age": df["age"].mean()}, inplace=True)
# #5
# df.fillna({"salary": df["salary"].median()}, inplace=True)
# #6
# emp = df[df["salary"] > 4000]
# #7
# avg_salary = df.groupby('department')['salary'].mean()
# #8
# df["salary_per_age"] = df["salary"]/df["age"]
# #9
# print(df.sort_values(by="salary", ascending=False))
# #10
# X = df[["age", "salary"]].to_numpy()

#11 
it_salary = df[(df["department"] == "IT") & (df["salary"] > 4000)]
print(it_salary)
#12
dept_avg = df.groupby("department")[["age", "salary"]].mean()
print(dept_avg)
#13
df["above_avg_salary"] = df["salary"] > df["salary"].mean()
print(df.head(5))
#14
# Age_Salary = df[(df["department"] == "IT")][["age", "salary"]].to_numpy()
Age_Salary = df.loc[df["department"] == "IT", ["age", "salary"]].to_numpy()
print(Age_Salary)