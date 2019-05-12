# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 22:48:37 2018

@author: Namish Kaushik
"""
import pandas as pd
import numpy as np
a= [77,2,3,46,4,2,11]
b=pd.Series(a)
b.index= np.arange(len(b))
c=np.array([12,34,56,1,90,75])
c1=pd.Series(c)
c1.index= np.arange(len(c))
c1.name = 'marks'
d1= {'In':'Del','UP':'Lko',"UK" :"Dehra"}
d=pd.Series(d1)
d.reset_index()
d.to_frame().reset_index()
#cobining series to form data frame
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))
ser3= pd.concat([ser1,ser2],axis=1)
ser3.columns = ['alpha','inde']

# get item in sera not present in serb

ser1 = pd.Series([1, 2,2,3,3,3, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
ser1[~ser1.isin(ser2)]
np.union1d(ser1, ser2)
ser_u = pd.Series(np.union1d(ser1, ser2))
ser_i = pd.Series(np.intersect1d(ser1, ser2))
ser_u[~ser_u.isin(ser_i)]

ser_3=pd.Series(np.random.normal(10,5,25)) #mean,sd,size
np.percentile(ser_3,q=[0,25,50,75,100])
ser1[~ser1.isin(ser1.value_counts().index[:2])] = 'Other'
#How to convert a numpy array to a dataframe of given shape? (L1)

ser = pd.Series(np.random.randint(1, 10, 35))
df= pd.DataFrame(ser.values.reshape(5,7))
len(ser)
for i in range(len(ser)):
    if ser[i]%3 ==0:
        print(i)
#or
np.argwhere(ser%3==0)  
# How to extract items at given positions from a series      
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
ser1= pd.Series(np.arange(26))
pos = [0,4,18,14,20]
ser[pos]
# How to stack two series vertically and horizontally ?
pd.concat([ser,ser1],axis=1)
ser1.append(ser)
#or
pd.concat([ser,ser1],axis=0)
ser.name="alpha"
ser1.name = "numeric"
#How to get the positions of items of series A in another series B?
ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])
#ser1[ser2.isin(ser1)].index.values
# Solution 1
[np.where(i == ser1)[0].tolist()[0] for i in ser2]

# Solution 2
[pd.Index(ser1).get_loc(i) for i in ser2]

#How to convert the first character of each element in a series to uppercase?
ser1= pd.Series(['how' ,'to', 'learn','datascience' ])
pd.Series([i.title() for i in ser1])
#or
ser1.map(lambda x: x.title())

# Solution 2
ser1.map(lambda x: x[0].upper() + x[1:])
# How to calculate the number of characters in each word in a series?
ser1.map(lambda x :len(x))
#How to compute difference of differences between consequtive numbers of a series?
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35],[11,45,67,61, 15, 21, 27, 35])
#ser1.map(lambda x : x[i+1]- x[i] for i in x)
print(ser.diff().tolist())
#How to convert a series of date-strings to a timeseries?

ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
pd.to_datetime(ser)
#How to get the day of month, week number, day of year and day of week from a series of date strings?
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
sert_t = pd.to_datetime(ser)
print("day of month",sert_t.dt.day.tolist())
print("day of week",sert_t.dt.weekday_name.tolist())
print("week of year",sert_t.dt.weekofyear.tolist())
print("day of year",sert_t.dt.dayofyear.tolist())
#How to filter words that contain atleast 2 vowels from a series?
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
#How to get the mean of a series grouped by another series?
fruit = pd.Series(np.random.choice(['apple','banana','orange'],10))
weight = pd.Series(np.linspace(2,20,10))
weight.groupby(fruit).mean()

# How to find all the local maxima (or peaks) in a numeric series?
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
np.sign(np.diff(ser))
dd=np.diff(np.sign(np.diff(ser)))
peak_locs = np.where(dd == -2)[0] +1
peak_locs
#How to replace missing spaces in a string with the least frequent character?
my_str = 'dbc deb abed gade'

c=pd.Series(list('dbc deb abed gade'))
d=c.value_counts()
c.value_counts().index[-1]
l_f= d.dropna().index[-1]
c.replace(' ',l_f)
ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))
autocorrelations = [ser.autocorr(i).round(2) for i in range(11)]
print(autocorrelations[1:])
print('Lag having highest correlation: ', np.argmax(np.abs(autocorrelations[1:]))+1)

#How to import only every nth row from a csv file to create a dataframe?
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', chunksize=50)
df2 = pd.DataFrame()
for chunk in df:
    df2 = df2.append(chunk.iloc[0,:])
# differnet pract    
df= pd.read_excel("movie.xlsx",chunk_size= 50)
df.columns
df.iloc[:,1].value_counts().index
np.unique(np.array(df.iloc[:,1]))
c= df.iloc[:,1].unique()
#How to change column values when importing csv to a dataframe?
df= pd.read_csv("BostonHousing.csv",converters= {'medv': lambda x : 'High' if float(x)> 25  else 'low'})

#How to get the nrows, ncolumns, datatype, summary stats of each column of a dataframe? Also get the array and list equivalent.
df1= pd.read_csv("BostonHousing.csv")
df1.shape
df1.describe()
c1=df1.values
d1= df1.values.tolist()
df.index(['medv'])
df.get_dtype_counts()
df.dtypes.value_counts()
#How to extract the row and column number of a particular cell with given criterion?
#Which manufacturer, model and type has the highest Price? What is the row and column number of the cell with the highest Price value?

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
max1= df['Price'].max()
max2= np.max(df.Price)
des= df.loc[df['Price'] == max1,['Manufacturer', 'Model', 'Type']]

e1 = df['Price']
e2= df.Price
row,column = np.where(df.values == np.max(df.Price))
row,column = np.where(df.values == df['Price'].max())
#How to rename a specific columns in a dataframe?
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv'
                 )
df.columns = df.columns.map(lambda x: x.replace('Type', 'CarType'))
print(df.columns)

#How to check if a dataframe has any missing values?
df.isnull().values.any()
#calculating column with  maximum number of missing  value 
n_miss = df.apply(lambda x : x.isnull().sum())
np.argmax(n_miss)
#How to replace missing values of multiple numeric columns with the mean?
df[["Min_Price","Max_Price"]] = df[["Min_Price","Max_Price"]].apply(lambda x : x.fillna(x.mean()) )
# to retwurn a clomn as df instead of series
df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
type(df[['a']])
#to interchange the column
df[list('cbade')]
# How to set the number of rows and columns displayed in the output?
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)

#How to filter every nth row in a dataframe?
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')

# Solution
print(df.iloc[::20, :][['Manufacturer', 'Model', 'Type']])
#How to create a primary key index by combining relevant columns

df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv', usecols=[0,1,2,3,5])
df[['Manufacturer', 'Model', 'Type']] = df[['Manufacturer', 'Model', 'Type']].fillna('missing')
df.index = df.Manufacturer + '_' + df.Model + '_' + df.Type
print(df.index.is_unique)
# How to get the row number of the nth largest value in a column?
df = pd.DataFrame(np.random.randint(1, 30, 30).reshape(10,-1), columns=list('abc'))
df['a'].argsort()[::-1][5]
#OR
df['a'].sort_values().index[-5]


#How to find the position of the nth largest value greater than a given value?
#In ser, find the position of the 2nd largest value greater than the mean.
ser = pd.Series(np.random.randint(1, 100, 15))
ser.sort_values()
ser[ser > ser.mean()].sort_values().values[1]
#or postion
ser[ser > ser.mean()].sort_values().index[1]
#How to get the last n rows of a dataframe with row sum > 100?

df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

rowsums = df.apply(np.sum, axis=1)

df.iloc[np.where(rowsums > 100)[0][-2:], :]
type(rowsums)

#How to reshape a dataframe to the largest possible square after removing the negative values?
df = pd.DataFrame(np.random.randint(-20, 50, 100).reshape(10,-1))
arr = df[df>0].values.flatten()
arr_qualified = arr[~np.isnan(arr)]

# Step 2: find side-length of largest possible square
n = int(np.floor(arr_qualified.shape[0]**.5))

# Step 3: Take top n^2 items without changing positions
top_indexes = np.argsort(arr_qualified)[::-1]
output = np.take(arr_qualified, sorted(top_indexes[:n**2])).reshape(n, -1)

#How to swap two rows of a dataframe?

df= pd.DataFrame(np.arange(25).reshape(5,-1))

def swap(df,i1,i2):
    a,b = df.iloc[i1,:].copy,df.iloc[i2,:].copy
    df.iloc[i1,:],df.iloc[i2,:]= b,a
    return df
print(swap(df,1,2))

#reverse all rows of a df
df = pd.DataFrame(np.arange(25).reshape(5, -1))
df2 =df.iloc[::-1,:]
    

# Which column contains the highest number of row-wise maximum values?

df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1))
df.apply(np.argmax, axis=1).value_counts().index[0]

#How to create a column containg maximum by minimum of each row

df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1))
df['ratio'] =df.apply( lambda x : np.max(x)/np.min(x),axis =1)
#How to create a column that contains the penultimate value in each row?
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
df['pen'] =df.apply(lambda x: x.sort_values().unique()[-2], axis=1)

#How to normalize all columns in a dataframe?
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
df.apply(lambda x : x.mean(), axis =1)
df.apply(lambda x: x.std(),axis =1)

normalize = df.apply(lambda x: ((x - x.mean())/x.std()).round(2))

#How to compute the correlation of each row with the suceeding row?

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
df.shape
[df.iloc[i+1].corr(df.iloc[i]) for i in range(df.shape[0])[:-1]]


#How to get the particular group of a groupby dataframe by key?
df = pd.DataFrame({'col1': ['apple', 'banana', 'orange'] * 3,
                   'col2': np.random.rand(9),
                   'col3': np.random.randint(0, 15, 9)})

df_grouped = df.groupby(['col1'])

df_grouped.get_group('apple')
#How to get the n’th largest value of a column when grouped by another column?
# in df, find the second largest value of 'taste' for 'banana'
df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                   'taste': np.random.rand(9),
                   'price': np.random.randint(0, 15, 9)})

print(df)
df_grpd = df['taste'].groupby(df.fruit)
df_grpd.get_group('banana').sort_values().iloc[-2]
#How to compute grouped mean on pandas dataframe and keep the grouped column as another column (not index)?
df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                   'rating': np.random.rand(9),
                   'price': np.random.randint(0, 15, 9)})
out = df.groupby('fruit', as_index=False)['price'].mean()
print(out)

#How to remove rows from a dataframe that are present in another dataframe?
#From df1, remove the rows that are present in df2. All three columns must be the same.
df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                    'weight': ['high', 'medium', 'low'] * 3,
                    'price': np.random.randint(0, 15, 9)})

df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,
                    'kilo': ['high', 'low'] * 3,
                    'price': np.random.randint(0, 15, 6)})
df1[~df1.isin(df2)].dropna()
# How to get the positions where values of two columns match?

df = pd.DataFrame({'fruit1': np.random.choice(['apple', 'orange', 'banana'], 10),
                    'fruit2': np.random.choice(['apple', 'orange', 'banana'], 10)})
    
np.where(df.fruit1 == df.fruit2)    

#How to create lags and leads of a column in a dataframe?

df = pd.DataFrame(np.random.randint(1, 100, 20).reshape(-1, 4), columns = list('abcd'))

df['a_lag1'] = df['a'].shift(1)
df['b_lead1'] = df['b'].shift(-1)
print(df)
# How to get the frequency of unique values in the entire dataframe?
#Get the frequency of unique values in the entire dataframe df.
df = pd.DataFrame(np.random.randint(1,40,20).reshape(-1,4))
pd.value_counts(df.values.ravel())
df.head()
pd.value_counts(df.values.ravel())
string = ' xoxo love xoxo   '

# Leading whitepsace are removed
print(string.strip())

print(string.strip(' xoxoe'))

print(string.strip('sti'))