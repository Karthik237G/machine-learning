import pandas as pd #is used for DataFrames and Series
import numpy as np #multi dimensional arrays

#mounting to drive in colab
from google.colab import drive
drive.mount('/content/drive')
data=pd.read_csv("dataset.csv")

#1  data cleaning
#1.1 handle missing values
data=data.fillna(data.mean()) #replace NaN values with column mean

#2 correcting the outliers
z=np.abs(stats.zscore(data))
data=data[(z<3).all(axis=1)] #removes rows with outliers based on Z-score

#3 data transformation 
#3.1 normalize numerical features
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data[["feature1","feature2"]]=scvalr.fit_transform(data[["feature1","feature2"]])

#3.2 encode categorical features
from sklearn.preprocessing import LabelEncoder
lable_encoder=LableEncoder()
data['category']=label_encoder.fit_transform(data['category'])




