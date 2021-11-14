import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import pickle


df = pd.read_csv('50_Startups.csv')

df1 = pd.get_dummies(df,drop_first=True)

X = df1.iloc[:,[0,1,2,4,5]].values
y = df1.iloc[:,3].values


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

pickle.dump(model, open('model.pkl','wb'))


m2 = pickle.load(open('model.pkl','rb'))
print(m2.predict([[27892.92,84710.77,164470.71,1,0]]))




