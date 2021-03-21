import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model


#display plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylabel('Dow Jones')
ax1.set_xlabel('Football ppg')
ax1.set_title('Dow Jones vs Football ppg')


football = [37.924, 37.35, 40.87, 39.9, 39, 40.67, 41.2, 42.95, 43.59]
basketball = [106.788, 108.455, 108.04, 109.892, 109.6, 110.7, 108.926]
dow = [10546, 11409, 13178, 11244, 8885, 10668, 11957, 12966, 15009]

x = np.array(football)
y = np.array(dow)

ax1.plot(x, y, 'o')
plt.show()


#create linear regression model of dow based on football and basketball
df = pd.DataFrame(list(zip(football, basketball, dow)), columns = ['Football', 'Basketball', 'Dow'])


Xf = df[['Football', 'Basketball']]
yf = df['Dow']

regr = linear_model.LinearRegression()
regr.fit(Xf, yf)


#interactive user prediction

foot = input("Enter football ppg: ")
bask = input("Enter basketball ADJOE: ")
print("Expected dow jones: ")
print(regr.predict([[float(foot), float(bask)]])[0])


        

        
