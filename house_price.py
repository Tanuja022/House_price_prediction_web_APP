from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

data = pd.DataFrame({
    'area': [1000, 1500, 2000, 1200],
    'bedrooms': [2, 3, 4, 2],
    'price': [300000, 450000, 600000, 350000]
})

X=data[['area','bedrooms']]
y=data['price']

model=LinearRegression()
model.fit(X,y)

# save model
joblib.dump(model,"house_price_model.pkl")