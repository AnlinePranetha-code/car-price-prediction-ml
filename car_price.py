import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv("cardata.csv")
X=data.drop("price",axis=1)
Y=data["price"]
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.4,random_state=42)
model=LinearRegression()
model.fit(Xtrain,Ytrain)
weights_df = pd.DataFrame({
    "Feature": X.columns,
    "Weight (Coefficient)": model.coef_
})
print("\nFeature Weights:")
print(weights_df)
print("\nModel Bias (Intercept):", model.intercept_)
bias_df = pd.DataFrame({
    "Feature": ["Bias (Intercept)"],
    "Weight (Coefficient)": [model.intercept_]
})
full_weights_df = pd.concat([weights_df, bias_df], ignore_index=True)
print("\nModel Parameters:")
print(full_weights_df)
Ypred=model.predict(Xtest)
print("\nR2 Score:",r2_score(Ytest,Ypred))
print("Mean Squared Error:",mean_squared_error(Ytest,Ypred))
plt.figure()
plt.scatter(Ytest, Ypred, label="Predicted Values")
m, b = np.polyfit(Ytest, Ypred, 1)
plt.plot(Ytest, m*Ytest + b, label="Regression Line",color="red")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Linear Regression)")
plt.legend()
plt.show(block=False)
residuals = Ytest - Ypred
plt.figure()
plt.scatter(Ypred, residuals,color="green")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0)
plt.show(block=False)
plt.figure()
plt.bar(weights_df["Feature"], weights_df["Weight (Coefficient)"],color="orange")
plt.xticks(rotation=45)
plt.xlabel("Features")
plt.ylabel("Weight Value")
plt.title("Feature Importance (Linear Regression Weights)")
plt.show(block=False)
print("\n Car Price Prediction")
ensize=int(input("Enter Engine Size(in cc):"))
horsepow=int(input("Enter Horse Power:"))
carwid=int(input("Enter Car Width(in inches):"))
carlen=int(input("Enter Car length(in inches):"))
fueltype=int(input("Fuel Type(Petrol=0,Diesel=1):"))
aspi=int(input("Aspiration(Std=0,Turbo=1):"))
userinput=pd.DataFrame([[ensize,horsepow,carwid,carlen,fueltype,aspi]],columns=X.columns)
finalprice=model.predict(userinput)
print("Predicted Car Price:Rs ",round(finalprice[0],2))