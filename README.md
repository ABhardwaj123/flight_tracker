# ✈️ Flight Price Predictor

A machine learning project that predicts Indian domestic flight ticket prices using linear regression implemented (from scratch) with NumPy — no sklearn for the model itself.

# One-hot encoding
Nominal categories (no order) - 'Airline' , 'Source' , 'Destination' were one hot encoded using pd.get_dummies(drop_first = True).
The drop_first = True argument removes one redundant columns per 
category to avoid the dummy varible trap

### Engineered binary features
Domain knowledge was used to add features a model cannot derive on its own:
- weekend_or_not
- festive_season
- time_bucket (departure hour classified as morning/evening/night)

### Interaction terms
Multiplications between related features help a linear model approximate non-linear patterns:
- 'weekend x duration'
- 'stops x duration'
- 'festive x weekend'
- 'airline x duration'


## Model: Linear Regression from scratch

### Equation -> Y = w * x + b

### Normalisation
All features were standardised using Z score normalisation(fit on training data)

X_scaled = (X - mean_train) / std_train
It is essential so that the gradient descent can converge

### Gradient Descent
weights are initially zero and updated iteratively to minimize mean sqaured error

for i in range(iterations):
    y_pred = X_train @ w + b
    error  = y_pred - Y_train

    dw = (2/n) * (X_train.T @ error)   # gradient w.r.t. weights
    db = (2/n) * np.sum(error)          # gradient w.r.t. bias

    w -= alpha * dw
    b -= alpha * db

## Results

The model was evaluated on held-out test data using RMSE (root mean squared error), which is interpretable in the original price units (INR):

Training RMSE: 2773.58
Testing RMSE: 2805.16

A small gap between training and testing RMSE indicates the model generalises without significant overfitting.