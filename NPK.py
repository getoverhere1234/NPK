import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Load dataset
dataset = pd.read_csv('dataset2.csv')

# Determine X (predictors) and y (dependent variables)
X = dataset.iloc[:, 3:].values  # Select all rows from column 4 onwards for predictors
y_N = dataset.iloc[:, 0].values  # Select first column for the dependent variable N
y_P = dataset.iloc[:, 1].values  # Select second column for the dependent variable P
y_K = dataset.iloc[:, 2].values  # Select third column for the dependent variable K

# Split data into training and testing sets for N
X_train_N, X_test_N, y_train_N, y_test_N = train_test_split(X, y_N, test_size=0.2, random_state=0)

# Standardize the data for N
sc_N = StandardScaler()
X_train_N = sc_N.fit_transform(X_train_N)
X_test_N = sc_N.transform(X_test_N)

# Train the model for N (Random Forest Regressor)
regressor_N = RandomForestRegressor(n_estimators=20, random_state=0)
regressor_N.fit(X_train_N, y_train_N)

# Split data into training and testing sets for P
X_train_P, X_test_P, y_train_P, y_test_P = train_test_split(X, y_P, test_size=0.2, random_state=0)

# Standardize the data for P
sc_P = StandardScaler()
X_train_P = sc_P.fit_transform(X_train_P)
X_test_P = sc_P.transform(X_test_P)

# Train the model for P (Random Forest Regressor)
regressor_P = RandomForestRegressor(n_estimators=20, random_state=0)
regressor_P.fit(X_train_P, y_train_P)

# Split data into training and testing sets for K
X_train_K, X_test_K, y_train_K, y_test_K = train_test_split(X, y_K, test_size=0.2, random_state=0)

# Standardize the data for K
sc_K = StandardScaler()
X_train_K = sc_K.fit_transform(X_train_K)
X_test_K = sc_K.transform(X_test_K)

# Train the model for K (Random Forest Regressor)
regressor_K = RandomForestRegressor(n_estimators=20, random_state=0)
regressor_K.fit(X_train_K, y_train_K)

# Create Streamlit app
st.title('Dependent Variable Prediction')

# Add a selection widget for the dependent variable
dependent_variable = st.radio('Select Dependent Variable to Predict:', ['N', 'P', 'K'])

# Create input fields for independent variables
ph = st.number_input('pH', step=0.1, value=7.0)
ec = st.number_input('EC', step=0.1, value=0.0)
oc = st.number_input('OC', step=0.1, value=0.0)
S = st.number_input('S', step=0.1, value=0.0)
B = st.number_input('B', step=0.1, value=0.0)
Zn = st.number_input('Zn', step=0.1, value=0.0)
Fe = st.number_input('Fe', step=0.1, value=0.0)
Mn = st.number_input('Mn', step=0.1, value=0.0)
Cu = st.number_input('Cu', step=0.1, value=0.0)

# Transform inputs into a format suitable for prediction
input_data = np.array([[ph, ec, oc, S, B, Zn, Fe, Mn, Cu]])

# Make prediction based on user choice
# Make prediction based on user choice
if st.button('Predict'):
    if dependent_variable == 'N':
        input_data_scaled_N = sc_N.transform(input_data)  # Scale the input data
        prediction_N = regressor_N.predict(input_data_scaled_N)
        st.write('Predicted value of N:', prediction_N[0])

        # Plot actual vs predicted for N
        st.subheader('Actual vs Predicted for N')
        df_actual_predicted_N = pd.DataFrame({'Actual': y_test_N, 'Predicted': regressor_N.predict(X_test_N)})
        st.line_chart(df_actual_predicted_N)

    elif dependent_variable == 'P':
        input_data_scaled_P = sc_P.transform(input_data)  # Scale the input data
        prediction_P = regressor_P.predict(input_data_scaled_P)
        st.write('Predicted value of P:', prediction_P[0])

        # Plot actual vs predicted for P
        st.subheader('Actual vs Predicted for P')
        df_actual_predicted_P = pd.DataFrame({'Actual': y_test_P, 'Predicted': regressor_P.predict(X_test_P)})
        st.line_chart(df_actual_predicted_P)

    elif dependent_variable == 'K':
        input_data_scaled_K = sc_K.transform(input_data)  # Scale the input data
        prediction_K = regressor_K.predict(input_data_scaled_K)
        st.write('Predicted value of K:', prediction_K[0])

        # Plot actual vs predicted for K
        st.subheader('Actual vs Predicted for K')
        df_actual_predicted_K = pd.DataFrame({'Actual': y_test_K, 'Predicted': regressor_K.predict(X_test_K)})
        st.line_chart(df_actual_predicted_K)

# Display evaluation metrics
st.write('Evaluation Metrics for N:')
st.write('Mean Absolute Error for N:', metrics.mean_absolute_error(y_test_N, regressor_N.predict(X_test_N)))
st.write('Mean Squared Error for N:', metrics.mean_squared_error(y_test_N, regressor_N.predict(X_test_N)))
st.write('Root Mean Squared Error for N:', np.sqrt(metrics.mean_squared_error(y_test_N, regressor_N.predict(X_test_N))))
st.write('Training Accuracy for N:', regressor_N.score(X_train_N, y_train_N))
st.write('Testing Accuracy for N:', regressor_N.score(X_test_N, y_test_N))

st.write('Evaluation Metrics for P:')
st.write('Mean Absolute Error for P:', metrics.mean_absolute_error(y_test_P, regressor_P.predict(X_test_P)))
st.write('Mean Squared Error for P:', metrics.mean_squared_error(y_test_P, regressor_P.predict(X_test_P)))
st.write('Root Mean Squared Error for P:', np.sqrt(metrics.mean_squared_error(y_test_P, regressor_P.predict(X_test_P))))
st.write('Training Accuracy for P:', regressor_P.score(X_train_P, y_train_P))
st.write('Testing Accuracy for P:', regressor_P.score(X_test_P, y_test_P))

st.write('Evaluation Metrics for K:')
st.write('Mean Absolute Error for K:', metrics.mean_absolute_error(y_test_K, regressor_K.predict(X_test_K)))
st.write('Mean Squared Error for K:', metrics.mean_squared_error(y_test_K, regressor_K.predict(X_test_K)))
st.write('Root Mean Squared Error for K:', np.sqrt(metrics.mean_squared_error(y_test_K, regressor_K.predict(X_test_K))))
st.write('Training Accuracy for K:', regressor_K.score(X_train_K, y_train_K))
st.write('Testing Accuracy for K:', regressor_K.score(X_test_K, y_test_K))
