import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error , r2_score
import matplotlib.pyplot as plt
from streamlit_run import *
import plotly.express as px
import numpy as np
import streamlit as st

def remove_null_zero_data(df, target_feature):
    df = df.dropna(subset=[target_feature])
    df = df[df[target_feature] != 0]
    return df

def create_neural_network_regressor(df, target_feature, feature_columns):
    df_cleaned = remove_null_zero_data(df, target_feature)

    # Convert target feature to its appropriate data type if needed
    df_cleaned[target_feature] = df_cleaned[target_feature].astype(float)

    # Convert feature columns to their appropriate data types if needed
    df_cleaned[feature_columns] = df_cleaned[feature_columns].astype(float)

    # Extract features (X) and target variable (y)
    X = df_cleaned[feature_columns]
    y = df_cleaned[target_feature]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Create a neural network regressor
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model using mean squared error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    r = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("Rsquared error:",r)

    for feature in feature_columns:
        plot_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            feature: X_test[feature]
        })
        print(plot_df)
        # Plotting using Plotly
        fig = px.histogram(plot_df, x='Actual', y='Predicted', color=feature,
                     labels={'Actual': 'Actual Values', 'Predicted': 'Predicted Values'},
                     title=f'Actual vs Predicted Values (Colored by {feature})')
        st.plotly_chart(fig,use_container_width=True,height=200)

    return fig

