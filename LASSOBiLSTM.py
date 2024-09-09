import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="IDR/USD Price Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Page title and description
st.title('IDR/USD Price Prediction')
st.markdown("""
This application predicts the IDR/USD exchange rate using Lasso Regression and BiLSTM models.
Please upload a CSV file with the historical exchange rates.
""")

# Load data function
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path, delimiter=';')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.sort_values(by='Tanggal', ascending=False, inplace=True)
    df.dropna(inplace=True)
    return df

# Sidebar for file upload
st.sidebar.header("Upload CSV file")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    y = df['IDR']
    X = df.drop(columns=['Tanggal', 'IDR'], axis=1)

    # Display normalized dataset
    st.subheader("Normalized Dataset")
    scaler = MinMaxScaler(feature_range=(0, 1))
    all_features = df.drop(columns=['Tanggal', 'IDR']).values
    normalized_features = scaler.fit_transform(all_features)

    # Combine normalized features and labels into a DataFrame
    normalized_df = pd.DataFrame(normalized_features, columns=df.columns[2:])
    st.write(normalized_df)
    
    # Min-Max Normalization
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44)
    x_train = pd.DataFrame(scaler.fit_transform(x_train), columns=X.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=X.columns)
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    y_train = pd.Series(y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel())
    y_test = pd.Series(y_scaler.transform(y_test.values.reshape(-1, 1)).ravel())

    # Lasso Regression
    st.subheader("Lasso Regression")
    pipeline = Pipeline([('model', Lasso())])
    #param_grid = {'model__alpha': np.arange(0.000001, 0.01, 0.0001)}
    param_grid = {'model__alpha': np.arange(0.001, 10, 0.01)}
    lasso = GridSearchCV(pipeline, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', verbose=3)

    with st.spinner('Training Lasso Regression...'):
        lasso.fit(x_train, y_train)

    lr_y_pred = lasso.predict(x_test)
    train_score = lasso.score(x_train, y_train)
    test_score = r2_score(y_test, lr_y_pred)

    st.write(f"r2 score: {test_score:.4f}")

    best_lasso_model = lasso.best_estimator_
    coef = best_lasso_model.named_steps['model'].coef_
    lasso_coefficient = pd.DataFrame()
    lasso_coefficient["Columns"] = x_train.columns
    lasso_coefficient['Coefficient Estimate'] = pd.Series(coef)

    st.write("Optimal alpha coefficients:")
    st.dataframe(lasso_coefficient)

    # Plot Lasso Coefficient Estimates using Plotly
    fig = px.bar(lasso_coefficient, x="Columns", y='Coefficient Estimate', title='Lasso Coefficient Estimates')
    st.plotly_chart(fig)

    # Identify and remove variables with zero coefficients
    zero_coef_columns = lasso_coefficient[lasso_coefficient['Coefficient Estimate'] == 0]['Columns'].values
    X = X.drop(columns=zero_coef_columns)
    #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44)
    

     # Hyperparameter tuning options
    #st.sidebar.subheader("Hyperparameter Tuning")
    #lstm_units_1 = st.sidebar.slider("LSTM Units Layer 1", 32, 256, 128, step=32)
    #dropout_rate_1 = st.sidebar.slider("Dropout Rate Layer 1", 0.0, 0.5, 0.2, step=0.1)
    #lstm_units_2 = st.sidebar.slider("LSTM Units Layer 2", 16, 128, 32, step=16)
    #dropout_rate_2 = st.sidebar.slider("Dropout Rate Layer 2", 0.0, 0.5, 0.4, step=0.1)
    #lstm_units_3 = st.sidebar.slider("LSTM Units Layer 3", 32, 256, 64, step=32)
    #dropout_rate_3 = st.sidebar.slider("Dropout Rate Layer 3", 0.0, 0.5, 0.0, step=0.1)
    #learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, step=0.0001)
    #epochs = st.sidebar.slider("Epochs", 10, 500, 50, step=10)
    #patience = st.sidebar.slider("Early Stopping Patience", 10, 100, 20, step=10)

    # BiLSTM Model
    st.subheader("BiLSTM Model")
    all_features = df.drop(columns=['Tanggal', 'IDR']).drop(columns=zero_coef_columns).values
    normalized_features = scaler.fit_transform(all_features)
    sequence_length = 1
    X, y = [], []
    for i in range(sequence_length, len(normalized_features)):
        X.append(normalized_features[i-sequence_length:i, :])
        y.append(normalized_features[i, 0])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # With Hyperparameter Tuning
    #tf.random.set_seed(1234)
    #model = Sequential()
    #model.add(Bidirectional(LSTM(units=lstm_units_1, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    #model.add(Dropout(dropout_rate_1))
    #model.add(Bidirectional(LSTM(units=lstm_units_2, return_sequences=True)))
    #model.add(Dropout(dropout_rate_2))
    #model.add(Bidirectional(LSTM(units=lstm_units_3)))
    #model.add(Dropout(dropout_rate_3))
    #model.add(Dense(units=1, activation='sigmoid'))
    #model.compile(optimizer=Adam(learning_rate=learning_rate), loss=MeanSquaredError(), metrics=[RootMeanSquaredError()])
    #monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=patience, verbose=1, mode='auto', restore_best_weights=True)
    
    #normal stacked BiLSTM
   #model = Sequential()
    #model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    #model.add(Dense(units=1, activation='sigmoid'))
    #model.compile(optimizer='adam', loss='mean_squared_error')
    #monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=500, verbose=1, mode='auto', restore_best_weights=True)
    #model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.005701), metrics=[RootMeanSquaredError()])

    # Without Hyperparameter Tuning
    tf.random.set_seed(1234)
    model = Sequential()
    model.add(Bidirectional(LSTM(units=128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dropout(0.0))
    model.add(Bidirectional(LSTM(units=64)))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=500, verbose=1, mode='auto', restore_best_weights=True)
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.005701), metrics=[RootMeanSquaredError()])

    with st.spinner('Training BiLSTM Model...'):
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[monitor], verbose=1, epochs=50)

    # Plot Loss
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss'))
    fig.update_layout(title='Model Loss', xaxis_title='Epoch', yaxis_title='Loss')
    st.plotly_chart(fig)

    y_val_pred = model.predict(X_train)
    val_dates = df['Tanggal'][-len(y_train):].values
    val_dates = pd.to_datetime(val_dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=val_dates, y=y_train, mode='lines', name='Actual IDR/USD Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=val_dates, y=y_val_pred.ravel(), mode='lines', name='Predicted IDR/USD Price', line=dict(color='red')))
    fig.update_layout(title='IDR/USD Price Prediction', xaxis_title='Year', yaxis_title='IDR/USD Price')
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig)

    st.success("Model training and evaluation complete!")

    # Evaluate BiLSTM Model
    scores = model.evaluate(X_test, y_test, verbose=3)
    loss = scores[0]
    rmse = scores[1]
    mae = mean_absolute_error(y_test, model.predict(X_test))
    mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
    r2 = r2_score(y_test, model.predict(X_test))

    st.subheader("Evaluation Metrics")
    st.write(f"**Loss:** {loss:.3f}")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAPE:** {mape:.2%}")
    

    st.sidebar.markdown("### Evaluation Metrics")
    st.sidebar.write(f"**Loss:** {loss:.3f}")
    st.sidebar.write(f"**MAE:** {mae:.4f}")
    st.sidebar.write(f"**RMSE:** {rmse:.4f}")
    st.sidebar.write(f"**MAPE:** {mape:.2%}")
   
    # Inverse transform predictions and actual values
    y_train_inverse = y_scaler.inverse_transform(y_train.reshape(-1, 1)).ravel()
    y_val_pred_inverse = y_scaler.inverse_transform(y_val_pred).ravel()
    
     # Display the prediction results in a table
    st.subheader("Prediction Results")
    result_data = {
        'Date': val_dates,
        'Actual IDR/USD Price': y_train_inverse,
        'Predicted IDR/USD Price': y_val_pred_inverse
    }
    result_df = pd.DataFrame(result_data)
    #result_df = result_df.sort_values(by='Date', ascending=True)
    st.dataframe(result_df)

    st.success("Model training and evaluation complete!")