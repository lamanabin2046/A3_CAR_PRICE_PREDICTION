# app.py

import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
import pickle
import mlflow
import os

# -------------------------------
# MLflow model loading
# -------------------------------
mlflow.set_tracking_uri("http://mlflow:5001")

# Use the run ID of the logged model
run_id = "aa17fb00f4674785adf26704c2ab3575"
model_uri = f"runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Load scaler and encoders
scaler = pickle.load(open("Model/car-scaling.model", "rb"))
label_car = pickle.load(open("Model/car_brand_encoder.model", "rb"))
fuel_car = pickle.load(open("Model/car_fuel_encoder.model", "rb"))

# Load vehicle data for dropdowns
vehicle_df = pd.read_csv('Cars.csv')

# -------------------------------
# Constants
# -------------------------------
num_cols = ['max_power', 'year', 'mileage']
default_values = {'max_power': 82.4, 'year': 2017, 'fuel': 'Diesel', 'brand': 'Maruti', 'mileage': 19.42}
brand_cats = list(label_car.categories_[0])
fuel_cat = list(fuel_car.classes_)
y_map = {0: 'Cheap', 1: 'Average', 2: 'Expensive', 3: 'Very expensive'}

# -------------------------------
# Helper functions
# -------------------------------
def one_hot_transform(encoder, df, feature):
    encoded = encoder.transform(df[[feature]])
    categories = encoder.categories_[0]
    feature_df = pd.DataFrame(encoded.toarray(), columns=categories[1:])
    df = pd.concat([df, feature_df], axis=1)
    return df.drop(feature, axis=1)

def get_X(brand, year, max_power, mileage, fuel):
    features = {'brand': brand, 'year': year, 'max_power': max_power, 'mileage': mileage, 'fuel': fuel}
    for f in features:
        if not features[f]:
            features[f] = default_values[f]
        elif f in num_cols and features[f] < 0:
            features[f] = default_values[f]
    X = pd.DataFrame(features, index=[0])
    X[num_cols] = scaler.transform(X[num_cols])
    X['fuel'] = fuel_car.transform(X['fuel'])
    X = one_hot_transform(label_car, X, 'brand')
    return X.to_numpy(), features

def get_y(X):
    return model.predict(X)

# -------------------------------
# Initialize Dash
# -------------------------------
external_stylesheets = [dbc.themes.MORPH]
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# -------------------------------
# Layout helpers
# -------------------------------
def create_input_card(label, component):
    return dbc.Card([
        dbc.CardBody([
            html.H5(children=label, style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
            component
        ])
    ], style={'marginBottom': '20px', 'marginRight': '20px'})

# Input cards
card_brand = create_input_card(
    'Select Car Brand',
    dcc.Dropdown(id="brand3", options=[{"label": b, "value": b} for b in brand_cats], value=brand_cats[0])
)

card_year = create_input_card(
    'Select Year of Manufacture',
    dcc.Dropdown(
        id="year3",
        value=vehicle_df['year'].unique()[0],
        options=[{"label": i, "value": i} for i in sorted(vehicle_df['year'].unique())]
    )
)

card_fuel = create_input_card(
    'Select Fuel Type',
    dcc.Dropdown(id="fuel3", options=[{"label": f, "value": f} for f in fuel_cat], value=fuel_cat[0])
)

card_mileage = create_input_card(
    'Mileage (km/l)',
    dcc.Input(id="mileage3", type="number", value=default_values['mileage'], style={'width': '100%'})
)

card_power = create_input_card(
    'Maximum Power (bhp)',
    dcc.Input(id="max_power3", type="number", value=default_values['max_power'], style={'width': '100%'})
)

card_predicted_price = dbc.Card([
    dbc.CardBody([
        html.H3("Predicted Price Class: ", className="card-title"),
        html.H3(" ", id="selling_price3", className="card-text", style={"font-weight": "bold"})
    ])
], style={'marginBottom': '20px', 'marginRight': '20px'})

# -------------------------------
# App Layout
# -------------------------------
app.layout = dbc.Container([
    html.H1('A3 Assignment', style={'text-align': 'center', 'color':'#531406'}),
    html.H3('Predict car prices based on features', style={'text-align': 'center', 'color':'white', 'background-color': '#051C75'}),
    html.Hr(),
    dbc.CardLink([card_brand, card_year, card_power, card_mileage, card_fuel]),
    html.Br(),
    dbc.Button("Calculate Selling Price", id="submit3", style={'margin-bottom':'20px'}),
    html.Br(),
    dbc.CardLink([card_predicted_price])
])

# -------------------------------
# Callback
# -------------------------------
@callback(
    Output("selling_price3", "children"),
    Output("brand3", "value"),
    Output("year3", "value"),
    Output("max_power3", "value"),
    Output("mileage3", "value"),
    Output("fuel3", "value"),
    State("brand3", "value"),
    State("year3", "value"),
    State("max_power3", "value"),
    State("mileage3", "value"),
    State("fuel3", "value"),
    Input("submit3", "n_clicks"),
    prevent_initial_call=True
)
def predict_selling_price(brand, year, max_power, mileage, fuel, n_clicks):
    X, features = get_X(brand, year, max_power, mileage, fuel)
    predicted_class = get_y(X)[0]
    selling_price = y_map[predicted_class]
    return [selling_price] + list(features.values())

# -------------------------------
# Run server
# -------------------------------
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
