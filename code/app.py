# app.py

import dash
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import pandas as pd
from model import get_y, get_X  # MLflow-based functions

# -------------------------------
# Load dataset for dropdowns
# -------------------------------
vehicle_df = pd.read_csv("Cars.csv")

# -------------------------------
# Constants
# -------------------------------
default_values = {
    "brand": "Maruti",
    "year": 2017,
    "max_power": 82.4,
    "mileage": 19.42,
    "fuel": "Diesel"
}
brand_cats = vehicle_df["brand"].unique().tolist()
fuel_cat = vehicle_df["fuel"].unique().tolist()
y_map = {0: "Cheap", 1: "Average", 2: "Expensive", 3: "Very Expensive"}

# -------------------------------
# Helpers
# -------------------------------
def prepare_features(brand, year, max_power, mileage, fuel):
    """
    Prepare features for MLflow model
    """
    brand = brand or default_values["brand"]
    year = year or default_values["year"]
    max_power = max_power or default_values["max_power"]
    mileage = mileage or default_values["mileage"]
    fuel = fuel or default_values["fuel"]

    X, features = get_X(brand, year, km_driven=mileage, owner=0, fuel=fuel)
    return X, features

def create_input_card(label, component):
    return dbc.Card(
        dbc.CardBody([
            html.H5(label, style={"text-align": "center", "color": "white", "background-color": "#051C75"}),
            component
        ]),
        style={"marginBottom": "20px"}
    )

# -------------------------------
# Input components
# -------------------------------
card_brand = create_input_card(
    "Select Car Brand",
    dcc.Dropdown(id="brand3", options=[{"label": b, "value": b} for b in brand_cats], value=brand_cats[0])
)

card_year = create_input_card(
    "Select Year of Manufacture",
    dcc.Dropdown(
        id="year3",
        options=[{"label": i, "value": i} for i in sorted(vehicle_df["year"].unique())],
        value=int(vehicle_df["year"].median())
    )
)

card_fuel = create_input_card(
    "Select Fuel Type",
    dcc.Dropdown(id="fuel3", options=[{"label": f, "value": f} for f in fuel_cat], value=fuel_cat[0])
)

card_mileage = create_input_card(
    "Mileage (km/l)",
    dcc.Input(id="mileage3", type="number", value=default_values["mileage"], style={"width": "100%"})
)

card_power = create_input_card(
    "Maximum Power (bhp)",
    dcc.Input(id="max_power3", type="number", value=default_values["max_power"], style={"width": "100%"})
)

card_predicted_price = dbc.Card(
    dbc.CardBody([
        html.H3("Predicted Price Class:", className="card-title"),
        html.H3(" ", id="selling_price3", className="card-text", style={"font-weight": "bold"})
    ]),
    style={"marginBottom": "20px"}
)

# -------------------------------
# Dash App
# -------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])
server = app.server

app.layout = dbc.Container([
    html.H1("A3 Assignment", style={"text-align": "center", "color": "#531406"}),
    html.H3("Predict car prices based on features",
            style={"text-align": "center", "color": "white", "background-color": "#051C75"}),
    html.Hr(),
    dbc.Row([dbc.Col(card_brand, md=4), dbc.Col(card_year, md=4), dbc.Col(card_fuel, md=4)]),
    dbc.Row([dbc.Col(card_power, md=6), dbc.Col(card_mileage, md=6)]),
    html.Br(),
    dbc.Button("Calculate Selling Price", id="submit3", style={"margin-bottom": "20px"}),
    html.Br(),
    dbc.Row([dbc.Col(card_predicted_price, md=12)])
], fluid=True)

# -------------------------------
# Callback
# -------------------------------
@callback(
    Output("selling_price3", "children"),
    Input("submit3", "n_clicks"),
    State("brand3", "value"),
    State("year3", "value"),
    State("max_power3", "value"),
    State("mileage3", "value"),
    State("fuel3", "value"),
    prevent_initial_call=True
)
def predict_price_callback(n_clicks, brand, year, max_power, mileage, fuel):
    X, _ = prepare_features(brand, year, max_power, mileage, fuel)
    predicted_class = get_y(X)[0]
    return y_map[predicted_class]

# -------------------------------
# Run Server
# -------------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
