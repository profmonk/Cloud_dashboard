import numpy as np 
import pandas as pd

df=pd.read_csv("Tableau Sample Sales Data.xlsx - Sales.csv")
df.head()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from IPython.display import display, HTML
display(HTML("<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"))


import plotly.graph_objects as go
import plotly.express as px

def plot_graph(f1, f2, graph_type):
    if graph_type == "Scatter Plot":
        fig = go.Figure(data=go.Scatter(x=df[f1], y=df[f2], mode='markers'))
        fig.update_layout(title='Scatter Plot', xaxis_title=f1, yaxis_title=f2)
        fig.show()

    elif graph_type == "Bar Chart":
        fig = go.Figure(data=go.Bar(x=df[f1], y=df[f2]))
        fig.update_layout(title='Bar Chart', xaxis_title=f1, yaxis_title=f2)
        fig.show()

    elif graph_type == "Box Plot":
        fig = px.box(df, x=f1, y=f2)
        fig.update_layout(title='Box Plot', xaxis_title=f1, yaxis_title=f2)
        fig.show()

    elif graph_type == "Pie Chart":
        fig = px.pie(df, names=f1, values=f2)
        fig.update_layout(title='Pie Chart')
        fig.show()

    else:
        fig = go.Figure(data=go.Scatter(x=df[f1], y=df[f2], mode='markers'))
        fig.show()


plot_graph("Sub-Category","Quantity","Pie Chart")