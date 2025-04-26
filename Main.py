# --*-- conding:utf-8 --*--
# @time:4/11/25 6:13 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File : benchmark.py

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc  # Optional for styling
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Adjust these file paths as needed.
train_x_path = "UCI HAR Dataset/train/X_train.txt"
train_y_path = "UCI HAR Dataset/train/y_train.txt"
test_x_path = "UCI HAR Dataset/test/X_test.txt"
test_y_path = "UCI HAR Dataset/test/y_test.txt"

# Read feature data using sep='\s+'.
X_train = pd.read_csv(train_x_path, sep='\s+', header=None)
X_test = pd.read_csv(test_x_path, sep='\s+', header=None)

# Read label data.
y_train = pd.read_csv(train_y_path, sep='\s+', header=None)
y_test = pd.read_csv(test_y_path, sep='\s+', header=None)

# Combine training and test sets.
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
y.columns = ['activity']

# Standardize the data (important for PCA and distance-based interactions).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Shape: (num_samples, 561)


# Use PCA to reduce 561-dimensional data to 2 dimensions for the scatter plot.
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Create a DataFrame for the low-dimensional view and add an 'id' column.
df = pd.DataFrame({
    "pca1": X_pca[:, 0],
    "pca2": X_pca[:, 1],
    "activity": y['activity']
})
df['id'] = df.index


# For display purposes, select the 5 best features from the original 561.
k_for_parallel = 5
selector = SelectKBest(f_classif, k=k_for_parallel)
X_selected = selector.fit_transform(X_scaled, y['activity'])

selected_feature_names = [f"feat_{i}" for i in range(k_for_parallel)]
df_parallel = pd.DataFrame(X_selected, columns=selected_feature_names)
df_parallel["activity"] = y['activity']
df_parallel["id"] = df.index
# Initially, no record is selected (highlighted = 0).
df_parallel["selected"] = 0


fig_scatter = px.scatter(
    df, x="pca1", y="pca2", color="activity",
    custom_data=["id"],
    title="PCA Low-Dimensional View"
)
fig_scatter.update_traces(marker=dict(size=7, line=dict(width=1, color='DarkSlateGrey')))


dimensions = [dict(label=col, values=df_parallel[col]) for col in selected_feature_names]
fig_parallel = go.Figure(data=
go.Parcoords(
    line=dict(
        color=df_parallel["selected"],
        colorscale=[[0, 'lightgray'], [1, 'red']],
        showscale=False,
        cmin=0, cmax=1
    ),
    dimensions=dimensions,
    customdata=df_parallel["id"]
)
)
fig_parallel.update_layout(title="High-Dimensional View (Parallel Coordinates, based on 5 features)")


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("HAR Dataset Linked Views"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(
                id="scatter-plot",
                figure=fig_scatter,
                config={"displayModeBar": True}
            )
        ], md=6),
        dbc.Col([
            dcc.Graph(
                id="parallel-plot",
                figure=fig_parallel,
                config={"displayModeBar": True}
            )
        ], md=6)
    ]),
    html.Div("Note: Use the Box or Lasso select tool on the scatter plot to select data points. "
             "When points are selected, the corresponding lines in the parallel coordinates plot should be highlighted. "
             "Hover over a line in the parallel coordinates plot to highlight its corresponding point in the scatter plot.")
], fluid=True)


@app.callback(
    Output("parallel-plot", "figure"),
    Input("scatter-plot", "selectedData"),
    State("parallel-plot", "figure")
)
def update_parallel(selectedData, fig_parallel_state):
    # Debug: print the incoming selectedData to check its structure.
    print("Selected Data from Scatter Plot:", selectedData)

    df_parallel_updated = df_parallel.copy()
    if selectedData is not None and "points" in selectedData:
        # Try to extract id from customdata; if missing, use pointIndex.
        selected_ids = []
        for pt in selectedData["points"]:
            if "customdata" in pt and pt["customdata"]:
                selected_ids.append(pt["customdata"][0])
            elif "pointIndex" in pt:
                selected_ids.append(pt["pointIndex"])
        print("Selected IDs:", selected_ids)
        df_parallel_updated["selected"] = df_parallel_updated["id"].apply(
            lambda x: 1 if x in selected_ids else 0
        )
    else:
        df_parallel_updated["selected"] = 0

    dims = [dict(label=col, values=df_parallel_updated[col]) for col in selected_feature_names]
    new_fig = go.Figure(data=
    go.Parcoords(
        line=dict(
            color=df_parallel_updated["selected"],
            colorscale=[[0, 'lightgray'], [1, 'red']],
            showscale=False,
            cmin=0, cmax=1
        ),
        dimensions=dims,
        customdata=df_parallel_updated["id"]
    )
    )
    new_fig.update_layout(title="High-Dimensional View (Parallel Coordinates, based on 5 features)")
    return new_fig



@app.callback(
    Output("scatter-plot", "figure"),
    Input("parallel-plot", "hoverData"),
    State("scatter-plot", "figure")
)
def update_scatter(hoverData, current_fig):
    fig = go.Figure(current_fig)
    new_marker = dict(size=7, line=dict(width=1, color='DarkSlateGrey'))

    if hoverData is not None and "points" in hoverData:
        # Extract hovered id from customdata if available; otherwise, use pointIndex.
        hovered_info = hoverData["points"][0]
        if "customdata" in hovered_info and hovered_info["customdata"]:
            hovered_id = hovered_info["customdata"][0]
        elif "pointIndex" in hovered_info:
            hovered_id = hovered_info["pointIndex"]
        else:
            hovered_id = None

        marker_lines = []
        all_ids = current_fig["data"][0]["customdata"]
        for pt in all_ids:
            # Each pt is a list, check first element.
            if pt[0] == hovered_id:
                marker_lines.append(dict(width=3, color='black'))
            else:
                marker_lines.append(dict(width=1, color='DarkSlateGrey'))
        fig.update_traces(marker=dict(size=7, line=marker_lines))
    else:
        fig.update_traces(marker=new_marker)
    return fig


if __name__ == '__main__':
    app.run(debug=True)
