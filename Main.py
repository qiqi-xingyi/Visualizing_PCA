import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc  # 可选，用于美化布局
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# -------------------------------
# 1. 数据读取与预处理
# -------------------------------
# 根据实际情况调整下面文件路径
train_x_path = "UCI HAR Dataset/train/X_train.txt"
train_y_path = "UCI HAR Dataset/train/y_train.txt"
test_x_path = "UCI HAR Dataset/test/X_test.txt"
test_y_path = "UCI HAR Dataset/test/y_test.txt"

# 读取特征数据（使用 sep='\s+' 替代 delim_whitespace=True）
X_train = pd.read_csv(train_x_path, sep='\s+', header=None)
X_test = pd.read_csv(test_x_path, sep='\s+', header=None)

# 读取标签数据
y_train = pd.read_csv(train_y_path, sep='\s+', header=None)
y_test = pd.read_csv(test_y_path, sep='\s+', header=None)

# 合并训练与测试数据
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
y.columns = ['activity']

# 数据标准化：PCA 与距离计算较敏感
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 数组形状: (样本数, 561)

# -------------------------------
# 2. 降维获得低维视图
# -------------------------------
# 使用 PCA 将 561 维数据降至二维用于散点图显示
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 将 PCA 结果合并入 DataFrame，并添加 id 用于关联
df = pd.DataFrame({
    "pca1": X_pca[:, 0],
    "pca2": X_pca[:, 1],
    "activity": y['activity']
})
df['id'] = df.index

# -------------------------------
# 3. 构造高维视图数据（平行坐标图）
# -------------------------------
# 为了展示效果，我们使用 SelectKBest 从 561 个特征中选取得分最高的 5 个特征
k_for_parallel = 5
selector = SelectKBest(f_classif, k=k_for_parallel)
X_selected = selector.fit_transform(X_scaled, y['activity'])

# 构造 DataFrame，其中列名为 feat_0, feat_1, ..., feat_4
selected_feature_names = [f"feat_{i}" for i in range(k_for_parallel)]
df_parallel = pd.DataFrame(X_selected, columns=selected_feature_names)
df_parallel["activity"] = y['activity']
df_parallel["id"] = df.index
# 初始状态下都未选中，后续回调中高亮的样本标记为 1
df_parallel["selected"] = 0

# -------------------------------
# 4. 构造初始图形（低维散点图和高维平行坐标图）
# -------------------------------

# 4.1 低维视图：PCA 散点图，保留 id 信息以便联动
fig_scatter = px.scatter(
    df, x="pca1", y="pca2", color="activity",
    custom_data=["id"],
    title="PCA 低维视图"
)
fig_scatter.update_traces(marker=dict(size=7, line=dict(width=1, color='DarkSlateGrey')))

# 4.2 高维视图：平行坐标图
# 构造 dimensions 列表
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
fig_parallel.update_layout(title="高维视图（平行坐标，基于 5 个特征）")

# -------------------------------
# 5. 构建 Dash 应用，实现双向联动
# -------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H2("HAR 数据集双向联动展示"),
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
    html.Div("注：在低维散点图中拉框选择数据点，或在高维平行坐标图中悬停于某条线，会联动高亮另一视图中的对应样本。")
], fluid=True)


# --------------------------------------------
# 5.1 回调1：当在散点图中选中数据时，更新平行坐标图的高亮显示
# --------------------------------------------
@app.callback(
    Output("parallel-plot", "figure"),
    Input("scatter-plot", "selectedData"),
    State("parallel-plot", "figure")
)
def update_parallel(selectedData, fig_parallel_state):
    df_parallel_updated = df_parallel.copy()
    if selectedData is not None and "points" in selectedData:
        selected_ids = [pt["customdata"] for pt in selectedData["points"]]
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
    new_fig.update_layout(title="高维视图（平行坐标，基于 5 个特征）")
    return new_fig


# --------------------------------------------
# 5.2 回调2：当在平行坐标图中悬停时，更新散点图中对应点的边框高亮
# --------------------------------------------
@app.callback(
    Output("scatter-plot", "figure"),
    Input("parallel-plot", "hoverData"),
    State("scatter-plot", "figure")
)
def update_scatter(hoverData, current_fig):
    fig = go.Figure(current_fig)
    new_marker = dict(size=7, line=dict(width=1, color='DarkSlateGrey'))

    if hoverData is not None and "points" in hoverData:
        hovered_id = hoverData["points"][0]["customdata"]
        marker_lines = []
        all_ids = current_fig["data"][0]["customdata"]
        for pt_id in all_ids:
            if pt_id == hovered_id:
                marker_lines.append(dict(width=3, color='black'))
            else:
                marker_lines.append(dict(width=1, color='DarkSlateGrey'))
        fig.update_traces(marker=dict(size=7, line=marker_lines))
    else:
        fig.update_traces(marker=new_marker)
    return fig


# -------------------------------
# 6. 运行 Dash 应用
# -------------------------------
if __name__ == '__main__':
    app.run_server(debug=True)
