import pandas as pd
import dash
import dash_core_components as dcc   #引入dash核心元素
import dash_html_components as html   #引入dash html元素

df = pd.DataFrame([
    {'city': 'hangzhou', 'man': 150, 'woman': 0.523, 'city-level': '2nd'},
    {'city': 'shanghai', 'man': 200, 'woman': 0.26, 'city-level': '1st'},
    {'city': 'shenzhen', 'man': 150, 'woman': 0.37, 'city-level': '1st'},
    {'city': 'wuhan', 'man': 160, 'woman': 0.17, 'city-level': '2nd'},
    {'city': 'ningbo', 'man': 100, 'woman': 0.88, 'city-level': '3rd'},
])

app = dash.Dash()
app.layout = html.Div([

    # 散点图
    dcc.Graph(
        id='graph-scatter',
        className='my_graph',
        figure={
            'data': [{
                'x': df['city'],
                'y': df['woman'],
                'type': 'scatter',
                'mode': 'markers',
            }],
            'layout': {
                'title': '各城市女性占比',
                'height': 600,
                'yaxis': {'hoverformat': '.2%'},
                'margin': {'l': 35, 'r': 35, 't': 50, 'b': 80},
            }
        },
        config={
            'displayModeBar': False
        },
    ),

    # 折线图
    dcc.Graph(
        id='graph-line',
        className='my_graph',
        figure={
            'data': [{
                'x': df['city'],
                'y': df['man'],
                'type': 'scatter',
                'mode': 'lines+markers',
            }],
            'layout': {
                'title': '各城市男性人口',
                'height': 600,
                'yaxis': {'hoverformat': '.0f'},
                'margin': {'l': 35, 'r': 35, 't': 50, 'b': 80},
            }
        },
        config={
            'displayModeBar': False
        },
    ),

    # 柱形图
    dcc.Graph(
        id='graph-bar',
        className='my_graph',
        figure={
            'data': [{
                'x': df['city'],
                'y': df['man'],
                'type': 'bar',
            }],
            'layout': {
                'title': '各城市男性人口',
                'height': 600,
                'yaxis': {'hoverformat': '.0f'},
                'margin': {'l': 35, 'r': 35, 't': 50, 'b': 80},
            }
        },
        config={
            'displayModeBar': False
        },
    ),

    # 横向柱形图
    dcc.Graph(
        id='graph-bar-horizontal',
        className='my_graph',
        figure={
            'data': [{
                'x': df['man'],
                'y': df['city'],
                'type': 'bar',
                'orientation': 'h'
            }],
            'layout': {
                'title': '各城市男性人口',
                'height': 600,
                'yaxis': {'hoverformat': '.0f'},
                'margin': {'l': 80, 'r': 35, 't': 50, 'b': 80},
            }
        },
        config={
            'displayModeBar': False
        },
    ),

    # 饼图
    dcc.Graph(
        id='graph-pie',
        className='my_graph',
        figure={
            'data': [{
                'labels': df['city-level'].value_counts().index,
                'values': df['city-level'].value_counts().values,
                'type': 'pie',
                'hoverinfo': 'label+value+percent',
                'hole': 0.4,
            }],
            'layout': {
                'title': '城市等级占比',
                'margin': {'l': 30, 'r': 30},
                'height': 600,
            }
        },
        config={
            'displayModeBar': False
        }
    ),

    # 表格
    dcc.Graph(
        id='graph-table',
        className='my_table',
        figure={
            'data': [{
                'type': 'table',
                'columnwidth': 0.3,
                'header': {
                    'values': [['<b>{}</b>'.format(i)] for i in df.columns],
                    'font': {'size': 13, 'color': 'white', },
                    'align': 'center',
                    'height': 30,
                    'fill': {'color': '#0076BA'},
                },
                'cells': {
                    'values': df.values.T,
                    'line': {'color': 'rgb(50, 50, 50)'},
                    'align': 'center',
                    'height': 27,
                    'fill': {'color': ['#56C1FF', '#f5f5fa']},
                    'format': [None]*3 + ['.2%']
                },
            }],

            'layout': {
                'height': 250,
                'margin': {'l': 10, 'r': 10, 't': 50, 'b': 10},
            }
        },
        config={'displayModeBar': False}
    ),

    # 图表排版
    html.Div([
        html.Div([
            dcc.Graph(
                id='graph-scatter2',
                className='my_graph',
                figure={
                    'data': [{
                        'x': df['city'],
                        'y': df['woman'],
                        'type': 'scatter',
                        'mode': 'markers',
                    }],
                    'layout': {
                        'title': '各城市女性占比',
                        'height': 600,
                        'yaxis': {'hoverformat': '.2%'},
                        'margin': {'l': 35, 'r': 35, 't': 50, 'b': 80},
                    }
                },
                config={
                    'displayModeBar': False
                },
            ),
            dcc.Graph(
                id='graph-line2',
                className='my_graph',
                figure={
                    'data': [{
                        'x': df['city'],
                        'y': df['man'],
                        'type': 'scatter',
                        'mode': 'lines+markers',
                    }],
                    'layout': {
                        'title': '各城市男性人口',
                        'height': 600,
                        'yaxis': {'hoverformat': '.0f'},
                        'margin': {'l': 35, 'r': 35, 't': 50, 'b': 80},
                    }
                },
                config={
                    'displayModeBar': False
                },
            ),
        ], style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(
                id='graph-bar2',
                className='my_graph',
                figure={
                    'data': [{
                        'x': df['city'],
                        'y': df['man'],
                        'type': 'bar',
                    }],
                    'layout': {
                        'title': '各城市男性人口',
                        'height': 600,
                        'yaxis': {'hoverformat': '.0f'},
                        'margin': {'l': 35, 'r': 35, 't': 50, 'b': 80},
                    }
                },
                config={
                    'displayModeBar': False
                },
            ),
            dcc.Graph(
                id='graph-bar-horizontal2',
                className='my_graph',
                figure={
                    'data': [{
                        'x': df['man'],
                        'y': df['city'],
                        'type': 'bar',
                        'orientation': 'h'
                    }],
                    'layout': {
                        'title': '各城市男性人口',
                        'height': 600,
                        'yaxis': {'hoverformat': '.0f'},
                        'margin': {'l': 80, 'r': 35, 't': 50, 'b': 80},
                    }
                },
                config={
                    'displayModeBar': False
                },
            ),
        ], style={'width': '49%', 'display': 'inline-block'}),
    ]),

'''
更复杂的图形如组合图、玫瑰图、sankey图等均可在plotly官网中找到
各图形的形状、格式、图例等的调整也均可在官方文档中找到
'''

])

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port='8052', debug=True)