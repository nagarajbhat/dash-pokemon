# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input,Output
import plotly.express as px
from mymodel import type_prediction
import os

#variables

df = pd.read_csv('./data/pokemon.csv')

#print(df.columns)

#data creation
df_grass = df[df['Type 1']=='Grass']
df_legendary = df[df['Legendary']==True]
type1 = pd.Series(df['Type 1']).unique()

#ghost and groud  pokemons
#df_gd = df.loc[(df['Type 1']=='Ground')| (df['Type 1']=='Ghost')]
#pokemon_names = pd.Series(df_gd['Name']).unique()

graph_template="plotly_dark"

#app
#good themes - SLATE,BOOTSTRAP, DARKLY, CYBORG
app = dash.Dash('DashPokemon',meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
        ],
        external_stylesheets=[dbc.themes.CYBORG])
app.title = 'Dash Pokemon'

app_name = 'Dash Pokemon'

server = app.server

#layout_style = {'background-color':'#FF9F1C','font-size':15}
layout_style = {    }
#test
#app controls
controls = dbc.Form([
        dbc.FormGroup([
                dbc.Label("Pokemon Type"),
                dcc.Dropdown(
                id='dropdown_type',
                options = [{'label':i,'value':i} for i in type1],
                value='Grass', 

                ),

                            
        ]),
        
        dbc.FormGroup([
                dbc.Label("Criteria"),
                 dcc.Dropdown(
                id='dropdown_criteria',
                options = [{'label':i,'value':i} for i in ['Total', 'HP', 'Attack', 'Defense',
       'Sp. Atk', 'Sp. Def', 'Speed']],
                value='HP'
                )
                ])
],style={'inline-block':True})

controls2_a = dbc.Form([
        dbc.FormGroup([
                dbc.Label("strentgth criteria"),
                dcc.Dropdown(
                id='strength',
                options = [{'label':i,'value':i} for i in ['Total', 'HP', 'Attack', 'Defense',
       'Sp. Atk', 'Sp. Def', 'Speed']],
                value='Total', 
                ),

                            
        ]),
        
         dbc.FormGroup([
                dbc.Label("Stat type"),
                 dcc.RadioItems(
                id='stat_type',
                options = [{'label':i,'value':i} for i in ['median', 'max']],
                value='median',
                inputStyle={"margin-left": "20px"}

                )
                ])
                 ])
                 
controls2_b =  dbc.FormGroup([
                dbc.Label("Color"),
                 dcc.RadioItems(
                id='color_tab2',
                options = [{'label':i,'value':i} for i in ['Legendary', 'Type 1']],
                value='Type 1',
                inputStyle={"margin-left": "20px"}

                )
                ])
                                  
                
controls3_a = dbc.Form([
        dbc.FormGroup([
                dbc.Label("x axis"),
                dcc.Dropdown(
                id='x_axis',
                options = [{'label':i,'value':i} for i in ['Total', 'HP', 'Attack', 'Defense',
       'Sp. Atk', 'Sp. Def', 'Speed']],
                value='Attack', 
                ),

                            
        ]),
        
        dbc.FormGroup([
                dbc.Label("y axis"),
                 dcc.Dropdown(
                id='y_axis',
                options = [{'label':i,'value':i} for i in ['Total', 'HP', 'Attack', 'Defense',
       'Sp. Atk', 'Sp. Def', 'Speed']],
                value='Defense'
                )
                ])])
                 
controls3_b = dbc.FormGroup([
                dbc.Label("Color"),
                 dcc.RadioItems(
                id='color',
                options = [{'label':i,'value':i} for i in ['Legendary', 'Type 1', 'Generation']],
                value='Legendary',
                inputStyle={"margin-left": "20px"}

                )
                ])
                
controls4 = dbc.Form([
                 dbc.FormGroup([
                        html.H4("Select a pokemon to predict its type"),
                        dbc.Label("pokemon name"),
                        dcc.Dropdown(
                                id='pokemon_name',
                                #options = [{'label':i,'value':i} for i in pokemon_names],
                                options = [[{'label':i,'value':i} for i in ['Gastly','Haunter']]],
                                #value=pokemon_names[0],
                                value = 'Haunter'
                                ),

                        ]),
                html.Hr(),
                dbc.FormGroup([
                        dbc.Label("Pokemon types - to include in the data"),
                        dcc.Checklist(
                                id='pokemon_type',
                                options = [{'label':i,'value':i} for i in type1],
                                value=['Ground','Ghost'],
                                inputStyle={"margin-left": "20px"}

                               

                                )  ,
                      

                        ]),
                html.Hr(),
                dbc.FormGroup([
                        dbc.Label("Features"),
                        dcc.Checklist(
                                id='features_checklist',
                                options=[
                                        {'label': 'Total', 'value': 'Total'},
                                        {'label': 'Generation', 'value': 'Generation'},
                                        {'label': 'Legendary', 'value': 'Legendary'},
                                        {'label': 'Attack', 'value': 'Attack'},
                                        {'label': 'Defense', 'value': 'Defense'},
                                        {'label': 'Sp. Atk', 'value': 'Sp. Atk'},
                                        {'label': 'Sp. Def', 'value': 'Sp. Def'},
                                        {'label': 'HP', 'value': 'HP'},
                                        {'label':'Type 2','value':'Type 2'}
                                        
                                        ],
                                value=['Total','Generation','Legendary','Attack','Defense','Sp. Atk','Sp. Def','HP'],
                                inputStyle={"margin-left": "20px"}

                                )  ,
                      

                        ])
                        ])
                 

#tabs                 
tab1_content = dbc.Card([
        dbc.CardBody([
        
        dbc.Row([
                dbc.Col(controls, md=2),
                dbc.Col(dbc.Card([dbc.CardHeader(html.H6(id='p_text')),
                        dbc.CardBody(html.H4(id='b_type',style={'color':'white'}))]
                        ,color="info", inverse=True)
                        
                        ,md=3,style={'margin-top':20})
        ]),
        html.Br(),
       dbc.Card(
       dbc.Row([
                dbc.Col(dcc.Graph(id='generation',figure='fig'), md=6),
                html.Br(),
                dbc.Col(dcc.Graph(id='legendary',figure='fig'),md=6)
              ])
                        )
                 ])
                
                 ])
                        
                
tab2_content =  dbc.Card(
            dbc.CardBody([
                    
                 dbc.Row([
                dbc.Col(controls2_a, md=2),
                dbc.Col([controls2_b,html.Br(), html.H5(id='tab2_criteria'),],md=2),
                dbc.Col(dbc.Card([dbc.CardHeader(html.H6("The Weakest type is:")),
                        dbc.CardBody(html.H4(id='weakest_type',style={'color':'white'}))]
                        ,color="danger", inverse=True)),
                dbc.Col(dbc.Card([dbc.CardHeader(html.H6("The Strongest type is:")),
                        dbc.CardBody(html.H4(id='strongest_type',style={'color':'white'}))]
                        ,color="success", inverse=True)),
                ]),
                html.Br(),
                dbc.Col(dcc.Graph(id='strength_type',figure='fig'),md=12)
                 ]))

tab3_content =  dbc.Card(
            dbc.CardBody([
                dbc.Row([dbc.Col(controls3_a,md=2),
                         dbc.Col([dbc.Card([dbc.CardHeader(html.H6(id='corr_text')),
                        dbc.CardBody(html.H4(id='correlation',style={'color':'white'}))]
                        ,color="info", inverse=True),html.Br()],md=2),
                          dbc.Col([dcc.Graph(id='attack_defense_legendary',figure='fig'),html.Br()],md=3),
                        dbc.Col(dcc.Graph(id='attack_defense_type1',figure='fig'),md=3)]),
                        
                html.Br(),
                dbc.Row([
                       
                        dbc.Col(controls3_b,md=2),
                        dbc.Col(dcc.Graph(id='heatmap',figure='fig'),md=4),
                        #dbc.Col(html.H2(id='all_corr'),md=4),
                        

                        ])
                        ])
                        )
                                 

                     
"""
tab3_content =  dbc.Card(
            dbc.CardBody([
                dbc.Row([
                dbc.Col([
                 
                        dbc.Row([dbc.Col(controls3_a,md=2)]),

                        dbc.Row([
                        dbc.Col(dcc.Graph(id='attack_defense_legendary',figure='fig'),md=5),
                        dbc.Col(dcc.Graph(id='attack_defense_type1',figure='fig'),md=5),


                        ])
                        ],style={'position':'relative'}),
                                 
                 dbc.Col([
                      dbc.Row([dbc.Col(controls3_b,md=6)]),

                      dbc.Row([
                         dbc.Col([dcc.Graph(id='heatmap',figure='fig')])

                        ])
                         ])
                        
                 ])
                        ])
                 )     
     
"""
                                 
tab4_content =  dbc.Card(
            dbc.CardBody([
                 dbc.Row([
                dbc.Col(controls4, md=3),
                html.Br(),
                dbc.Col([dbc.Card([dbc.CardHeader([html.H6("The type predicted for"),html.H5(id='p_name')]),
                        dbc.CardBody(html.H4(id='predict_text',style={'color':'white'}))]
                        ,color="info", inverse=True),
                         html.Br(),
                        dbc.Card([dbc.CardHeader(html.H6("Model Accuracy:")),
                        dbc.CardBody(html.H4(id='model_acc',style={'color':'white'}))]
                        ,color="info", inverse=True),html.Br()],md=3),
                        
                dbc.Col([dcc.Graph(id='pokemon_stat',figure='fig'),html.Br()],md=3),
                dbc.Col(dcc.Graph(id='vc_type',figure='fig'),md=3   ),
                

                ]),
                html.Hr(),
               
                
                dbc.Row([
                html.H4("""Selected pokemon types:"""),
                html.H4(id="ptype_list",style={'color':'white'}),
                ]),
                dbc.Row([
                html.H4("""Selected Features:"""),
                html.H4(id="f_checklist",style={'color':'white'}),
                ]),
                
                html.Hr(),

                dbc.Row([
                dcc.Markdown("""
                             * We have used Random forest algorithm to classify Pokemon types.
                             * Select a pokemon to see its predicted type value.
                             * This model performs at 100% accuracy when classifying Ghost vs Ground type pokemon(select all fields except Type 2 to acheive this result)
                             * The class in this dataset in imbalanced.
                             * Hence Generally it performs better on binary classification when compared to multi-classification.
                             * Therefore select only 2 types to ge the best results.
                             * The Model accuracy will go down as you select more Pokemon types.
                             * Pokemon name list will get updated based on the types selected.
                             * The more features that you select,the better the model is likely to perform.
                                """),
                             ])
               
                ])
                 
                )

    #app layout
app.layout = dbc.Container([
       
                        
                        
        html.H3(app_name,style={'display': 'inline-block'}),
        html.Img(src=app.get_asset_url('bg-2.jpg'),height=50,style={'margin-left' :20,'margin-bottom':20,'display': 'inline-block',  'border-radius': 50}),
        #html.A(
         #   id = "gh-link",
          #  children = list("View on GitHub"),
           # href = "https://github.com/nagarajbhat/dash-pokemon",
           # style = {'color' : "white", 'margin-top':20,'font-size':15,'border' : "solid 1px white",'float':"right"}
          #),  
        
        dbc.Badge("github", href="https://github.com/nagarajbhat/dash-pokemon", color="secondary",
                              style = {'color' : "white", 'margin-top':20,'margin-right':10,'font-size':15,'border' : "solid 1px white",'float':"right"}  ),
        dbc.Badge("twitter", href="https://twitter.com/nagarajbhat92", color="secondary",
                              style = {'color' : "white", 'margin-top':20,'margin-right':10,'font-size':15,'border' : "solid 1px white",'float':"right"}  ),
        
                  dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Best Pokemons"),
        dbc.Tab(tab2_content, label="Strongest and Weakest Types"),
        dbc.Tab(tab3_content, label="Attack vs  Defense"),
        dbc.Tab(tab4_content, label="Pokemon Type Prediction")
             
                ])
                ],
                fluid=True,style=layout_style
                )
        
 
#callbacks
@app.callback([Output('generation','figure'),
              Output('p_text','children'),
              Output('b_type','children')],
              [Input('dropdown_type','value'),
               Input('dropdown_criteria','value')])
      
def update_fig(dropdown_type,dropdown_criteria):
    filtered_df = df[df['Type 1']==dropdown_type]
    fig = px.bar(filtered_df,y='Name',x=dropdown_criteria,color='Generation',template=graph_template,height=500  )
    best_pokemon = filtered_df.loc[filtered_df[dropdown_criteria].idxmax()].Name
    text = f"Pokemon  of {dropdown_type} type with the best {dropdown_criteria} is : "
    #container = f"{dropdown_type} type selected"
    fig.update_traces(textfont_size=30)     

    fig.update_layout(title= "Six Generations of pokemons",uniformtext_minsize=15, transition_duration=500)

    return fig,text,best_pokemon

@app.callback(Output('legendary','figure'),
              [Input('dropdown_type','value'),
               Input('dropdown_criteria','value')])

def update_type_compare_fig(dropdown_type,dropdown_criteria):
    filtered_df = df[df['Type 1']==dropdown_type]
    
    fig = px.bar(filtered_df,y='Name',x=dropdown_criteria,color='Legendary',template=graph_template,height=500)

    fig.update_traces(textfont_size=30)

    fig.update_layout(title= "Legendary/Non-legendary pokemons ",uniformtext_minsize=15, uniformtext_mode='hide',transition_duration=500)

    return fig


@app.callback([Output('strength_type','figure'),
               Output('tab2_criteria','children'),
               Output('weakest_type','children'),
               Output('strongest_type','children'),
               ],
              [Input('strength','value'),
               Input('color_tab2','value'),
               Input('stat_type','value')])

def strength_fig(strength,color_tab2,stat_type):
    #filtered_df = df[df['Name']==pokemon_name]
    
    #df = px.data.iris()
    fig = px.box(df, x=strength, y="Type 1", points="all",hover_name="Name",color=color_tab2,template=graph_template)

    #fig.update_traces(textfont_size=30)

    fig.update_layout(title="Boxplot for different types of pokemon",uniformtext_minsize=15, uniformtext_mode='hide',transition_duration=500,height=500)
    #finding meadian values of all Type 1
    if(stat_type == 'median'):
        type_medians = {type:df[df['Type 1']==type][strength].median() for type  in df['Type 1'].unique()}
    elif(stat_type == 'max'):
        type_medians = {type:df[df['Type 1']==type][strength].max() for type  in df['Type 1'].unique()}

    key_max = max(type_medians.keys(), key=(lambda k: type_medians[k]))
    key_min = min(type_medians.keys(), key=(lambda k: type_medians[k]))
    
    
    tab2_criteria = f"Based on the {stat_type} {strength} value:"
    
    return fig, tab2_criteria, key_min,key_max



@app.callback([Output('attack_defense_legendary','figure'),
              Output('corr_text','children'),
             Output('correlation','children')]  ,
              [Input('x_axis','value'),
               Input('y_axis','value')])

def attack_defense_leg_fig(x_axis,y_axis):
    fig = px.scatter(df,x=x_axis,y=y_axis,color='Legendary',hover_name="Name",hover_data=["Attack","Defense","HP","Total"],template=graph_template,height=300)
    fig.update_traces(textfont_size=30)

    fig.update_layout(title="scatterplot - legendary/non legendary pokemons",uniformtext_minsize=15, uniformtext_mode='hide',transition_duration=500)
    corr_text = f"Correlation between {x_axis} and {y_axis} is:"
    correlation = round(df[x_axis].corr(df[y_axis]),2)
    return fig,corr_text,correlation


@app.callback(Output('attack_defense_type1','figure'),
              [Input('x_axis','value'),
               Input('y_axis','value')])

def attack_defense_type1_fig(x_axis,y_axis):
    
    fig = px.scatter(df,x=x_axis,y=y_axis,color='Type 1',hover_name="Name",hover_data=["Attack","Defense","HP","Total"],template=graph_template,height=300)
    fig.update_traces(textfont_size=30)

    fig.update_layout(title="scatterplot - different types of pokemon",uniformtext_minsize=15, uniformtext_mode='hide',transition_duration=500)

    return fig
 
@app.callback(Output('heatmap','figure'),
              [Input('color','value')])

def heatmap_fig(color):
    #filtered_df = df[df['Name']==pokemon_name]
    
    #df = px.data.iris()
    fig = px.scatter_matrix(df,
    dimensions=["Total","HP","Attack","Defense","Sp. Atk","Sp. Def","Speed"],
    color= color, hover_name="Name",template=graph_template,height=500)

    #fig.update_traces(textfont_size=30)
    fig.update_layout(title="A pairplot showing relationship between different features of pokemon",uniformtext_minsize=15, uniformtext_mode='hide',transition_duration=500)
    #all_corr = df.corr()

    return fig

@app.callback(Output('pokemon_name','options'),
              [Input('pokemon_type','value')])

def update_pokemon_type(pokemon_type):
    df_gd = df[df['Type 1'].isin(pokemon_type)]
    p_names = pd.Series(df_gd['Name']).unique()
    pokemon_names = [{'label':i,'value':i} for i in p_names]
    return pokemon_names


@app.callback([Output('p_name','children'),     
                Output('predict_text','children'),
               Output('model_acc','children'),
               Output('f_checklist','children'),
               Output('ptype_list','children'),
               Output('pokemon_stat','figure'),
               Output('vc_type','figure')
               
                              
               ],
              [Input('pokemon_name','value'),
               Input('features_checklist','value'),
               Input('pokemon_type','value'),
               ])

def prediction_update(pokemon_name,features_checklist,pokemon_type):
    #filtered_df = df[df['Name']==pokemon_name]
    df_gd = df[df['Type 1'].isin(pokemon_type)]
    #Model prediction
    p_type,acc,fig,clf_report = type_prediction(pokemon_name,df_gd,features_checklist)
    
    p_name=pokemon_name
    #text = f"The type predicted for Pokemon {pokemon_name} is {p_type}."
    model_acc = round(acc*100,2) 
    f_checklist =  [v + str(', ') for v in features_checklist]
    ptype_list =   [v + str(', ') for v in pokemon_type]
    
    #plotting pokemon stat figure
    data = df[df['Name']==pokemon_name].values
    cols = list(df[df['Name']==pokemon_name].columns)
    fig = px.bar(data,x=cols[4:11],y=data[0][4:11],template=graph_template,height=300)
    
    #fig = px.bar(df,x='Defense',y='Attack',width=500,height=500)
    fig.update_traces(textfont_size=30)
    fig.update_layout(title='Pokemon stat',uniformtext_minsize=15, uniformtext_mode='hide',transition_duration=50)

    #pie chart - 
    vc = df_gd['Type 1'].value_counts()
    vc_types = list(vc.index)
    vc_counts = list(vc.values)
    vc_type_fig = px.pie(vc, values=vc_counts, names=vc_types,template=graph_template,height=300)
    vc_type_fig.update_traces(hole=0.4,textfont_size=30)
    vc_type_fig.update_layout(title='Selected Pokemon types share',uniformtext_minsize=15, uniformtext_mode='hide',transition_duration=50)
   
    
    return p_name,p_type,model_acc,f_checklist,ptype_list,fig ,vc_type_fig


if __name__ == '__main__':
    app.run_server(debug=False)
