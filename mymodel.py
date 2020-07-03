# -*- coding: utf-8 -*-

#depedencies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#df = pd.read_csv('./data/pokemon.csv')
#df_gd = df.loc[(df['Type 1']=='Ground')| (df['Type 1']=='Ghost')]



def type_prediction(pokemon_name,df,features):
        df = df.set_index('#')
        #cor = df.corr()
        #print(sns.heatmap(cor))
        
        
        #X,y values
        #X = df.filter(['Legendary','Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Type 2'])
        X = df.filter(features)

        y = df.filter(['Type 1'])

        # Break off validation set from training data
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

        # "Cardinality" means the number of unique values in a column
        # Select categorical columns with relatively low cardinality (convenient but arbitrary)
        categorical_cols = [cname for cname in X_train.columns if
                    X_train[cname].dtype in ["object"]]

        # Select numerical columns
        numerical_cols = [cname for cname in X_train.columns if 
                X_train[cname].dtype in ['int64', 'float64','bool']]
        # Preprocessing for numerical data
        numerical_transformer = SimpleImputer(strategy='constant')

        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),

            ])

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
        transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
                ])

        # Define model
        model = RandomForestClassifier(n_estimators=1000, random_state=0)
        #model = KNeighborsClassifier(n_neighbors=7)
        
        # Bundle preprocessing and modeling code in a pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])
        
        # Preprocessing of training data, fit model 
        clf.fit(X_train, y_train)
        
        # Preprocessing of validation data, get predictions
        preds = clf.predict(X_valid)
        
        clf_report = classification_report(y_valid,preds)
        #model accuracy
        accuracy = accuracy_score(y_valid, preds)
        
        #predicting pokemon type from name
        X_to_predict = df[df['Name']==pokemon_name].drop(['Type 1','Name'],axis=1)
        y_value = df[df['Name']==pokemon_name]['Type 1']
        predicted = clf.predict(X_to_predict)
        #individual accuracy
        #accuracy = accuracy_score(y_value, predicted)
        pokemon_type = predicted[0]
        
        
       
        fig=""
        return pokemon_type,accuracy,fig,clf_report

#pokemon_type,accuracy,fig = type_prediction('Sandslash',df_gd)
#print(pokemon_type,accuracy,fig)