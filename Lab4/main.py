import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import streamlit.components.v1 as components

#Import classification models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.model_selection import cross_val_score

#Import performance metrics, imbalanced rectifiers
from sklearn.metrics import  confusion_matrix,classification_report,matthews_corrcoef
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
np.random.seed(42) #for reproducibility since SMOTE and Near Miss use randomizations

from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from yahoo_fin import stock_info as si

import numpy as np
import chart_studio.plotly as plotly
import plotly.figure_factory as ff
from plotly import graph_objs as go
import glob


from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import streamlit as st
import warnings
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import re

import yfinance as yf
import base64
import sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import plotly.express as px
import plotly.figure_factory as ff
import time

from PIL import Image
import plotly.graph_objects as go

from datetime import datetime
import os
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
#import talib

from typing import TypeVar, Callable, Sequence
from functools import reduce

import plotly.graph_objects as go

from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
import shap

st.set_option('deprecation.showPyplotGlobalUse', False)

T = TypeVar('T')
#nltk.download(['punkt', 'wordnet'])

@st.cache(allow_output_mutation=True)
def profile_clusters(df_profile, categorical=None):
    # get the important features via LGBM and SHAP
    X = df_profile.drop('cluster', axis=1)
    y = df_profile['cluster']

    clf = LGBMClassifier(class_weight='balanced', colsample_bytree=0.6)
    scores = cross_val_score(clf, X, y, scoring='f1_weighted')

    print(scores.mean())
    # Fit the model if
    if scores.mean() > 0.5:
        clf.fit(X, y)
    else:
        raise ValueError("Clusters are not distinguishable. Can't profile. ")

    # Get importance
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)

    # Get 7 most important features
    importance_dict = {f: 0 for f in X.columns}
    topn = 7
    topn = min(len(X.columns), topn)
    for c in np.unique(df_profile['cluster']):
        shap_df = pd.DataFrame(shap_values[c], columns=X.columns)
        abs_importance = np.abs(shap_df).sum()
        for f in X.columns:
            importance_dict[f] += abs_importance[f]

    importance_dict = {k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)}
    important_features = [k for k, v in importance_dict.items()]
    n_important_features = [k for k, v in importance_dict.items()][:topn]

    # Dataframe output
    for k in np.unique(df_profile['cluster']):
        if k == 0:
            profile = pd.DataFrame(columns=['cluster', 'feature', 'mean_value'], index=range(len(n_important_features)))
            profile['cluster'] = k
            profile['feature'] = n_important_features
            profile['mean_value'] = df_profile.loc[df_profile.cluster == k, n_important_features].mean().values
        else:
            profile_2 = pd.DataFrame(columns=['cluster', 'feature', 'mean_value'],
                                     index=range(len(n_important_features)))
            profile_2['cluster'] = k
            profile_2['feature'] = n_important_features
            profile_2['mean_value'] = df_profile.loc[df_profile.cluster == k, n_important_features].mean().values
            profile = pd.concat([profile, profile_2])

    profile.reset_index(drop=True, inplace=True)

    # Scaling for plotting
    for c in X.columns:
        df_profile[c] = MinMaxScaler().fit_transform(np.array(df_profile[c]).reshape(-1, 1))

    # Plotly output
    cluster_names = [f'Cluster {k}' for k in np.unique(df_profile['cluster'])]  # clust 1, 2, 3
    data = [go.Bar(name=f, x=cluster_names, y=df_profile.groupby('cluster')[f].mean()) for f in n_important_features]
    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(barmode='group')

    return fig, profile, important_features

def profile_feature(df_profile, feature):
    if df_profile[feature].nunique() > 2:
        box_data = [go.Box(y=df_profile.loc[df_profile.cluster == k, feature].values, name=f'Cluster {k}') for k in np.unique(df_profile.cluster)]
        fig = go.Figure(data=box_data)
    else:
        x =[f'Cluster {k}' for k in np.unique(df_profile.cluster)]
        y = [df_profile.loc[df_profile.cluster == k, feature].mean() for k in np.unique(df_profile.cluster)]
        fig = go.Figure([go.Bar(x=x, y=y)])
    return fig

def get_table_download_link(df, filename, linkname):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{linkname}</a>'
    return href

prof = False

st.title('SIA Partners Data Analytics Event')

df=st.cache(pd.read_csv)('data/creditcard.csv')
df = df.sample(frac=0.1, random_state = 48)

df2=pd.read_csv('data/data.csv' , sep=";")

app_mode = st.sidebar.selectbox('Mode', ['About', 'EDA Credit Card Fraud', 'Analysis Credit Card Fraud', 'EDA Claims Prediction', 'Analysis Claims Prediction', 'Monte Carlo', 'NLP_Analysis', 'Prediction', 'Customer Clustering'])

if app_mode == "About":
    st.markdown('Data Analytics demonstration dashboard')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
            width: 350px
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
            width: 350px
            margin-left: -350px
        }
        </style>
        """,

        unsafe_allow_html=True,
    )

    st.markdown('')

    st.markdown('This dashboard allows you to follow the SIA Partners Data Analytics examples of today. We will introduce some techniques which can help you with exloratory data analysis, fraud and claim prediction, customer segmentation and more.')
    st.markdown('')
    st.markdown('Please remember that this is only a demo based on data which was prepared specifically for this exercise. In practice, a lot of time is necessary to get the data to this point & projects in this field often focus on these first initial steps')


elif app_mode == 'EDA Credit Card Fraud':

    st.title("Credit Card Fraud Detection")
    st.markdown('')
    st.markdown('Our first example focuses on the detection of credit card fraud. This demo is based on real credit card data where the first steps in the analysis have already been performed. This means that the data has been aggregated out of different systems and through a technique called PCA or Principal Component Analysis the most important features have already been selected for our model development' )
    st.sidebar.subheader(' Quick  Explore')

    st.subheader('Dataset')
    df

    st.subheader('Show Columns List')
    all_columns = df.columns.to_list()
    st.write(all_columns)

    st.subheader('Statistical Data Descripition')
    st.write(df.describe())

    st.subheader('Missing values')
    st.write(df.isnull().sum())


elif app_mode == "Analysis Credit Card Fraud":

    # Print shape and description of the data
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if st.sidebar.checkbox('Show DataFrame'):
        st.write(df.head(100))
        st.write('Shape of the dataframe: ',df.shape)
        st.write('Data decription: \n',df.describe())
    # Print valid and fraud transactions
    fraud=df[df.Class==1]
    valid=df[df.Class==0]
    outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100
    if st.sidebar.checkbox('Show fraud and valid transaction details'):
        st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
        st.write('Fraud Cases: ',len(fraud))
        st.write('Valid Cases: ',len(valid))


        #Obtaining X (features) and y (labels)
    X=df.drop(['Class'], axis=1)
    y=df.Class

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)

    #Print shape of train and test sets
    if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
        st.write('X_train: ',X_train.shape)
        st.write('y_train: ',y_train.shape)
        st.write('X_test: ',X_test.shape)
        st.write('y_test: ',y_test.shape)


    #Import classification models and metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
    from sklearn.model_selection import cross_val_score


    logreg=LogisticRegression()
    svm=SVC()
    knn=KNeighborsClassifier()
    etree=ExtraTreesClassifier(random_state=42)
    rforest=RandomForestClassifier(random_state=42)


    features=X_train.columns.tolist()


    #Feature selection through feature importance
    @st.cache
    def feature_sort(model,X_train,y_train):
        #feature selection
        mod=model
        # fit the model
        mod.fit(X_train, y_train)
        # get importance
        imp = mod.feature_importances_
        return imp

    #Classifiers for feature importance
    clf=['Extra Trees','Random Forest']
    mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)

    start_time = timeit.default_timer()
    if mod_feature=='Extra Trees':
        model=etree
        importance=feature_sort(model,X_train,y_train)
    elif mod_feature=='Random Forest':
        model=rforest
        importance=feature_sort(model,X_train,y_train)
    elapsed = timeit.default_timer() - start_time
    st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))

    #Plot of feature importance
    if st.sidebar.checkbox('Show plot of feature importance'):
        plt.bar([x for x in range(len(importance))], importance)
        plt.title('Feature Importance')
        plt.xlabel('Feature (Variable Number)')
        plt.ylabel('Importance')
        st.pyplot()

    feature_imp=list(zip(features,importance))
    feature_sort=sorted(feature_imp, key = lambda x: x[1])

    n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

    top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

    if st.sidebar.checkbox('Show selected top features'):
        st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

    X_train_sfs=X_train[top_features]
    X_test_sfs=X_test[top_features]

    X_train_sfs_scaled=X_train_sfs
    X_test_sfs_scaled=X_test_sfs
    smt = SMOTE()
    nr = NearMiss()
    def compute_performance(model, X_train, y_train,X_test,y_test):
        start_time = timeit.default_timer()
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
        'Accuracy: ',scores
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        cm=confusion_matrix(y_test,y_pred)
        'Confusion Matrix: ',cm
        cr=classification_report(y_test, y_pred)
        'Classification Report: ',cr
        mcc= matthews_corrcoef(y_test, y_pred)
        'Matthews Correlation Coefficient: ',mcc
        elapsed = timeit.default_timer() - start_time
        'Execution Time for performance computation: %.2f minutes'%(elapsed/60)
    #Run different classification models with rectifiers
    if st.sidebar.checkbox('Run a credit card fraud detection model'):

        alg=['Extra Trees','Random Forest','k Nearest Neighbor','Support Vector Machine','Logistic Regression']
        classifier = st.sidebar.selectbox('Which algorithm?', alg)
        rectifier=['SMOTE','Near Miss','No Rectifier']
        imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier)

        if classifier=='Logistic Regression':
            model=logreg
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)


        elif classifier == 'k Nearest Neighbor':
            model=knn
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

        elif classifier == 'Support Vector Machine':
            model=svm
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

        elif classifier == 'Random Forest':
            model=rforest
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

        elif classifier == 'Extra Trees':
            model=etree
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)


elif app_mode == 'EDA Claims Prediction':

    st.title("Claims Prediction EDA")
    st.markdown('')
    st.markdown('Our second demonstration focuses on the development of a claims development algorithm based on a clean dataset. Once again, this dataset was prepared for this exercise so that major problems have already been dealt with.' )
    st.sidebar.subheader('Quick  Explore')

    st.subheader('Dataset')
    df2

    st.subheader('Show Columns List')
    all_columns = df2.columns.to_list()
    st.write(all_columns)

    st.subheader('Statistical Data Descripition')
    st.write(df2.describe())
    st.subheader('Missing values')
    st.write(df2.isnull().sum())

elif app_mode == "Analysis Claims Prediction":

    # Print shape and description of the data
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if st.sidebar.checkbox('Show DataFrame'):
        st.write(df2.head(100))
        st.write('Shape of the dataframe: ',df.shape)
        st.write('Data decription: \n',df.describe())
    # Print valid and fraud transactions
    fraud=df2[df2.ClaimNb==1]
    valid=df2[df2.ClaimNb==0]
    outlier_percentage=(df2.ClaimNb.value_counts()[1]/df2.ClaimNb.value_counts()[0])*100
    if st.sidebar.checkbox('Show claim and valid transaction details'):
        st.write('Claim cases are: %.3f%%'%outlier_percentage)
        st.write('Claim Cases: ',len(fraud))
        st.write('Valid Cases: ',len(valid))


        #Obtaining X (features) and y (labels)
    df2=df2.drop(['PolicyID'], axis=1)
    df2=df2.drop(['Power'], axis=1)
    df2=df2.drop(['Brand'], axis=1)
    df2=df2.drop(['Gas'], axis=1)
    df2=df2.drop(['Region'], axis=1)
    df2=df2.dropna()
    X=df2.drop(['ClaimNb'], axis=1)
    y=df2.ClaimNb

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    size = st.sidebar.slider('Test Set Size', min_value=0.2, max_value=0.4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = size, random_state = 42)

    #Print shape of train and test sets
    if st.sidebar.checkbox('Show the shape of training and test set features and labels'):
        st.write('X_train: ',X_train.shape)
        st.write('y_train: ',y_train.shape)
        st.write('X_test: ',X_test.shape)
        st.write('y_test: ',y_test.shape)


    #Import classification models and metrics
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
    from sklearn.model_selection import cross_val_score


    logreg=LogisticRegression()
    svm=SVC()
    knn=KNeighborsClassifier()
    etree=ExtraTreesClassifier(random_state=42)
    rforest=RandomForestClassifier(random_state=42)


    features=X_train.columns.tolist()


    #Feature selection through feature importance
    @st.cache
    def feature_sort(model,X_train,y_train):
        #feature selection
        mod=model
        # fit the model
        mod.fit(X_train, y_train)
        # get importance
        imp = mod.feature_importances_
        return imp

    #Classifiers for feature importance
    clf=['Extra Trees','Random Forest']
    mod_feature = st.sidebar.selectbox('Which model for feature importance?', clf)

    start_time = timeit.default_timer()
    if mod_feature=='Extra Trees':
        model=etree
        importance=feature_sort(model,X_train,y_train)
    elif mod_feature=='Random Forest':
        model=rforest
        importance=feature_sort(model,X_train,y_train)
    elapsed = timeit.default_timer() - start_time
    st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))

    #Plot of feature importance
    if st.sidebar.checkbox('Show plot of feature importance'):
        plt.bar([x for x in range(len(importance))], importance)
        plt.title('Feature Importance')
        plt.xlabel('Feature (Variable Number)')
        plt.ylabel('Importance')
        st.pyplot()

    feature_imp=list(zip(features,importance))
    feature_sort=sorted(feature_imp, key = lambda x: x[1])

    n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

    top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

    if st.sidebar.checkbox('Show selected top features'):
        st.write('Top %d features in order of importance are: %s'%(n_top_features,top_features[::-1]))

    X_train_sfs=X_train[top_features]
    X_test_sfs=X_test[top_features]

    X_train_sfs_scaled=X_train_sfs
    X_test_sfs_scaled=X_test_sfs
    smt = SMOTE()
    nr = NearMiss()
    def compute_performance(model, X_train, y_train,X_test,y_test):
        start_time = timeit.default_timer()
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy').mean()
        'Accuracy: ',scores
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        cm=confusion_matrix(y_test,y_pred)
        'Confusion Matrix: ',cm
        cr=classification_report(y_test, y_pred)
        'Classification Report: ',cr
        mcc= matthews_corrcoef(y_test, y_pred)
        'Matthews Correlation Coefficient: ',mcc
        elapsed = timeit.default_timer() - start_time
        'Execution Time for performance computation: %.2f minutes'%(elapsed/60)
    #Run different classification models with rectifiers
    if st.sidebar.checkbox('Run a claims prediction model'):

        alg=['Extra Trees','Random Forest','k Nearest Neighbor','Support Vector Machine','Logistic Regression']
        classifier = st.sidebar.selectbox('Which algorithm?', alg)
        rectifier=['SMOTE','Near Miss','No Rectifier']
        imb_rect = st.sidebar.selectbox('Which imbalanced class rectifier?', rectifier)

        if classifier=='Logistic Regression':
            model=logreg
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)


        elif classifier == 'k Nearest Neighbor':
            model=knn
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

        elif classifier == 'Support Vector Machine':
            model=svm
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

        elif classifier == 'Random Forest':
            model=rforest
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

        elif classifier == 'Extra Trees':
            model=etree
            if imb_rect=='No Rectifier':
                compute_performance(model, X_train_sfs_scaled, y_train,X_test_sfs_scaled,y_test)
            elif imb_rect=='SMOTE':
                    rect=smt
                    st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                    X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                    st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                    compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)
            elif imb_rect=='Near Miss':
                rect=nr
                st.write('Shape of imbalanced y_train: ',np.bincount(y_train))
                X_train_bal, y_train_bal = rect.fit_resample(X_train_sfs_scaled, y_train)
                st.write('Shape of balanced y_train: ',np.bincount(y_train_bal))
                compute_performance(model, X_train_bal, y_train_bal,X_test_sfs_scaled,y_test)

elif app_mode == "Monte Carlo":

    def comma_format(number):
        if not pd.isna(number) and number != 0:
            return '{:,.0f}'.format(number)

    def percentage_format(number):
        if not pd.isna(number) and number != 0:
            return '{:.1%}'.format(number)

    def calculate_value_distribution(parameter_dict_1, parameter_dict_2, parameter_dict_distribution):
        parameter_list = []
        parameter_list.append(parameter_dict_1['latest revenue'])
        for i in parameter_dict_2:
            if parameter_dict_distribution[i] == 'normal':
                parameter_list.append((np.random.normal(parameter_dict_1[i], parameter_dict_2[i]))/100)
            if parameter_dict_distribution[i] == 'triangular':
                lower_bound = parameter_dict_1[i]
                mode = parameter_dict_2[i]
                parameter_list.append((np.random.triangular(lower_bound, mode, 2*mode-lower_bound))/100)
            if parameter_dict_distribution[i] == 'uniform':
                parameter_list.append((np.random.uniform(parameter_dict_1[i], parameter_dict_2[i]))/100)
        parameter_list.append(parameter_dict_1['net debt'])
        return parameter_list

    class Company:

        def __init__(self, ticker):
            self.income_statement = si.get_income_statement(ticker)
            self.balance_sheet = si.get_balance_sheet(ticker)
            self.cash_flow_statement = si.get_cash_flow(ticker)
            self.inputs = self.get_inputs_df()

        def get_inputs_df(self):
            income_statement_list = ['totalRevenue', 'ebit',
            'incomeBeforeTax', 'incomeTaxExpense'
            ]
            balance_sheet_list = ['totalCurrentAssets', 'cash',
            'totalCurrentLiabilities', 'shortLongTermDebt',
            'longTermDebt'
            ]
            balance_sheet_list_truncated = ['totalCurrentAssets', 'cash',
            'totalCurrentLiabilities', 'longTermDebt'
            ]
            balance_sheet_list_no_debt = ['totalCurrentAssets', 'cash',
            'totalCurrentLiabilities'
            ]

            cash_flow_statement_list = ['depreciation',
            'capitalExpenditures'
            ]

            income_statement_df = self.income_statement.loc[income_statement_list]
            try:
                balance_sheet_df = self.balance_sheet.loc[balance_sheet_list]
            except KeyError:
                try:
                    balance_sheet_df = self.balance_sheet.loc[balance_sheet_list_truncated]
                except KeyError:
                    balance_sheet_df = self.balance_sheet.loc[balance_sheet_list_no_debt]
            cash_flow_statement_df = self.cash_flow_statement.loc[cash_flow_statement_list]

            df = income_statement_df.append(balance_sheet_df)
            df = df.append(cash_flow_statement_df)

            columns_ts = df.columns
            columns_str = [str(i)[:10] for i in columns_ts]
            columns_dict = {}
            for i,f in zip(columns_ts, columns_str):
                columns_dict[i] = f
            df.rename(columns_dict, axis = 'columns', inplace = True)

            columns_str.reverse()
            df = df[columns_str]

            prior_revenue_list = [None]
            for i in range(len(df.loc['totalRevenue'])):
                if i != 0 and i != len(df.loc['totalRevenue']):
                    prior_revenue_list.append(df.loc['totalRevenue'][i-1])

            df.loc['priorRevenue'] = prior_revenue_list
            df.loc['revenueGrowth'] = (df.loc['totalRevenue'] - df.loc['priorRevenue']) / df.loc['priorRevenue']
            df.loc['ebitMargin'] = df.loc['ebit']/df.loc['totalRevenue']
            df.loc['taxRate'] = df.loc['incomeTaxExpense']/df.loc['incomeBeforeTax']
            df.loc['netCapexOverSales'] = (- df.loc['capitalExpenditures'] - df.loc['depreciation']) / df.loc['totalRevenue']
            try:
                df.loc['nwc'] = (df.loc['totalCurrentAssets'] - df.loc['cash']) - (df.loc['totalCurrentLiabilities'] - df.loc['shortLongTermDebt'])
            except KeyError:
                df.loc['nwc'] = (df.loc['totalCurrentAssets'] - df.loc['cash']) - (df.loc['totalCurrentLiabilities'])
            df.loc['nwcOverSales'] = df.loc['nwc']/df.loc['totalRevenue']
            try:
                df.loc['netDebt'] = df.loc['shortLongTermDebt'] + df.loc['longTermDebt'] - df.loc['cash']
            except KeyError:
                try:
                    df.loc['netDebt'] = df.loc['longTermDebt'] - df.loc['cash']
                except KeyError:
                    df.loc['netDebt'] = - df.loc['cash']
            df = df[12:len(df)].drop('nwc')
            df['Historical average'] = [df.iloc[i].mean() for i in range(len(df))]
            return df

        def get_free_cash_flow_forecast(self, parameter_list):
            df = pd.DataFrame(columns = [1, 2, 3, 4, 5])
            revenue_list = []
            for i in range(5):
                revenue_list.append(parameter_list[0] * (1 + parameter_list[1]) ** (i+1))
            df.loc['Revenues'] = revenue_list
            ebit_list = [i * parameter_list[2] for i in df.loc['Revenues']]
            df.loc['EBIT'] = ebit_list
            tax_list = [i * parameter_list[3] for i in df.loc['EBIT']]
            df.loc['Taxes'] = tax_list
            nopat_list = df.loc['EBIT'] - df.loc['Taxes']
            df.loc['NOPAT'] = nopat_list
            net_capex_list = [i * parameter_list[4] for i in df.loc['Revenues']]
            df.loc['Net capital expenditures'] = net_capex_list
            nwc_list = [i * parameter_list[5] for i in df.loc['Revenues']]
            df.loc['Changes in NWC'] = nwc_list
            free_cash_flow_list = df.loc['NOPAT'] - df.loc['Net capital expenditures'] - df.loc['Changes in NWC']
            df.loc['Free cash flow'] = free_cash_flow_list
            return df

        def discount_free_cash_flows(self, parameter_list, discount_rate, terminal_growth):
            free_cash_flow_df = self.get_free_cash_flow_forecast(parameter_list)
            df = free_cash_flow_df
            discount_factor_list = [(1 + discount_rate) ** i for i in free_cash_flow_df.columns]
            df.loc['Discount factor'] = discount_factor_list
            present_value_list = df.loc['Free cash flow'] / df.loc['Discount factor']
            df.loc['PV free cash flow'] = present_value_list
            df[0] = [0 for i in range(len(df))]
            df.loc['Sum PVs', 0] = df.loc['PV free cash flow', 1:5].sum()
            df.loc['Terminal value', 5] = df.loc['Free cash flow', 5] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            df.loc['PV terminal value', 0] = df.loc['Terminal value', 5] / df.loc['Discount factor', 5]
            df.loc['Company value (enterprise value)', 0] = df.loc['Sum PVs', 0] + df.loc['PV terminal value', 0]
            df.loc['Net debt', 0] = parameter_list[-1]
            df.loc['Equity value', 0] = df.loc['Company value (enterprise value)', 0] - df.loc['Net debt', 0]
            equity_value = df.loc['Equity value', 0]
            df = df.applymap(lambda x: comma_format(x))
            df = df.fillna('')
            column_name_list = range(6)
            df = df[column_name_list]
            return df, equity_value


    st.title('Monte Carlo Valuation')

    with st.beta_expander('How to Use'):
        st.write('This application allows you to conduct a **probabilistic** \
            valuation of companies you are interested in. Please enter the \
            **stock ticker** of your company. Subsequently, the program will \
            provide you with **historical key metrics** you can use to specify \
            key inputs required for valuing the company of your choice. \
            In addition, you need to provide a **discount rate** and a **terminal \
            growth rate** at which your company is assumed to grow after year 5 \
            into the future.')

    st.header('General company information')
    ticker_input = st.text_input('Please enter your company ticker here:')
    status_radio = st.radio('Please click Search when you are ready.', ('Entry', 'Search'))


    @st.cache
    def get_company_data():
        company = Company(ticker_input)
        return company

    if status_radio == 'Search':
        company = get_company_data()
        st.header('Key Valuation Metrics')
        st.dataframe(company.inputs)


    with st.beta_expander('Monte Carlo Simulation'):

        st.subheader('Random variables')
        st.write('When conducting a company valuation through a Monte Carlo simulation, \
            a variety of input metrics can be treated as random variables. Such \
            variables can be distributed according to different distributions. \
            Below, please specify the distribution from which the respective \
            variable values should be drawn.')

        parameter_dict_1 = {
            'latest revenue' : 0,
            'revenue growth': 0,
            'ebit margin' : 0,
            'tax rate' : 0,
            'capex ratio' : 0,
            'NWC ratio' : 0,
            'net debt' : 0
        }

        parameter_dict_2 = {
            'latest revenue' : 0,
            'revenue growth': 0,
            'ebit margin' : 0,
            'tax rate' : 0,
            'capex ratio' : 0,
            'NWC ratio' : 0
        }

        parameter_dict_distribution = {
            'latest revenue' : '',
            'revenue growth': '',
            'ebit margin' : '',
            'tax rate' : '',
            'capex ratio' : '',
            'NWC ratio' : ''
        }


        col11, col12, col13 = st.beta_columns(3)


        with col11:
            st.subheader('Revenue growth')
            radio_button_revenue_growth = st.radio('Choose growth rate distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_revenue_growth == 'Normal':
                mean_input = st.number_input('Mean revenue growth rate (in %)')
                stddev_input = st.number_input('Revenue growth rate std. dev. (in %)')
                parameter_dict_1['revenue growth'] = mean_input
                parameter_dict_2['revenue growth'] = stddev_input
                parameter_dict_distribution['revenue growth'] = 'normal'

            elif radio_button_revenue_growth == 'Triangular':
                lower_input = st.number_input('Lower end growth rate (in %)')
                mode_input = st.number_input('Mode growth rate (in %)')
                parameter_dict_1['revenue growth'] = lower_input
                parameter_dict_2['revenue growth'] = mode_input
                parameter_dict_distribution['revenue growth'] = 'triangular'

            elif radio_button_revenue_growth == 'Uniform':
                lower_input = st.number_input('Lower end growth rate (in %)')
                upper_input = st.number_input('Upper end growth rate (in %)')
                parameter_dict_1['revenue growth'] = lower_input
                parameter_dict_2['revenue growth'] = upper_input
                parameter_dict_distribution['revenue growth'] = 'uniform'


        with col12:
            st.subheader('EBIT margin')
            radio_button_ebit_margin = st.radio('Choose EBIT margin distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_ebit_margin == 'Normal':
                mean_input = st.number_input('Mean EBIT margin (in %)')
                stddev_input = st.number_input('EBIT margin std. dev. (in %)')
                parameter_dict_1['ebit margin'] = mean_input
                parameter_dict_2['ebit margin'] = stddev_input
                parameter_dict_distribution['ebit margin'] = 'normal'

            elif radio_button_ebit_margin == 'Triangular':
                lower_input = st.number_input('Lower end EBIT margin (in %)')
                mode_input = st.number_input('Mode EBIT margin (in %)')
                parameter_dict_1['ebit margin'] = lower_input
                parameter_dict_2['ebit margin'] = mode_input
                parameter_dict_distribution['ebit margin'] = 'triangular'

            elif radio_button_ebit_margin == 'Uniform':
                lower_input = st.number_input('Lower end EBIT margin (in %)')
                upper_input = st.number_input('Upper end EBIT margin (in %)')
                parameter_dict_1['ebit margin'] = lower_input
                parameter_dict_2['ebit margin'] = upper_input
                parameter_dict_distribution['ebit margin'] = 'uniform'


        with col13:
            st.subheader('Tax rate')
            radio_button_tax_rate = st.radio('Choose tax rate distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_tax_rate == 'Normal':
                mean_input = st.number_input('Mean tax rate (in %)')
                stddev_input = st.number_input('Tax rate std. dev. (in %)')
                parameter_dict_1['tax rate'] = mean_input
                parameter_dict_2['tax rate'] = stddev_input
                parameter_dict_distribution['tax rate'] = 'normal'

            elif radio_button_tax_rate == 'Triangular':
                lower_input = st.number_input('Lower end tax rate (in %)')
                mode_input = st.number_input('Mode tax rate (in %)')
                parameter_dict_1['tax rate'] = lower_input
                parameter_dict_2['tax rate'] = mode_input
                parameter_dict_distribution['tax rate'] = 'triangular'

            elif radio_button_tax_rate == 'Uniform':
                lower_input = st.number_input('Lower end tax rate (in %)')
                upper_input = st.number_input('Upper end tax rate (in %)')
                parameter_dict_1['tax rate'] = lower_input
                parameter_dict_2['tax rate'] = upper_input
                parameter_dict_distribution['tax rate'] = 'uniform'


        col21, col22, col23 = st.beta_columns(3)

        with col21:
            st.subheader('Net capex/sales')
            radio_button_tax_rate = st.radio('Choose capex ratio distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_tax_rate == 'Normal':
                mean_input = st.number_input('Mean capex ratio (in %)')
                stddev_input = st.number_input('capex ratio std. dev. (in %)')
                parameter_dict_1['capex ratio'] = mean_input
                parameter_dict_2['capex ratio'] = stddev_input
                parameter_dict_distribution['capex ratio'] = 'normal'

            elif radio_button_tax_rate == 'Triangular':
                lower_input = st.number_input('Lower end capex ratio (in %)')
                mode_input = st.number_input('Mode capex ratio (in %)')
                parameter_dict_1['capex ratio'] = lower_input
                parameter_dict_2['capex ratio'] = mode_input
                parameter_dict_distribution['capex ratio'] = 'triangular'

            elif radio_button_tax_rate == 'Uniform':
                lower_input = st.number_input('Lower end capex ratio (in %)')
                upper_input = st.number_input('Upper end capex ratio (in %)')
                parameter_dict_1['capex ratio'] = lower_input
                parameter_dict_2['capex ratio'] = upper_input
                parameter_dict_distribution['capex ratio'] = 'uniform'

        with col22:
            st.subheader('NWC/sales')
            radio_button_tax_rate = st.radio('Choose NWC ratio distribution', ('Normal', 'Triangular', 'Uniform'))

            if radio_button_tax_rate == 'Normal':
                mean_input = st.number_input('Mean NWC ratio (in %)')
                stddev_input = st.number_input('NWC ratio std. dev. (in %)')
                parameter_dict_1['NWC ratio'] = mean_input
                parameter_dict_2['NWC ratio'] = stddev_input
                parameter_dict_distribution['NWC ratio'] = 'normal'

            elif radio_button_tax_rate == 'Triangular':
                lower_input = st.number_input('Lower end NWC ratio (in %)')
                mode_input = st.number_input('Mode NWC ratio (in %)')
                parameter_dict_1['NWC ratio'] = lower_input
                parameter_dict_2['NWC ratio'] = mode_input
                parameter_dict_distribution['NWC ratio'] = 'triangular'

            elif radio_button_tax_rate == 'Uniform':
                lower_input = st.number_input('Lower end NWC ratio (in %)')
                upper_input = st.number_input('Upper end NWC ratio (in %)')
                parameter_dict_1['NWC ratio'] = lower_input
                parameter_dict_2['NWC ratio'] = upper_input
                parameter_dict_distribution['NWC ratio'] = 'uniform'

        with col23:
            st.subheader('Additional inputs')
            discount_rate = (st.number_input('Discount rate:')/100)
            terminal_growth = (st.number_input('Terminal growth rate:')/100)
            simulation_iterations = (st.number_input('Number of simulation iterations (at most 1000):'))
            inputs_radio = st.radio('Please click Search if you are ready.', ('Entry', 'Search'))

        equity_value_list = []
        revenue_list_of_lists = []
        ebit_list_of_lists = []
        if inputs_radio == 'Search':
            parameter_dict_1['latest revenue'] = company.income_statement.loc['totalRevenue', company.income_statement.columns[-1]]
            parameter_dict_1['net debt'] = company.inputs.loc['netDebt', 'Historical average']
            if simulation_iterations > 1000:
                simulation_iterations = 1000
            elif simulation_iterations < 0:
                simulation_iterations = 100
            for i in range(int(simulation_iterations)):
                model_input = calculate_value_distribution(parameter_dict_1, parameter_dict_2, parameter_dict_distribution)
                forecast_df = company.get_free_cash_flow_forecast(model_input)
                revenue_list_of_lists.append(forecast_df.loc['Revenues'])
                ebit_list_of_lists.append(forecast_df.loc['EBIT'])
                model_output, equity_value = company.discount_free_cash_flows(model_input, discount_rate, terminal_growth)
                equity_value_list.append(equity_value)

        st.header('MC Simulation Output')

        mean_equity_value = np.mean(equity_value_list)
        stddev_equity_value = np.std(equity_value_list)
        st.write('Mean equity value: $' + str(comma_format(mean_equity_value )))
        st.write('Equity value std. deviation: $' + str(comma_format(stddev_equity_value)))

        font_1 = {
            'family' : 'Arial',
                'size' : 12
        }

        font_2 = {
            'family' : 'Arial',
                'size' : 14
        }

        fig1 = plt.figure()
        plt.style.use('seaborn-whitegrid')
        plt.title(ticker_input + ' Monte Carlo Simulation', fontdict = font_1)
        plt.xlabel('Equity value (in $)', fontdict = font_1)
        plt.ylabel('Number of occurences', fontdict = font_1)
        plt.hist(equity_value_list, bins = 50, color = '#006699', edgecolor = 'black')
        st.pyplot(fig1)


        col31, col32 = st.beta_columns(2)
        with col31:
            fig2 = plt.figure()
            x = range(6)[1:6]
            plt.style.use('seaborn-whitegrid')
            plt.title('Revenue Forecast Monte Carlo Simulation', fontdict = font_2)
            plt.xticks(ticks = x)
            plt.xlabel('Year', fontdict = font_2)
            plt.ylabel('Revenue (in $)', fontdict = font_2)
            for i in revenue_list_of_lists:
                plt.plot(x, i)
            st.pyplot(fig2)

        with col32:
            fig3 = plt.figure()
            x = range(6)[1:6]
            plt.style.use('seaborn-whitegrid')
            plt.title('EBIT Forecast Monte Carlo Simulation', fontdict = font_2)
            plt.xticks(ticks = x)
            plt.xlabel('Year', fontdict = font_2)
            plt.ylabel('EBIT (in $)', fontdict = font_2)
            for i in ebit_list_of_lists:
                plt.plot(x, i)
            st.pyplot(fig3)

elif app_mode == "Prediction":

    st.title('Stock Forecast App')

    def configure_plotly_browser_state():
        import IPython
        display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
              });
            </script>
            '''))

    class Stocks:
        def __init__(self, ticker, start_date, forcast_horz):
            self.Ticker = ticker
            self.Start_Date = start_date
            self.forcast_horz = forcast_horz


        def get_stock_data(self, Ticker):

            st.write('Loading Historical Price data for ' + self.Ticker + '....')
            #self.df_Stock, self.Stock_info = self.ts.get_daily(self.Ticker, outputsize='full')
            Stock_obj = yf.Ticker(self.Ticker)
            self.df_Stock = Stock_obj.history(start=self.Start_Date)
            st.write(self.df_Stock)
            #st.write(self.Stock_info)
            #self.df_Stock = self.df_Stock.rename(columns={'1. open' : 'Open', '2. high': 'High', '3. low':'Low', '4. close': 'Close', '5. volume': 'Volume' })
            #self.df_Stock = self.df_Stock.rename_axis(['Date'])
            #sorting index
            self.Stock = self.df_Stock.sort_index(ascending=True, axis=0)
            self.Stock = self.Stock.drop(columns=['Dividends', 'Stock Splits'])
            st.write(self.Stock)
            #slicing the data for 15 years from '2004-01-02' to today
            #self.Stock = self.Stock.loc[self.Start_Date:]

            fig = self.Stock[['Close', 'High']].plot()
            plt.title("Stock Price Over time", fontsize=17)
            plt.ylabel('Price', fontsize=14)
            plt.xlabel('Time', fontsize=14)
            plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
            #plt.show()
            #st.pyplot(fig)


        def extract_Technical_Indicators(self, Ticker):

            st.write(' ')
            st.write('Feature extraction of technical Indicators....')
            #get Boolinger Bands
            self.Stock['MA_20'] = self.Stock.Close.rolling(window=20).mean()
            self.Stock['SD20'] = self.Stock.Close.rolling(window=20).std()
            self.Stock['Upper_Band'] = self.Stock.Close.rolling(window=20).mean() + (self.Stock['SD20']*2)
            self.Stock['Lower_Band'] = self.Stock.Close.rolling(window=20).mean() - (self.Stock['SD20']*2)
            st.write('Boolinger bands..')

            st.write(self.Stock.shape)
            #shifting for lagged data
            self.Stock['S_Close(t-1)'] = self.Stock.Close.shift(periods=1)
            self.Stock['S_Close(t-2)'] = self.Stock.Close.shift(periods=2)
            self.Stock['S_Close(t-3)'] = self.Stock.Close.shift(periods=3)
            self.Stock['S_Close(t-5)'] = self.Stock.Close.shift(periods=5)
            self.Stock['S_Open(t-1)'] = self.Stock.Open.shift(periods=1)
            st.write('Lagged Price data for previous days....')

            #simple moving average
            self.Stock['MA5'] = self.Stock.Close.rolling(window=5).mean()
            self.Stock['MA10'] = self.Stock.Close.rolling(window=10).mean()
            self.Stock['MA20'] = self.Stock.Close.rolling(window=20).mean()
            self.Stock['MA50'] = self.Stock.Close.rolling(window=50).mean()
            self.Stock['MA200'] = self.Stock.Close.rolling(window=200).mean()
            st.write('Simple Moving Average....')

            #Exponential Moving Averages
            self.Stock['EMA10'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
            self.Stock['EMA20'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
            self.Stock['EMA50'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
            self.Stock['EMA100'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
            self.Stock['EMA200'] = self.Stock.Close.ewm(span=5, adjust=False).mean().fillna(0)
            st.write('Exponential Moving Average....')

            #Moving Average Convergance Divergances
            self.Stock['EMA_12'] = self.Stock.Close.ewm(span=12, adjust=False).mean()
            self.Stock['EMA_26'] = self.Stock.Close.ewm(span=26, adjust=False).mean()
            self.Stock['MACD'] = self.Stock['EMA_12'] - self.Stock['EMA_26']

            self.Stock['MACD_EMA'] = self.Stock.MACD.ewm(span=9, adjust=False).mean()

            #Average True Range
            #self.Stock['ATR'] = talib.ATR(self.Stock['High'].values, self.Stock['Low'].values, self.Stock['Close'].values, timeperiod=14)

            #Average Directional Index
            #self.Stock['ADX'] = talib.ADX(self.Stock['High'], self.Stock['Low'], self.Stock['Close'], timeperiod=14)

            #Commodity Channel index
            tp = (self.Stock['High'] + self.Stock['Low'] + self.Stock['Close']) /3
            ma = tp/20
            md = (tp-ma)/20
            self.Stock['CCI'] = (tp-ma)/(0.015 * md)
            st.write('Commodity Channel Index....')

            #Rate of Change
            self.Stock['ROC'] = ((self.Stock['Close'] - self.Stock['Close'].shift(10)) / (self.Stock['Close'].shift(10)))*100

            #Relative Strength Index
            #self.Stock['RSI'] = talib.RSI(self.Stock.Close.values, timeperiod=14)

            #William's %R
            #self.Stock['William%R'] = talib.WILLR(self.Stock.High.values, self.Stock.Low.values, self.Stock.Close.values, 14)

            #Stocastic K
            self.Stock['SO%K'] = ((self.Stock.Close - self.Stock.Low.rolling(window=14).min()) / (self.Stock.High.rolling(window=14).max() - self.Stock.Low.rolling(window=14).min())) * 100
            st.write('Stocastic %K ....')
            #Standard Deviation of last 5 day returns
            self.Stock['per_change'] = self.Stock.Close.pct_change()
            self.Stock['STD5'] = self.Stock.per_change.rolling(window=5).std()

            #Force Index
            self.Stock['ForceIndex1'] = self.Stock.Close.diff(1) * self.Stock.Volume
            self.Stock['ForceIndex20'] = self.Stock.Close.diff(20) * self.Stock.Volume
            st.write('Force index....')

            #st.write('Stock Data ', self.Stock)

            self.Stock[['Close', 'MA_20', 'Upper_Band', 'Lower_Band']].plot(figsize=(12,6))
            plt.title('20 Day Bollinger Band')
            plt.ylabel('Price (USD)')
            plt.show();
            #st.pyplot(fig1)

            self.Stock[['Close', 'MA20', 'MA200', 'MA50']].plot()
            plt.show();

            self.Stock[['MACD', 'MACD_EMA']].plot()
            plt.show();
            #st.pyplot(fig2)
            #Dropping unneccesary columns
            self.Stock = self.Stock.drop(columns=['MA_20', 'per_change', 'EMA_12', 'EMA_26'])
            st.write(self.Stock.shape)


        def extract_info(self, date_val):

            Day = date_val.day
            DayofWeek = date_val.dayofweek
            Dayofyear = date_val.dayofyear
            Week = date_val.week
            Is_month_end = date_val.is_month_end.real
            Is_month_start = date_val.is_month_start.real
            Is_quarter_end = date_val.is_quarter_end.real
            Is_quarter_start = date_val.is_quarter_start.real
            Is_year_end = date_val.is_year_end.real
            Is_year_start = date_val.is_year_start.real
            Is_leap_year = date_val.is_leap_year.real
            Year = date_val.year
            Month = date_val.month

            return Day, DayofWeek, Dayofyear, Week, Is_month_end, Is_month_start, Is_quarter_end, Is_quarter_start, Is_year_end, Is_year_start, Is_leap_year, Year, Month


        def extract_date_features(self, Ticker):
            st.write(' ')

            self.Stock['Date_col'] = self.Stock.index

            self.Stock[['Day', 'DayofWeek', 'DayofYear', 'Week', 'Is_month_end', 'Is_month_start',
              'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start', 'Is_leap_year', 'Year', 'Month']] = self.Stock.Date_col.apply(lambda date_val: pd.Series(self.extract_info(date_val)))
            st.write('Extracting information from dates....')
            st.write(self.Stock.shape)


        def get_IDXFunds_features(self, Ticker):
            st.write(' ')
            st.write('Fetching data for NASDAQ-100 Index Fund ETF QQQ & S&P 500 index ......')
            st.write(self.Stock.shape)
            # Nasdaq-100 Index Fund ETF QQQ
            #QQQ, QQQ_info = self.ts.get_daily('QQQ', outputsize='full')
            #QQQ = QQQ.rename(columns={'1. open' : 'Open', '2. high': 'High', '3. low':'Low', '4. close': 'QQQ_Close', '5. volume': 'Volume' })
            #QQQ = QQQ.rename_axis(['Date'])
            Stock_obj = yf.Ticker('QQQ')
            QQQ = Stock_obj.history(start=self.Start_Date)
            QQQ = QQQ.rename(columns={'Close': 'QQQ_Close'})
            QQQ = QQQ.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])
            #sorting index
            QQQ = QQQ.sort_index(ascending=True, axis=0)
            #slicing the data for 15 years from '2004-01-02' to today
            #QQQ = QQQ.loc[self.Start_Date:]
            QQQ['QQQ(t-1)'] = QQQ.QQQ_Close.shift(periods=1)
            QQQ['QQQ(t-2)'] =  QQQ.QQQ_Close.shift(periods=2)
            QQQ['QQQ(t-5)'] =  QQQ.QQQ_Close.shift(periods=5)

            QQQ['QQQ_MA10'] = QQQ.QQQ_Close.rolling(window=10).mean()
            #QQQ['QQQ_MA10_t'] = QQQ.QQQ_ClosePrev1.rolling(window=10).mean()
            QQQ['QQQ_MA20'] = QQQ.QQQ_Close.rolling(window=20).mean()
            QQQ['QQQ_MA50'] = QQQ.QQQ_Close.rolling(window=50).mean()
            st.write(QQQ.shape)



            #S&P 500 Index Fund
            #SnP, SnP_info = self.ts.get_daily('INX', outputsize='full')
            #SnP = SnP.rename(columns={'1. open' : 'Open', '2. high': 'High', '3. low':'Low', '4. close': 'SnP_Close', '5. volume': 'Volume' })
            #SnP = SnP.rename_axis(['Date'])
            #SnP = SnP.drop(columns=['Open', 'High', 'Low', 'Volume'])

            Stock_obj = yf.Ticker('^GSPC')
            SnP = Stock_obj.history(start=self.Start_Date)
            SnP = SnP.rename(columns={'Close': 'SnP_Close'})
            SnP = SnP.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])

            #sorting index
            SnP = SnP.sort_index(ascending=True, axis=0)
            #slicing the data for 15 years from '2004-01-02' to today
            #SnP = SnP.loc[self.Start_Date:]
            SnP
            SnP['SnP(t-1))'] = SnP.SnP_Close.shift(periods=1)
            SnP['SnP(t-5)'] =  SnP.SnP_Close.shift(periods=5)
            st.write(SnP.shape)

            #S&P 500 Index Fund
            #DJIA, DJIA_info = self.ts.get_daily('DJI', outputsize='full')
            #DJIA = DJIA.rename(columns={'1. open' : 'Open', '2. high': 'High', '3. low':'Low', '4. close': 'DJIA_Close', '5. volume': 'Volume' })
            #DJIA = DJIA.rename_axis(['Date'])
            #DJIA = DJIA.drop(columns=['Open', 'High', 'Low', 'Volume'])

            Stock_obj = yf.Ticker('^DJI')
            DJIA = Stock_obj.history(start=self.Start_Date)
            DJIA = DJIA.rename(columns={'Close': 'DJIA_Close'})
            DJIA = DJIA.drop(columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'])


            #sorting index
            DJIA = DJIA.sort_index(ascending=True, axis=0)
            #slicing the data for 15 years from '2004-01-02' to today
            #DJIA = DJIA.loc[self.Start_Date:]
            DJIA
            DJIA['DJIA(t-1))'] = DJIA.DJIA_Close.shift(periods=1)
            DJIA['DJIA(t-5)'] =  DJIA.DJIA_Close.shift(periods=5)
            st.write(DJIA.shape)
            st.write(self.Stock.shape)

            #Merge index funds
            IDXFunds = QQQ.merge(SnP, left_index=True, right_index=True)
            IDXFunds = IDXFunds.merge(DJIA, left_index=True, right_index=True)
            self.Stock = self.Stock.merge(IDXFunds, left_index=True, right_index=True)
            st.write(self.Stock.shape)

        def forcast_Horizon(self, Ticker):

            st.write(' ')
            st.write('Adding the future day close price as a target column for Forcast Horizon of ' + str(self.forcast_horz))
            #Adding the future day close price as a target column which needs to be predicted using Supervised Machine learning models
            self.Stock['Close_forcast'] = self.Stock.Close.shift(-self.forcast_horz)
            self.Stock = self.Stock.rename(columns={'Close': 'Close(t)'})
            self.Stock = self.Stock.dropna()
            st.write(self.Stock.shape)


        def save_features(self, Ticker):
            st.write('Saving extracted features data in S3 Bucket....')
            self.Stock.to_csv(self.Ticker + '.csv')
            st.write('Extracted features shape - '+ str(self.Stock.shape))
            st.write(' ')
            st.write('Extracted features dataframe - ')
            st.write(self.Stock)
            return self.Stock


        T = TypeVar('T')

        def pipeline(self,
            value: T,
            function_pipeline: Sequence[Callable[[T], T]],
            ) -> T:

            return reduce(lambda v, f: f(v), function_pipeline, value)

        def pipeline_sequence(self):

            st.write('Initiating Pipeline....')
            z = self.pipeline(
                value=self.Ticker,
                function_pipeline=(
                    self.get_stock_data,
                    self.extract_Technical_Indicators,
                    self.extract_date_features,
                    self.get_IDXFunds_features,
                    self.forcast_Horizon,
                    self.save_features
                        )
                    )

            st.write(f'z={z}')

    class Stock_Prediction_Modeling():
        def __init__(self, Stocks, models, features):
            self.Stocks = Stocks
            self.train_Models = models
            self.metrics = {}
            self.features_selected = features


        def get_stock_data(self, Ticker):

            file = self.Ticker + '.csv'
            Stock = pd.read_csv(file,  index_col=0)
            st.write(Stock)
            #st.write(self.features_selected)
            st.write('Loading Historical Price data for ' + self.Ticker + '....')

            self.df_Stock = Stock.copy() #[features_selected]
            #self.df_Stock = self.df_Stock.drop(columns=['Date_col'])
            self.df_Stock = self.df_Stock[self.features_selected]

            self.df_Stock = self.df_Stock.rename(columns={'Close(t)':'Close'})

            #self.df_Stock = self.df_Stock.copy()
            self.df_Stock['Diff'] = self.df_Stock['Close'] - self.df_Stock['Open']
            self.df_Stock['High-low'] = self.df_Stock['High'] - self.df_Stock['Low']

            #st.write('aaaa')
            st.write('Training Selected Machine Learning models for ', self.Ticker)
            #features_selected = ['Close', 'Diff', 'High-low', 'QQQ_Close', 'SnP_Close','DJIA_Close', 'ATR', 'RSI', 'MA50', 'EMA200', 'Upper_Band']
            st.markdown('Your **_final_ _dataframe_ _for_ Training** ')
            st.write(self.df_Stock)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.success('Training Completed!')

            #self.df_Stock = self.df_Stock[:-70]

            st.write(self.df_Stock.columns)


        def prepare_lagged_features(self, lag_stock, lag_index, lag_diff):

            st.write('Preparing Lagged Features for Stock, Index Funds.....')
            lags = range(1, lag_stock+1)
            lag_cols= ['Close']
            self.df_Stock=self.df_Stock.assign(**{
                '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
                for l in lags
                for col in lag_cols
            })


            lags = range(1, lag_index+1)
            lag_cols= ['QQQ_Close','SnP_Close','DJIA_Close']
            self.df_Stock= self.df_Stock.assign(**{
                '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
                for l in lags
                for col in lag_cols
            })

            self.df_Stock = self.df_Stock.drop(columns=lag_cols)


            lags = range(1, lag_diff+1)
            lag_cols= ['Diff','High-low']
            self.df_Stock= self.df_Stock.assign(**{
                '{}(t-{})'.format(col, l): self.df_Stock[col].shift(l)
                for l in lags
                for col in lag_cols
            })

            self.df_Stock = self.df_Stock.drop(columns=lag_cols)

            remove_lags_na = max(lag_stock, lag_index, lag_diff) + 1
            st.write('Removing NAN rows - ', str(remove_lags_na))
            self.df_Stock = self.df_Stock.iloc[remove_lags_na:,]
            return self.df_Stock

        def get_lagged_features(self, Ticker):

            self.df_Stock_lagged = self.prepare_lagged_features(lag_stock = 20, lag_index = 10, lag_diff = 5)

            st.write(self.df_Stock_lagged.columns)

            self.df_Stock = self.df_Stock_lagged.copy()
            st.write(self.df_Stock.shape)
            st.write('Extracted Feature Columns after lagged effect - ')
            st.write(self.df_Stock.columns)

            '''
            self.df_Stock['Close'].plot(figsize=(10, 7))
            plt.title("Stock Price", fontsize=17)
            plt.ylabel('Price', fontsize=14)
            plt.xlabel('Time', fontsize=14)
            plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
            plt.show()
            st.write(self.df_Stock)
            '''

        def create_train_test_set(self):

            #self.df_Stock = self.df_Stock[:-60]
            self.features = self.df_Stock.drop(columns=['Close'], axis=1)
            self.target = self.df_Stock['Close']


            data_len = self.df_Stock.shape[0]
            st.write('Historical Stock Data length is - ', str(data_len))

            #create a chronological split for train and testing
            train_split = int(data_len * 0.9)
            st.write('Training Set length - ', str(train_split))

            val_split = train_split + int(data_len * 0.08)
            st.write('Validation Set length - ', str(int(data_len * 0.1)))

            st.write('Test Set length - ', str(int(data_len * 0.02)))

            # Splitting features and target into train, validation and test samples
            X_train, X_val, X_test = self.features[:train_split], self.features[train_split:val_split], self.features[val_split:]
            Y_train, Y_val, Y_test = self.target[:train_split], self.target[train_split:val_split], self.target[val_split:]

            #st.write shape of samples
            st.write(X_train.shape, X_val.shape, X_test.shape)
            st.write(Y_train.shape, Y_val.shape, Y_test.shape)

            return X_train, X_val, X_test, Y_train, Y_val, Y_test

        def get_train_test(self):
            st.write('Splitting the data into Train and Test ...')
            st.write(' ')
            if self.ML_Model == 'LSTM':
                self.scale_LSTM_features()
                self.X_train, self.X_test, self.Y_train, self.Y_test = self.create_train_test_LSTM()
            else:
                self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = self.create_train_test_set()
                #st.write('here6')

        def get_mape(self, y_true, y_pred):
            """
            Compute mean absolute percentage error (MAPE)
            """
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


        def calc_metrics(self):
            from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
            st.write('Evaluating Metrics - MAE, MAPE, RMSE, R Square')
            st.write(' ')
            if self.ML_Model == 'LSTM':

                self.Train_RSq = round(r2_score(self.Y_train,self.Y_train_pred),2)
                self.Train_EV = round(explained_variance_score(self.Y_train,self.Y_train_pred),2)
                self.Train_MAPE = round(self.get_mape(self.Y_train,self.Y_train_pred), 2)
                self.Train_MSE = round(mean_squared_error(self.Y_train,self.Y_train_pred), 2)
                self.Train_RMSE = round(np.sqrt(mean_squared_error(self.Y_train,self.Y_train_pred)),2)
                self.Train_MAE = round(mean_absolute_error(self.Y_train,self.Y_train_pred),2)


                self.Test_RSq = round(r2_score(self.Y_test,self.Y_test_pred),2)
                self.Test_EV = round(explained_variance_score(self.Y_test,self.Y_test_pred),2)
                self.Test_MAPE = round(self.get_mape(self.Y_test,self.Y_test_pred), 2)
                self.Test_MSE = round(mean_squared_error(self.Y_test,self.Y_test_pred), 2)
                self.Test_RMSE = round(np.sqrt(mean_squared_error(self.Y_test,self.Y_test_pred)),2)
                self.Test_MAE = round(mean_absolute_error(self.Y_test,self.Y_test_pred),2)
            else:
                #st.write('here6')
                self.Train_RSq = round(r2_score(self.Y_train,self.Y_train_pred),2)
                self.Train_EV = round(explained_variance_score(self.Y_train,self.Y_train_pred),2)
                self.Train_MAPE = round(self.get_mape(self.Y_train,self.Y_train_pred), 2)
                self.Train_MSE = round(mean_squared_error(self.Y_train,self.Y_train_pred), 2)
                self.Train_RMSE = round(np.sqrt(mean_squared_error(self.Y_train,self.Y_train_pred)),2)
                self.Train_MAE = round(mean_absolute_error(self.Y_train,self.Y_train_pred),2)

                self.Val_RSq = round(r2_score(self.Y_val,self.Y_val_pred),2)
                self.Val_EV = round(explained_variance_score(self.Y_val,self.Y_val_pred),2)
                self.Val_MAPE = round(self.get_mape(self.Y_val,self.Y_val_pred), 2)
                self.Val_MSE = round(mean_squared_error(self.Y_train,self.Y_train_pred), 2)
                self.Val_RMSE = round(np.sqrt(mean_squared_error(self.Y_val,self.Y_val_pred)),2)
                self.Val_MAE = round(mean_absolute_error(self.Y_val,self.Y_val_pred),2)

                self.Test_RSq = round(r2_score(self.Y_test,self.Y_test_pred),2)
                self.Test_EV = round(explained_variance_score(self.Y_test,self.Y_test_pred),2)
                self.Test_MAPE = round(self.get_mape(self.Y_test,self.Y_test_pred), 2)
                self.Test_MSE = round(mean_squared_error(self.Y_test,self.Y_test_pred), 2)
                self.Test_RMSE = round(np.sqrt(mean_squared_error(self.Y_test,self.Y_test_pred)),2)
                self.Test_MAE = round(mean_absolute_error(self.Y_test,self.Y_test_pred),2)

        def update_metrics_tracker(self):
            st.write('Updating the metrics tracker....')
            if self.ML_Model == 'LSTM':
                #self.metrics[self.Ticker] = {}
                self.metrics[self.Ticker][self.ML_Model] = {'Train_MAE': self.Train_MAE, 'Train_MAPE': self.Train_MAPE , 'Train_RMSE': self.Train_RMSE,
                              'Test_MAE': self.Test_MAE, 'Test_MAPE': self.Test_MAPE, 'Test_RMSE': self.Test_RMSE}
            else:
                ##self.metrics[self.Ticker] = {{}}
                self.metrics[self.Ticker][self.ML_Model] = {'Train_MAE': self.Train_MAE, 'Train_MAPE': self.Train_MAPE , 'Train_RMSE': self.Train_RMSE,
                              'Test_MAE': self.Val_MAE, 'Test_MAPE': self.Val_MAPE, 'Test_RMSE': self.Val_RMSE}



        def train_model(self, Ticker):

            for model in self.train_Models:
                self.ML_Model = model
                if self.ML_Model == 'Linear Regression':

                    st.write(' ')
                    st.write('Training Linear Regression Model')

                    self.get_train_test()


                    from sklearn.linear_model import LinearRegression
                    lr = LinearRegression()
                    lr.fit(self.X_train, self.Y_train)
                    st.write('LR Coefficients: \n', lr.coef_)
                    st.write('LR Intercept: \n', lr.intercept_)

                    st.write("Performance (R^2): ", lr.score(self.X_train, self.Y_train))

                    self.Y_train_pred = lr.predict(self.X_train)
                    self.Y_val_pred = lr.predict(self.X_val)
                    self.Y_test_pred = lr.predict(self.X_test)

                    self.calc_metrics()
                    self.update_metrics_tracker()
                    self.plot_prediction()

                elif self.ML_Model == 'XGBoost':
                    st.write(' ')
                    st.write('Training XGBoost Model')

                    self.get_train_test()

                    from xgboost import XGBRegressor
                    n_estimators = 100             # Number of boosted trees to fit. default = 100
                    max_depth = 10                 # Maximum tree depth for base learners. default = 3
                    learning_rate = 0.2            # Boosting learning rate (xgb’s “eta”). default = 0.1
                    min_child_weight = 1           # Minimum sum of instance weight(hessian) needed in a child. default = 1
                    subsample = 1                  # Subsample ratio of the training instance. default = 1
                    colsample_bytree = 1           # Subsample ratio of columns when constructing each tree. default = 1
                    colsample_bylevel = 1          # Subsample ratio of columns for each split, in each level. default = 1
                    gamma = 2                      # Minimum loss reduction required to make a further partition on a leaf node of the tree. default=0

                    model_seed = 42



                    xgb = XGBRegressor(seed=model_seed,
                                             n_estimators=n_estimators,
                                             max_depth=max_depth,
                                             learning_rate=learning_rate,
                                             min_child_weight=min_child_weight,
                                             subsample=subsample,
                                             colsample_bytree=colsample_bytree,
                                             colsample_bylevel=colsample_bylevel,
                                             gamma=gamma)
                    xgb.fit(self.X_train, self.Y_train)

                    self.Y_train_pred = xgb.predict(self.X_train)
                    self.Y_val_pred = xgb.predict(self.X_val)
                    self.Y_test_pred = xgb.predict(self.X_test)

                    self.calc_metrics()
                    self.update_metrics_tracker()

                    fig = plt.figure(figsize=(8,8))
                    plt.xticks(rotation='vertical')
                    plt.bar([i for i in range(len(xgb.feature_importances_))], xgb.feature_importances_.tolist(), tick_label=self.X_test.columns)
                    plt.title('Feature importance of the technical indicators.')
                    plt.show()

                    self.plot_prediction()

                elif self.ML_Model == 'Random Forest':
                    st.write(' ')
                    st.write('Training Random Forest Model')

                    self.get_train_test()
                    rf = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=42)
                    rf.fit(self.X_train, self.Y_train)

                    self.Y_train_pred = rf.predict(self.X_train)
                    self.Y_val_pred = rf.predict(self.X_val)
                    self.Y_test_pred = rf.predict(self.X_test)

                    self.calc_metrics()
                    self.update_metrics_tracker()
                    self.plot_prediction()


        def plot_prediction(self):

            st.write(' ')
            st.write('Predicted vs Actual for ', self.ML_Model)
            st.write('Predicted vs Actual for ', self.ML_Model)
            self.df_pred = pd.DataFrame(self.Y_val.values, columns=['Actual'], index=self.Y_val.index)
            self.df_pred['Predicted'] = self.Y_val_pred
            self.df_pred = self.df_pred.reset_index()
            self.df_pred.loc[:, 'Date'] = pd.to_datetime(self.df_pred['Date'],format='%Y-%m-%d')
            st.write('Stock Prediction on Test Data - ',self.df_pred)
            st.write('Stock Prediction on Test Data for - ',self.Ticker)
            st.write(self.df_pred)

            st.write('Plotting Actual vs Predicted for - ', self.ML_Model)
            fig = self.df_pred[['Actual', 'Predicted']].plot()
            plt.title('Actual vs Predicted Stock Prices')

            st.pyplot()

        def save_results(self, Ticker):
            import json
            st.write('Saving Metrics in Json for Stock - ', self.Ticker)
            with open('./metrics.txt', 'w') as json_file:
                json.dump(self.metrics, json_file)


        def pipeline(self,
            value: T,
            function_pipeline: Sequence[Callable[[T], T]],
            ) -> T:

            return reduce(lambda v, f: f(v), function_pipeline, value)

        def pipeline_sequence(self):
            for stock in self.Stocks:
                self.Ticker = stock
                self.metrics[self.Ticker] = {}
                st.write('Initiating Pipeline for Stock Ticker ---- ', self.Ticker)
                z = self.pipeline(
                    value=self.Ticker,
                    function_pipeline=(
                        self.get_stock_data,
                        self.get_lagged_features,
                        self.train_model,
                        self.save_results
                            )
                        )

                st.write(f'z={z}')

    def stock_financials(stock):
        df_ticker = yf.Ticker(stock)
        sector = df_ticker.info['sector']
        prevClose = df_ticker.info['previousClose']
        marketCap = df_ticker.info['marketCap']
        twoHunDayAvg = df_ticker.info['twoHundredDayAverage']
        fiftyTwoWeekHigh = df_ticker.info['fiftyTwoWeekHigh']
        fiftyTwoWeekLow = df_ticker.info['fiftyTwoWeekLow']
        Name = df_ticker.info['longName']
        averageVolume = df_ticker.info['averageVolume']
        shortRatio = df_ticker.info['shortRatio']
        ftWeekChange = df_ticker.info['52WeekChange']
        website = df_ticker.info['website']


        st.write('Company Name -', Name)
        st.write('Sector -', sector)
        st.write('Company Website -', website)
        st.write('Average Volume -', averageVolume)
        st.write('Market Cap -', marketCap)
        st.write('Previous Close -', prevClose)
        st.write('52 Week Change -', ftWeekChange)
        st.write('52 Week High -', fiftyTwoWeekHigh)
        st.write('52 Week Low -', fiftyTwoWeekLow)
        st.write('200 Day Average -', twoHunDayAvg)
        st.write('Short Ratio -', shortRatio)


    def plot_time_series(stock):
        df_ticker = yf.Ticker(stock)
        data = df_ticker.history()
        data = data.sort_index(ascending=True, axis=0)
        data['Date'] = data.index

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.Date, y=data['Open'], name="stock_open",line_color='crimson'))
        fig.add_trace(go.Scatter(x=data.Date, y=data['Close'], name="stock_close",line_color='dimgray'))
        fig.add_trace(go.Scatter(x=data.Date, y=data['High'], name="stock_high",line_color='blueviolet'))
        fig.add_trace(go.Scatter(x=data.Date, y=data['Low'], name="stock_low",line_color='darksalmon'))

        fig.layout.update(title_text='Stock Price with Rangeslider',xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    df_stock = pd.DataFrame()
    eval_metrics = {}

    st.title("Stock Prediction")

    st.markdown("""
    <style>
    body {
        color: #fff;
        background-color: #0A3648;

    }
    </style>
        """, unsafe_allow_html=True)
    #0A3648
    #13393E
    menu=["Stocks Exploration & Feature Extraction", "Machine Learning Models"]
    choices = st.sidebar.selectbox("Select Dashboard",menu)



    if choices == 'Stocks Exploration & Feature Extraction':
        st.subheader('Stock Exploration & Feature extraction')

        st.write('Feature Extraction is a tedious job to do more so when we are talking about stocks. We have \
                     created this Pipeline to extract many Technical Indicators as well as create lagged features \
                     for training a Machine Learning algorithm for forcasting Stock Prices.')
        user_input = ''
        st.markdown('Enter **_Ticker_ Symbol** for the **Stock**')
        user_input = st.text_input("", '')

        if not user_input:
                pass
        else:

            st.markdown('Select from the options below to Explore Stocks')

            selected_explore = st.selectbox("", options=['Select your Option', 'Stock Financials Exploration', 'Extract Features for Stock Price Forecasting'], index=0)
            if selected_explore == 'Stock Financials Exploration':
                st.markdown('')
                st.markdown('**_Stock_ Financial** Information')
                st.markdown('')
                st.markdown('')
                stock_financials(user_input)
                plot_time_series(user_input)


            elif selected_explore == 'Extract Features for Stock Price Forecasting':


                st.markdown('**_Real-Time_ _Feature_ Extraction** for any Stocks')

                st.write('Select a Date from a minimum of a year before as some of the features we extract uses upto 200 days of data. ')
                st.markdown('Select **_Start_ _Date_ _for_ _Historical_ Stock** Data & features')
                start_date = st.date_input(
                "", datetime(2015, 5, 4))
                st.write('You selected data from -', start_date)

                submit = st.button('Extract Features')
                if submit:
                    try:

                        with st.spinner('Extracting Features... '):
                            time.sleep(2)
                        st.write('Date - ', start_date)
                        features = Stocks(user_input, start_date, 1)
                        features.pipeline_sequence()

                    except:
                        st.markdown('If you wants to make money, your **_Ticker_ symbol** should be correct!!! :p ')
                    file_name = user_input + '.csv'
                    df_stock = pd.read_csv(file_name)
                    st.write('Extracted Features Dataframe for ', user_input)
                    st.write(df_stock)
                    #st.write('Download Link')

                    st.write('We have extracted', len(df_stock.columns), 'columns for this stock. You can Analyse it or even train it for Stock Prediction.')


                    st.write('Extracted Feature Columns are', df_stock.columns)

    elif choices == 'Machine Learning Models':
        st.subheader('Train Machine Learning Models for Stock Prediction & Generate your own Buy/Sell Signals using the best Model')


        st.markdown('**_Real_ _Time_ ML Training** for any Stocks')

        st.write('Make sure you have Extracted features for the Stocks you want to train models on using first Tab')

        result = glob.glob( '*.csv' )
        #st.write( result )
        stock = []
        for val in result:
            stock.append(val.split('.')[0])

        st.markdown('**_Recently_ _Extracted_ Stocks** -')
        st.write(stock[:5])
        cols1 = ('NKE', 'JNJ')
        st.markdown('**_Select_ _Stocks_ _to_ Train**')
        Stocks = st.multiselect("Choose", stock, default=(cols1))

        options = ('Linear Regression', 'Random Forest', 'XGBoost')
        cols2 = ('Linear Regression', 'Random Forest')
        st.markdown('**_Select_ _Machine_ _Learning_ Algorithms** to Train')
        models = st.multiselect("Choose", options, default=cols2)


        file = './' + stock[0] + '.csv'
        df_stock = pd.read_csv(file)
        df_stock = df_stock.drop(columns=['Date', 'Date_col'])
        #st.write(df_stock.columns)
        st.markdown('Select from your **_Extracted_ features** or use default')
        st.write('Select all Extracted features')
        all_features = st.checkbox('Select all Extracted features')
        cols = ['Open', 'High', 'Low', 'Close(t)', 'Upper_Band', 'MA200', 'ATR', 'ROC', 'QQQ_Close', 'SnP_Close', 'DJIA_Close', 'DJIA(t-5)']
        if all_features:
            cols = df_stock.columns.tolist()
            cols.pop(len(df_stock.columns)-1)

        features = st.multiselect("", df_stock.columns.tolist(), default=df_stock.columns.tolist())

        submit = st.button('Train')
        import json
        if submit:
            try:
                training = Stock_Prediction_Modeling(Stocks, models, features)
                training.pipeline_sequence()
                with open('./metrics.txt') as f:
                    eval_metrics = json.load(f)

            except:
                st.markdown('There seems to be a error - **_check_ logs**')
                st.write("Unexpected error:", sys.exc_info())
                st.write()

            Metrics = pd.DataFrame.from_dict({(i,j): eval_metrics[i][j]
                                   for i in eval_metrics.keys()
                                   for j in eval_metrics[i].keys()},
                               orient='index')

            st.write(Metrics)

    elif choices == 'LSTM':
        st.subheader('Look Into The Future to Predict Stock Prices for Any Stocks and Generate Buy/Sell Signals')

elif app_mode == "NLP_Analysis":

    st.sidebar.title("NLP News Sentimental Analysis")

    def load_data():
        data = pd.read_csv(
            './data/NLP/Combined_DJIA.csv')

        data['Top23'].fillna(data['Top23'].median, inplace=True)
        data['Top24'].fillna(data['Top24'].median, inplace=True)
        data['Top25'].fillna(data['Top25'].median, inplace=True)

        return data

    def create_df(dataset):

        dataset = dataset.drop(columns=['Date', 'Label'])
        dataset.replace("[^a-zA-Z]", " ", regex=True, inplace=True)
        for col in dataset.columns:
            dataset[col] = dataset[col].str.lower()

        headlines = []
        for row in range(0, len(dataset.index)):
            headlines.append(' '.join(str(x) for x in dataset.iloc[row, 0:25]))

        dataset = load_data()

        df = pd.DataFrame(headlines, columns=['headlines'])
        df['label'] = dataset.Label
        df['date'] = dataset.Date

        return df

    def tokenize(text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for token in tokens:
            clean_token = lemmatizer.lemmatize(token).lower().strip()
            clean_tokens.append(clean_token)

        return clean_tokens

    def split(df):

        train = df[df['date'] < '20150101']
        test = df[df['date'] > '20141231']
        x_train = train.headlines
        y_train = train.label
        x_test = test.headlines
        y_test = test.label

        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):

        if 'Confusion Matrix' in metrics_list:
            st.subheader('Confusion Matrix')
            predictions = model.predict(x_test)
            matrix = confusion_matrix(y_test, predictions)
            st.write("Confusion Matrix ", matrix)

        if 'Classification_Report' in metrics_list:
            st.subheader('Classification_Report')
            predictions = model.predict(x_test)
            report = classification_report(y_test, predictions)
            st.write("Classification_Report ", report)

        if 'Accuracy_Score' in metrics_list:
            st.subheader('Accuracy_Score')
            predictions = model.predict(x_test)
            score = accuracy_score(y_test, predictions)
            st.write("Accuracy_Score: ", score.round(2))

    def Vectorize():

        pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
            ('tfidf', TfidfTransformer())
        ])
        return pipeline

    df = load_data()
    df = create_df(df)
    x_train, x_test, y_train, y_test = split(df)
    vector = Vectorize()

    if st.sidebar.checkbox("show raw data", False):
        st.subheader("Top 25 Headline News from Reddit")
        st.write(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox(
        "Classifier", ("Random Forest Classifier", "Logistic Regression"))

    if classifier == "Random Forest Classifier":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input(
            "n_estimators", 50, 300, step=50, key='n_estimators')

        metrics = st.sidebar.multiselect(
            "what metrics to plot?", ("Confusion Matrix", "Classification_Report", "Accuracy_Score"))

        if st.sidebar.button("Classify", key="classify"):

            st.subheader("Random Forest Classifier")
            x_train = vector.fit_transform(x_train)
            x_test = vector.transform(x_test)
            model = RandomForestClassifier(
                n_estimators=n_estimators, criterion='entropy')
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            matrix = confusion_matrix(y_test, predictions)
            score = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            # st.write("Accuracy_Score: ", score.round(2))
            # st.write("Classification_Report ", report)
            # st.write("Confusion Matrix ", matrix)
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input(
            "C", 1, 1000, step=1, key='classify')

        metrics = st.sidebar.multiselect(
            "what metrics to plot?", ("Confusion Matrix", "Classification_Report", "Accuracy_Score"))

        if st.sidebar.button("Classify", key="classify"):

            st.subheader("Logistic Regression")
            x_train = vector.fit_transform(x_train)
            x_test = vector.transform(x_test)
            model = LogisticRegression(C=C)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            matrix = confusion_matrix(y_test, predictions)
            score = accuracy_score(y_test, predictions)
            report = classification_report(y_test, predictions)
            # st.write("Accuracy_Score: ", score.round(2))
            # st.write("Classification_Report ", report)
            # st.write("Confusion Matrix ", matrix)
            plot_metrics(metrics)

if app_mode == 'Customer Clustering':

    uploaded_file = './data/ga_customers_clustered.csv'

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.markdown('### Data Sample')
        st.write(data.head())
        id_col = st.sidebar.selectbox('Pick your ID column', options=data.columns)
        cat_features = st.sidebar.multiselect('Pick your categorical features', options=[c for c in data.columns], default = [v for v in data.select_dtypes(exclude=[int, float]).columns.values if v != id_col])
        clusters = data['cluster']
        df_p = data.drop(id_col, axis=1)
        if cat_features:
            df_p = pd.get_dummies(df_p, columns=cat_features) #OHE the categorical features
        prof = st.checkbox('Check to profile the clusters')

    else:
        st.markdown("""
        <h3 > Data Science for Marketing </h3>

        """, unsafe_allow_html=True)

    if (prof == True) & (uploaded_file is not None):
        fig, profiles, imp_feat = profile_clusters(df_p, categorical=cat_features)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(f'<h3 "> Profiles for {len(np.unique(clusters))} Clusters </h3>',
                                unsafe_allow_html=True)
        fig.update_layout(
            autosize=True,
                margin=dict(
                    l=0,
                    r=0,
                    b=0,
                    t=40,
                    pad=0
                ),
            )
        st.plotly_chart(fig)

        show = st.checkbox('Show up to 20 most important features')
        if show == True:
            l = np.min([20, len(imp_feat)])
            st.write(imp_feat[:l])

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(f'<h3 "> Features Overview </h3>',
                    unsafe_allow_html=True)
        feature = st.selectbox('Select a feature...', options = df_p.columns)
        feat_fig = profile_feature(df_p, feature)
        st.plotly_chart(feat_fig)

        st.subheader('Downloads')
        st.write(get_table_download_link(profiles,'profiles.csv', 'Download profiles'), unsafe_allow_html=True)

    elif (prof == False) & (uploaded_file is not None):
        st.markdown("""
        <br>
        <h2 style="color:#26608e;"> Data is read in. Check the box to profile </h2>

        """, unsafe_allow_html=True)
