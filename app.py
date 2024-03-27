

import streamlit as st
import subprocess
import os
import webbrowser 
import pandas as pd
import sklearn
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV



st.set_page_config(
    page_title="Diabetics Model", 
    layout="centered",
    initial_sidebar_state="expanded") 

def open_mlflow_ui():
    cmd = "mlflow ui --port 8080"
    subprocess.Popen(cmd, shell=True)
def open_browser(url):
    webbrowser.open_new_tab(url)

if st.sidebar.button("Launch ML server"):
    open_mlflow_ui()
    st.sidebar.success("MLflow Server http://127.0.0.1:8080")
    mlflow.set_experiment("Experimentation with ML Flow")
    open_browser("http://127.0.0.1:8080")



raw_data = pd.read_csv("diabetes.csv")
raw_data2=raw_data.loc[:,raw_data.columns!='outcome']
selected_features=st.multiselect("Selecting features to train",raw_data2.columns)


class Diabetes_m:
    def __init__(self,raw_data):
        self.data=raw_data
        self.input_data()
        self.training()

    def __call__(self):
        return [self.acc,self.f1]
    
    def input_data(self):
        self.data.rename(columns=str.lower,inplace=True)
        self.column_names=list(self.data.columns.values)
        del self.column_names[-1]
        ### Scaling data
        scale=StandardScaler()
        self.scaled_data=pd.DataFrame(scale.fit_transform(self.data.iloc[:,0:8]),columns=self.column_names)

    def training(self):
        self.l=[]
        x=self.scaled_data
        y=self.data['outcome']
        x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
        params={
            "n_estimators":[90,100],
            "max_depth":[3,5],
            "min_samples_split":[2,3],
            "learning_rate":[0.09,0.95,0.1],
            "subsample":[0.8,0.9]}
    
        model = GradientBoostingClassifier()
        grid=GridSearchCV(model,params).fit(x_train, y_train)
        model1=grid.best_estimator_
        pred=model1.predict(x_test)
        self.acc=accuracy_score(y_test, pred)
        self.f1=f1_score(y_test, pred)
        print(grid.best_params_,grid.best_score_,grid.best_estimator_)
        print("Accuracy score:",round(self.acc,2))
        self.l.extend([self.acc,self.f1])
        #mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
        mlflow.set_experiment("Experimentation with ML Flow")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metrics({"Accuracy":round(self.acc,2),"F1-Score":round(self.f1,2)})
            mlflow.set_tag('Loading  training set',"Diabetes model")
            signature = infer_signature(x_train, model1.predict(x_train))
            model_ = mlflow.sklearn.log_model(
                sk_model=model1,
                artifact_path="diabetes_model",
                signature=signature,
                input_example=x_train,
                registered_model_name="tracking diabetes model",
            )
            return self.acc,self.f1
        
if st.button("Train"):
    n=Diabetes_m(raw_data2)
    st.success("Final model after training")
    st.write("Accuracy and F1 score",n())
