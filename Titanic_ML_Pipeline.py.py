import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

df=sns.load_dataset("titanic")
#print(df.columns)
df.drop(columns=['class', 'who', 'adult_male', 'deck', 'embark_town',
       'alive', 'alone'],inplace=True)
#print(df.columns)
X = df.drop(columns='survived')
y=df['survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

categoric_col=X_train.select_dtypes(include=['category','object']).columns
numeric_col=X_train.select_dtypes(include=['number','float']).columns

class VisualOutlierHandler(BaseEstimator,TransformerMixin):
    def __init__(self,skew_thresh=0.5):
        self.skew_thresh = skew_thresh
        self.already_printed =False
        self.already_check=False
        self.already_done=False
        self.already_box=False
        self.already_box2=False
    def fit(self,X,y=None):
        X=pd.DataFrame(X)
        self.columns=X.columns
        self.mean=X.mean()
        self.mode=X.mode()
        self.std=X.std()
        self.skew=X.skew()
        self.median=X.median()
        Q1=X.quantile(0.25)
        Q3=X.quantile(0.75)
        IQR=Q3-Q1
        self.skew_lower=(Q1-1.5*IQR).to_dict()
        self.skew_upper=(Q3+1.5*IQR).to_dict()
        self.upper=(self.mean + 3*self.std).to_dict()
        self.lower=(self.mean - 3*self.std).to_dict()
        return self

    def transform(self,X):
        X=pd.DataFrame(X)[self.columns].copy()
        skew_col=[]
        nor_col=[]

        skew_col=[]
        nor_col=[]
        if not self.already_check:
            self.already_check=True
            for col in self.columns:
                if self.skew[col]>self.skew_thresh or self.skew[col]<-self.skew_thresh:
                    skew_col.append(col)
                else:
                    nor_col.append(col)
        if not self.already_printed:
           a = 1
           self.already_printed =True
           # This is to check how the data is distributed.
           print("number of columns for hist plot: ", (len(self.columns)))
           row = abs(int(input("Enter subplot row: ")))
           coll = abs(int(input("Enter subplot column: ")))
           for col in self.columns:
            plt.subplot(row,coll,a)
            sns.histplot(X[col],kde=True)
            a+=1
            print(f"======{col}==========")
            print(" skew: ", self.skew[col])
            print(" mean: ", self.mean[col])
            print(" median: ", self.median[col])
            print(" mode: ",self.mode[col][0])
            print()
        if not self.already_done:
            self.already_done=True
            print("skew columns: ",skew_col)
            plt.show()
            a=input('''
            1. Do you agree
            2. No
            ''')
            if a=='2':
                new_col = input("Enter skew col separated by commas: ").split(',')
                nor_col = list(set(nor_col).intersection(set(skew_col)-set(new_col)))
                skew_col=new_col
                print(f"new skew col: ",skew_col)
        if not self.already_box:
          b=1
          self.already_box=True
          #This is to check how how many outlier in skew data
          print("number of columns for box plot: ", (len(skew_col)))
          row1 = abs(int(input("Enter subplot row: ")))
          col2 = abs(int(input("Enter subplot column: ")))
          for col in skew_col:

            plt.subplot(row1, col2,b)
            sns.boxplot(X[col],palette="muted")
            plt.title("with outlier")
            b+=1
        for col in skew_col:
                X[col] = X[col].clip(lower=self.skew_lower[col], upper=self.skew_upper[col])
        for col in nor_col:
                X[col]=X[col].clip(lower=self.lower[col],upper=self.upper[col])
        if not self.already_box2:
         c=1
         self.already_box2=True
         plt.show()
         #This is to check if the outliers have been removed in the skewed data
         print("number of columns for box plot: ", (len(skew_col)))
         row3 = abs(int(input("Enter subplot row: ")))
         col3 = abs(int(input("Enter subplot column: ")))
         for col in skew_col:
            plt.subplot(row3, col3, c)
            sns.boxplot(X[col], palette="muted")
            plt.title("with outlier")
            c += 1

        return X
numeric_Transformer=pipeline.Pipeline([
    ('outlier',VisualOutlierHandler()),
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
])

categoric_Transformer=pipeline.Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('OneHotEncoder', OneHotEncoder(handle_unknown='ignore',sparse_output=False)),
    ('scaler', StandardScaler()),
])


preprocessor = ColumnTransformer([
    ('numeric_col',numeric_Transformer,numeric_col ),
    ('categorical_col',categoric_Transformer,categoric_col ),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()),
])

pipeline.fit(X_train,y_train)

y_pred = pipeline.predict(X_test)

print("accuracy_score: ",accuracy_score(y_test,y_pred))
print("precision_score: ",precision_score(y_test,y_pred))
print("recall_score: ",recall_score(y_test,y_pred))
print("f1_score: ",f1_score(y_test,y_pred))
print("confusion_matrix:\n ",confusion_matrix (y_test,y_pred))
plt.show()