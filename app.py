import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from flask import Flask, request, jsonify, render_template
import pickle

# =============================================================================
# Loading 
# =============================================================================

columns = ['ID','Agency','Agency Type', 'DistChannel', 'Product', 'Duration', 'Destination', 'NetSales', 'CommisionValue', 'Age','Claim']
df = pd.read_csv('data/train.csv',names=columns,skiprows=1)


le = LabelEncoder()
df["Agency"] = le.fit_transform(df["Agency"])
df["Agency Type"] = le.fit_transform(df["Agency Type"])
df["DistChannel"] = le.fit_transform(df["DistChannel"])
df["Product"] = le.fit_transform(df["Product"])
df["Destination"] = le.fit_transform(df["Destination"])
    

# =============================================================================
# df =  df.drop(["ID"],1)
# =============================================================================


le = LabelEncoder()
df["Agency"] = le.fit_transform(df["Agency"])
df["Agency Type"] = le.fit_transform(df["Agency Type"])
df["DistChannel"] = le.fit_transform(df["DistChannel"])
df["Product"] = le.fit_transform(df["Product"])
df["Destination"] = le.fit_transform(df["Destination"])
    
# removing duration outliear
#print((df["Duration"]<=0).value_counts())
df.drop(df[df["Duration"]<=0].index, inplace = True)
df.drop(df[df["Duration"]>=4000].index, inplace = True)
# end

X = df.drop(["Claim"],1)
y = df["Claim"]
X_train, X_test, y_train, y_test = tts(X,y,random_state = 42, test_size = 0.25, stratify = y)


# =============================================================================
# df =  df.drop(["ID"],1)
# =============================================================================


le = LabelEncoder()
df["Agency"] = le.fit_transform(df["Agency"])
df["Agency Type"] = le.fit_transform(df["Agency Type"])
df["DistChannel"] = le.fit_transform(df["DistChannel"])
df["Product"] = le.fit_transform(df["Product"])
df["Destination"] = le.fit_transform(df["Destination"])
    
# removing duration outliear
#print((df["Duration"]<=0).value_counts())
df.drop(df[df["Duration"]<=0].index, inplace = True)
df.drop(df[df["Duration"]>=4000].index, inplace = True)
# end

X = df.drop(["Claim"],1)
y = df["Claim"]
X_train, X_test, y_train, y_test = tts(X,y,random_state = 42, test_size = 0.25, stratify = y)
# =============================================================================
# End
# =============================================================================

app = Flask(__name__)


# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [(x) for x in request.form.values()]
    model_name = int_features[0]
    # prediction = model.predict(final_features)
    if(model_name=='dtc_simple'):
        model = pickle.load(open('pickle/dtc.pkl', 'rb'))
    elif(model_name=='rf_simple'):
        model = pickle.load(open('pickle/ran_for.pkl', 'rb'))
    elif(model_name=='log_cls'):
        model = pickle.load(open('pickle/log_cls.pkl', 'rb'))



    y_pred = model.predict(X_test)
    accuracy = round(accuracy_score(y_test,y_pred),2)
    
    classify = (classification_report(y_test,y_pred,output_dict=True))
    dfcc = pd.DataFrame(classify).transpose()
    zero_prec= round(dfcc.iloc[0,0],3)
    one_prec= round(dfcc.iloc[1,0],3)
    
    zero_rec= round(dfcc.iloc[0,1],3)
    one_rec= round(dfcc.iloc[1,1],3)
    
    zero_fscore= round(dfcc.iloc[0,2],3)
    one_fscore= round(dfcc.iloc[1,2],3)

    return render_template('index.html', prediction_text='Accuracy Score is {}'.format(accuracy), model_text=model.best_estimator_,zero_prec=zero_prec,zero_rec=zero_rec,zero_fscore=zero_fscore,one_prec=one_prec,one_rec=one_rec,one_fscore=one_fscore)



if __name__ == "__main__":
    app.run(debug=True)