from base64 import encode
import numpy as np
import pandas as pd
from scipy.stats import uniform, randint

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import chardet
import json
import csv
import math
import os 
import sys
import re
import base64
import shutil
from jinja2 import Template
import codecs
from io import StringIO
import io

from flask import Flask, redirect, render_template, request, send_from_directory

app = Flask(__name__)
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = '/code/files'
DOWNLOAD_FOLDER = '/code/download'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] =  DOWNLOAD_FOLDER


def load_csv(csvFilePath):
    jsonArray = []

    f = open(csvFilePath,"rb").read()

    asdf = chardet.detect(f)

    #read csv file
    with open(csvFilePath, encoding=asdf['encoding']) as csvf: 
        
        csvReader = csv.DictReader(csvf) 

        #convert each csv row into python dict
        for row in csvReader: 
            #add this python dict to json array
            jsonArray.append(row)
    return jsonArray
    
    
@app.route('/', methods=['GET'])
def index():
    onlyfiles = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]

    if "tmp.csv" in onlyfiles:
        onlyfiles.remove("tmp.csv")

    return render_template('index.html', data=onlyfiles)

@app.route('/remove_model', methods=['POST'])
def remove_model():

    feture_vec = list( request.form.to_dict().values() )
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], feture_vec[0])
    os.remove(file_path)
    
    return redirect("/")

@app.route('/load_model', methods=['POST'])
def load_model():

    feture_vec = list( request.form.to_dict().values() )
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], feture_vec[0])
    
    data = load_csv( file_path)

    return render_template('model.html', data={
        "data": data,
        "model_name": feture_vec[0]
    })

# Get the uploaded files
@app.route("/upload", methods=['POST'])
def uploadFiles():

    uploaded_file = request.files['file']
    sep = request.form['seperator']
    ending = request.form['ending']
    linestart = request.form['linestart']

    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
    
    f = open(file_path,"rb").read()

    codec = chardet.detect(f)

    BLOCKSIZE = 1048576 
    tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], "tmp.csv")
    with codecs.open(file_path, "r", codec['encoding']) as sourceFile:
        with codecs.open(tmp_name , "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)
    os.remove(file_path)

    with open(tmp_name, 'r') as fin:
        data = fin.read().splitlines(True)
        
    with open(tmp_name, 'w') as fout:
        fout.writelines( [ e for e in data[int(linestart):] ])

    asdd = pd.read_csv(tmp_name, sep=sep)
    asdd.drop(asdd.filter(regex="Unname"),axis=1, inplace=True)
    asdd.to_csv(tmp_name, index=False)

    return render_template('label.html', data={
        "lable_predict_single": [],
        "data": load_csv(tmp_name)
    })

@app.route('/save_model', methods=['POST'])
def save_model_create():
    feture_vec = list( request.form.to_dict().values() )

    model_name = feture_vec[0] + ".csv"
    feture_vec = feture_vec[1:]

    final_name = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
    tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], "tmp.csv")

    df_pred = pd.read_csv(tmp_name, delimiter=',')

    df_pred.insert(len(df_pred.columns), "Kategorie", feture_vec)

    df_pred.to_csv(final_name, sep=",", index=False)

    data = load_csv(final_name)

    onlyfiles = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))]

    onlyfiles.remove("tmp.csv")

    return render_template('index.html', data=onlyfiles)


@app.route('/save', methods=['POST'])
def save_model_predict():
    feture_vec = list( request.form.to_dict().values() )

    train_model = os.path.join(app.config['UPLOAD_FOLDER'], feture_vec[0])
    predict_model = os.path.join(app.config['UPLOAD_FOLDER'], feture_vec[1])
    predict_model_down_filename = feture_vec[1]
    predict_model_down = os.path.join(app.config['DOWNLOAD_FOLDER'], feture_vec[1])

    shutil.move(predict_model, predict_model_down)

    predict_model = os.path.join(app.config['DOWNLOAD_FOLDER'], feture_vec[1])

    feture_vec = feture_vec[2:]

    print(feture_vec)


    df = pd.read_csv(train_model, delimiter=',')
    df_pred = pd.read_csv(predict_model_down, delimiter=',')

    df_pred.insert(len(df_pred.columns), "Kategorie", feture_vec)

    df_all = pd.concat([df, df_pred])
    
    df_all.to_csv(train_model, sep=",", index=False)
    df_pred.to_csv(predict_model_down, sep=",", index=False)

    data = load_csv(predict_model_down)

    return send_from_directory(directory=app.config['DOWNLOAD_FOLDER'], path=predict_model_down_filename)
    
    return render_template('model.html', data=data)


@app.route('/train', methods=['POST'])
def train_model_predict():
    feture_vec = list( request.form.to_dict().values() )

    uploaded_file = request.files['form_file']

    form_sep = feture_vec[0]
    form_end = feture_vec[1]
    form_linestart = feture_vec[2]
    form_model_name = feture_vec[3]

    # print(form_sep)
    # print(form_end)
    # print(form_linestart)
    # print(form_model_name)

    feture_vec = feture_vec[3:]

    ### --- Save the file ---

    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "tmp_"+uploaded_file.filename)
        uploaded_file.save(file_path)
    
    f = open(file_path,"rb").read()

    codec = chardet.detect(f)

    BLOCKSIZE = 1048576 
    tmp_name = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    with codecs.open(file_path, "r", codec['encoding']) as sourceFile:
        with codecs.open(tmp_name , "w", "utf-8") as targetFile:
            while True:
                contents = sourceFile.read(BLOCKSIZE)
                if not contents:
                    break
                targetFile.write(contents)
    os.remove(file_path)

    with open(tmp_name, 'r') as fin:
        data = fin.read().splitlines(True)
        
    with open(tmp_name, 'w') as fout:
        fout.writelines( [ e for e in data[int(form_linestart):] ])

    asdd = pd.read_csv(tmp_name, sep=form_sep)
    asdd.drop(asdd.filter(regex="Unname"),axis=1, inplace=True)
    asdd.to_csv(tmp_name, index=False)

    train_path_name = os.path.join(app.config['UPLOAD_FOLDER'], form_model_name)


    df = pd.read_csv(train_path_name, delimiter=',')
    df_pred = pd.read_csv(tmp_name, delimiter=',')

    df_x = pd.DataFrame(df, columns=feture_vec).values.tolist()
    df_x = [ ' '.join(str(e) for e in st) for st in df_x ]

    df_y = pd.DataFrame(df, columns=['Kategorie']).values.tolist()

    df_x_predict = pd.DataFrame(df_pred, columns=feture_vec).values.tolist()
    df_x_predict = [ ' '.join(str(e) for e in st) for st in df_x_predict ]

    vec = CountVectorizer(analyzer="word",token_pattern="[A-Z]{3,}[a-z]{3,}|[a-zA-Z]{3,}",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)

    df_x_trans = vec.fit_transform(df_x + df_x_predict)
    output_transformer = OrdinalEncoder()

    X_all = df_x_trans.toarray()

    X_train = X_all[0:len(df_x)]
    X_predict = X_all[len(df_x):]
    Y = output_transformer.fit_transform(df_y)
    clf = LogisticRegression(random_state=0)
    #clf = DecisionTreeClassifier(random_state=0)
    #clf = RandomForestClassifier(max_depth=2, random_state=0)

    clf.fit(X_train,Y)


    y_pred = clf.predict(X_predict)
    y_pred_prob = clf.predict_proba(X_predict)
    y_pred_str = output_transformer.inverse_transform(y_pred.reshape(-1, 1))
    y_pred_str = [e for ee in y_pred_str for e in ee]
    # print(y_pred_prob)
    # print( type(y_pred_prob) )

    y_pred_prob_single = 120.0 * np.amax(y_pred_prob, axis=1)

    return render_template('train.html', data={
        "fet_vec": feture_vec,
        "lable_predict": y_pred_str,
        "lable_predict_single": [ e[0] for e in df_y],
        "lable_probs": y_pred_prob_single,
        "data": load_csv(tmp_name),
        "train_model_name": form_model_name,
        "predict_model_name": uploaded_file.filename
    })

@app.route('/asdf/', methods=['GET'])
def fetch():
    data = load_csv('/code/train.csv')

    return render_template('train.html', data=json)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')