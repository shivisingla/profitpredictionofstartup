import numpy as np
import pickle
from flask import Flask, render_template , request, redirect, jsonify

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        RnD_Spend = request.form['RnD_Spend']
        Administration = request.form['Administration']
        Marketing_Spend = request.form['Marketing_Spend']
        X_list = [float(RnD_Spend),float(Administration),float(Marketing_Spend)]
        state = request.form['state']
        if state == 'NewYork':
            X_list.append(0)
            X_list.append(1)
        elif state == 'California':
            X_list.append(0)
            X_list.append(0)
        elif state == 'Florida':
            X_list.append(1)
            X_list.append(0)
        
        y_pred = model.predict([X_list])
        return render_template('index.html',y_pred = int(y_pred[0]))


if __name__=="__main__":
    app.run(debug=True)