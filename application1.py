import pickle
from flask import Flask, request ,render_template
import numbers as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


application1 = Flask(__name__)
app = application1

## import ridge regressor model and scaler pkl file
 
ridge_model = pickle.load(open('Models/Ridge_regressor_Algerian.pkl','rb'))
std_scaler = pickle.load(open('Models/Scaler_Algerian.pkl','rb'))

# Rout for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
    #     pass

        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scaled_data = std_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_scaled_data) 

        return render_template('home.html',result = result[0])
    
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0")