from flask import Flask,render_template,request
import pickle
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error
from xgboost import XGBRegressor

model = pickle.load(open('pipe.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_player_value():
    fpl_value = float(request.form.get('fpl_value'))
    nationality = request.form.get('nationality')
    page_views = int(request.form.get('page_views'))
    fpl_points = int(request.form.get('fpl_points'))
    position_cat = int(request.form.get('posCat'))
    new_signing = int(request.form.get('new_signing'))
    new_Foreign = int(request.form.get('newForeign'))
    fpl_sel = float(request.form.get('fpl_sel'))
    region = int(request.form.get('Region'))
    club_id = int(request.form.get('clubId'))
    age_cat = int(request.form.get('AgeCat'))
    big_club = int(request.form.get('BigClub'))


    # making porediction
    result = model.predict([[ 
                              region,
                              fpl_value, 
                              age_cat, 
                              new_signing, 
                              new_Foreign, 
                              club_id, 
                              fpl_points, 
                              nationality, 
                              big_club, 
                              page_views, 
                              fpl_sel,
                              position_cat
                            ]])

    return render_template('index.html',result = np.round(np.exp(result[0]),decimals=3))


# ***********************************************

# def predict_placement():
#     fpl_value = float(request.form.get('fpl_value'))
#     iq = int(request.form.get('iq'))
#     profile_score = int(request.form.get('profile_score'))

#     # prediction
#     result = model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))

#     if result[0] == 1:
#         result = 'placed'
#     else:
#         result = 'not placed'

#     return render_template('index.html',result=result)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)