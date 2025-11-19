from flask import Flask,request,jsonify,render_template
import pandas as pd
import pickle
app3 = Flask(__name__)

def get_cleaned_data(form_data):
    gestation = float(form_data['gestation'])
    parity = int(form_data['parity'])
    age = float(form_data['age'])
    height = float(form_data['height'])
    weight = float(form_data['weight'])
    smoke = float(form_data['smoke'])

    cleaned_data = {"gestation":[gestation],
                    "parity":[parity],
                    "age":[age],
                    "height":[height],
                    "weight":[weight],
                    "smoke":[smoke]
                    }
    return cleaned_data
@app3.route('/',methods=["GET"])
def home():
    return render_template("index3.html")
## define endpoint
@app3.route("/predict",methods=['POST'])
def get_prediction():
    # get data form user
    # baby_data = request.get_json()
    baby_data_form = request.form
    baby_data_cleaned = get_cleaned_data(baby_data_form)

    # convert into dataframe
    baby_df = pd.DataFrame(baby_data_cleaned)

    #load machine learning trained model
    with open("model.pkl",'rb') as obj:
        model = pickle.load(obj)
    
    # make prediction on user data
    prediction = model.predict(baby_df)
    prediction = round(float(prediction),2)

    # return response in a json format
    response = {"prediction":prediction}

    # return jsonify(response)
    return render_template("index3.html",prediction=prediction)



if __name__ == '__main__':
    app3.run(debug=True)