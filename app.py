import numpy as np
from flask import Flask,request,jsonify, render_template
import pickle

# create flask app
app= Flask(__name__)

# load the pickle model
model= pickle.load(open("smote.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    BMI = int(request.form.get('BMI'))
    print(BMI)
    Smoker = int(request.form.get('Smoker'))
    print(Smoker)
    Stroke = int(request.form.get('Stroke'))
    print(Stroke)
    HeartDiseaseorAttack = int(request.form.get('HeartDiseaseorAttack'))
    print(HeartDiseaseorAttack)
    PhysActivity = int(request.form.get('PhysActivity'))
    print(PhysActivity)
    Veggies = int(request.form.get('Veggies'))
    print(Veggies)
    HvyAlcoholConsump = int(request.form.get('HvyAlcoholConsump'))
    print(HvyAlcoholConsump)
    GenHlth = int(request.form.get('GenHlth'))
    print(GenHlth)
    PhysHlth = int(request.form.get('PhysHlth'))
    print(PhysHlth)
    DiffWalk = int(request.form.get('DiffWalk'))
    print(DiffWalk)
    Age = int(request.form.get('Age'))
    print(Age)
    Fruits = int(request.form.get('Fruits'))
    print(Age)

    HighBP = int(request.form.get('HighBP'))
    print(HighBP)

    HighChol = int(request.form.get('HighChol'))
    print(HighChol)
    # BMI = int(request.form.get('BMI'))
    # print(BMI)
    # Smoker = int(request.form.get('Smoker'))
    # print(Smoker)
    # Stroke  = int(request.form.get('Stroke'))
    # print(Stroke)
    # HeartDiseaseorAttack = int(request.form.get('HeartDiseaseorAttack'))
    # print(HeartDiseaseorAttack)
    # PhysActivity = int(request.form.get('PhysActivity'))
    # print(PhysActivity)
    # Veggies = int(request.form.get('Veggies'))
    # print(Veggies)
    # HvyAlcoholConsump = int(request.form.get('HvyAlcoholConsump'))
    # print(HvyAlcoholConsump)
    # GenHlth = int(request.form.get('GenHlth'))
    # print(GenHlth)
    # PhysHlth= int(request.form.get('PhysHlth'))
    # print(PhysHlth)
    # DiffWalk = int(request.form.get('DiffWalk'))
    # print(DiffWalk)
    # Age = int(request.form.get('Age'))
    # print(Age)
    # Fruits = int(request.form.get('Fruits'))
    # print(Age)
    #
    # No_HighBP = int(request.form.get('No_HighBP'))
    # print(No_HighBP)
    # Yes_HighBP = int(request.form.get('Yes_HighBP'))
    # print(Yes_HighBP)
    # No_HighChol = int(request.form.get('HighChol'))
    # print(No_HighChol)
    # Yes_HighChol = int(request.form.get('Yes_HighChol'))
    # print("yep")
    # print("hifh",Yes_HighChol)







    # prediction= model.predict(np.array([BMI, Smoker,
    #    Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
    #    HvyAlcoholConsump, GenHlth,PhysHlth, DiffWalk, Age
    #    ,No_HighBP,Yes_HighBP,No_HighChol,Yes_HighChol]).reshape(1,16))

    prediction = model.predict(np.array([BMI, Smoker,
                Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
                HvyAlcoholConsump, GenHlth, PhysHlth, DiffWalk, Age
                ,HighBP, HighChol ]).reshape(1, 14))
    print(prediction)
    if (prediction ==0):
        return render_template("index.html", prediction_text="You are healthy.")
    elif (prediction ==1):
        return render_template("index.html", prediction_text="You are in Pre-Diabetes stage.")
    else:
        return render_template("index.html", prediction_text="You have Diabetes.")
if __name__=="__main__":
    app.run(debug=True)