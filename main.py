from flask import Flask,render_template,request
import joblib


app=Flask(__name__)

@app.route('/')
def base():
    return render_template('home.html')


@app.route('/predict', methods=['post'])
def predict():
    preg=request.form.get('Pregnancies')
    glc=request.form.get('Glucose')
    bp=request.form.get('BloodPressure')
    st=request.form.get('SkinThickness')
    insu=request.form.get('Insulin')
    bmi=request.form.get('BMI')
    dpf=request.form.get('DiabetesPedigreeFunction')
    age=request.form.get('Age')

   
    
   # print(preg , glc , bp, st,insu,bmi,dpf,age)
   
    model=joblib.load('diabetic.pkl')


    data=model.predict([[preg, glc , bp, st,insu,bmi,dpf,age]])

    if data[0]==0:
        result="Person is not Diabetic"
    else:
        result="Better take care buddy"

    return render_template('/predict.html',data=result)


if __name__=="__main__":
    app.run(debug=True)