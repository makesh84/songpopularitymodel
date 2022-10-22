from flask import Flask, request, render_template
import pickle
import numpy as np
app=Flask(__name__,template_folder='./templates')
model = pickle.load(open("model.pkl", "rb"))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
#if above line does not fetch expected output, try code given below:
#   int_features = [float(x) for x in list(request.form.values())]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 0:
        output = "This song is not popular."
    else :
        output = "This song is popular."
    return render_template('index.html', result=output)
if __name__ =="__main__":
    app.run()