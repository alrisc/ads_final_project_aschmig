#This will be used to access other "endpoint pages" in your file

#This is also your Flask app.

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#Then load the pickle file, which contains our model created in the "model.py" file.
model = pickle.load(open("model.pkl","rb"))

#Next, we'll create Flask endpoints.  Endpoints are objects that can be queried to retrieve something from the (package?/pickle?).

@app.route('/')
def home():
    #"render_template" is used to render files.  In our case, this will render our html files.
    return render_template("index.html")

#"request" is used to receive values that have been returned from the model.py file.
@app.route("/predict",methods=['POST'])
def predict():
    #transformed_vals = request.form.values()
    transformed_vals = [x for x in request.form.values()] #this converts the string values retrived via request.form.values() from model.py into float values.

    vals = [np.array(transformed_vals)]
    
    #vals = vals.reshape(1,-1)

    predictions = model.predict(vals)

    #Lastly, we specify which page we want to send this information to so it is available for request.
    #We also must specify what it's object name is for reference in the page, which in this case is "Predictions".
    return render_template("index.html",Predictions = predictions)


if __name__ == '__main__':
    app.run(debug=True)