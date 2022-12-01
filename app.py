#This will be used to access other "endpoint pages" in your file

#This is also your Flask app.

from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

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
    print("\nRequest Form: ",request.form.values(),'\n')
    
    transformed_vals = [x for x in request.form.values()] #this converts the string values retrived via request.form.values() from model.py into float values.
    print("\nTransformed Vals: ",transformed_vals,'\n')

    data = {
        "experience_level": [transformed_vals[0]],
        "employment_type": [transformed_vals[1]],
        "remote_ratio": [transformed_vals[2]],
        "company_size": [transformed_vals[3]]
    }
    #load data into a DataFrame object:
    vals = pd.DataFrame(data)

    #vals = [np.array(transformed_vals)]
    #print("\nVals in array: ",vals,'\n')
    #vals = vals.reshape(1,-1)

    predictions = model.predict(vals)


    #This is where I uncode the values selected by the user to use as part of the output message.
    original_vals = transformed_vals

    exp_lvl = " "
    if original_vals[0] == '1':
        exp_lvl = 'an entry level'
    elif original_vals[0] == '2':
        exp_lvl = 'a middle level'
    elif original_vals[0] == '3':
        exp_lvl = 'a senior level'
    elif original_vals[0] == '4':
        exp_lvl = 'an executive level'
    else:
        exp_lvl = 'an unknown level'

        
    emp_type = " "
    if original_vals[1] == '1':
        emp_type = 'full-time employee'
    elif original_vals[1] == '2':
        emp_type = 'part-time employee'
    elif original_vals[1] == '3':
        emp_type = 'contractor'
    elif original_vals[1] == '4':
        emp_type = 'freelancer'
    else:
        emp_type = 'Unknown Employment Type'


    rem_rat = " "
    if original_vals[2] == '1':
        rem_rat = 'mostly in-office'
    elif original_vals[2] == '2':
        rem_rat = 'partly in-office and partly remote'
    elif original_vals[2] == '3':
        rem_rat = 'mostly remote'
    else:
        rem_rat = 'Unknown Work Ratio'
        

    comp_size = " "
    if original_vals[3] == '1':
        comp_size = 'small'
    elif original_vals[3] == '2':
        comp_size = 'medium'
    elif original_vals[3] == '3':
        comp_size = 'large'
    else:
        comp_size = 'Unknown Company Size'

    predicted_income = ("As {} {} working {} at a {} company, you could expect to make around {} USD.".format(exp_lvl,emp_type,rem_rat,comp_size,f'${predictions[0]:,.2f}'))
    
    
    #Lastly, we specify which page we want to send this information to so it is available for request.
    #We also must specify what it's object name is for reference in the page, which in this case is "Predictions".
    return render_template("index.html",Predictions = predicted_income)


if __name__ == '__main__':
    app.run(debug=True)