import pickle
from flask import Flask, render_template, request  
import numpy as np 

import init

import pymongo 
myclient = pymongo.MongoClient("mongodb://127.0.0.1:27017/")  

app = Flask(__name__) 

# creating database and collection 
try: 
    mydb = myclient['titanicDatabase']
    mycol = mydb['titanicCollection'] 
    print("database and collection has created") 
except: 
    print('oops! something wrong. Database or collection has not created') 

# loading model 
file_name = "models/random_forest_hyperparameter_tuning.sav"
loaded_model = pickle.load(open(file_name, 'rb')) 



@app.route('/') 
def input_data(): 
    return render_template('input.html') 


@app.route('/result', methods=['POST', 'GET'])
def result(): 
    if request.method == 'POST': 
        
        pclass = request.form['pclass']
        sex = request.form['sex']
        age = request.form['age']
        sibsp = request.form['sibsp']
        fare = request.form['fare']
        embarked = request.form['embarked'] 
        parch = request.form['parch'] 
        deck =  request.form['deck']
        title = request.form['title']
            

        # proprocessing inputted data 
        pclass_val = init.pclass_fea(int(pclass)) 
        sex_val = init.sex_fea(sex) 
        age_val = init.age_fea(int(age))  
        sibsp_val = init.sibsp_fea(int(sibsp))   
        fare_val = init.fare_fea(fare)     
        embarked_val = init.embarked_fea(embarked)
        parch_val = init.parch_fea(int(parch)) # using for counting relatives
        relative_val = sibsp_val + parch_val # finding relatives 
        deck_val = init.deck_fea(deck) 
        title_val = init.title_fea(title) 
        age_class_val = age_val * pclass_val 
        fare_per_person_val = int(fare_val /  (relative_val + 1))  

        # predicting with imported data 
        try:
            predicted_result = init.predicting(
                pclass_val, 
                sex_val, 
                age_val, 
                sibsp_val, 
                fare_val, 
                embarked_val, 
                relative_val, 
                deck_val, 
                title_val, 
                age_class_val, 
                fare_per_person_val, 
                loaded_model
                )
            msg_prediction = 'prediction using model succied'
        except: 
            msg_prediction = 'Something wrong! Prediction not success' 
        
        try: 
            # making collection of the imported and predicted data 
            data_dict = {
                "pclass": pclass_val, 
                "sex": sex_val, 
                "age": age_val, 
                "sibsp": sibsp_val, 
                'fare': fare_val, 
                'embarked': embarked_val, 
                'relative': relative_val, 
                'deck': deck_val, 
                'title': title_val, 
                'age_class': age_class_val, 
                'fare_per_person': fare_per_person_val, 
                'predicted': predicted_result
                }

            # inserting data into database 
            mycol.insert_one(data_dict) 
            msg_insert_database = "data are inserted into database" 
        except: 
            msg_insert_database = "oops something wrong. Data are not inserted into database"
        finally:
            return render_template(
                                    'result.html', 
                                    predicted = predicted_result, 
                                    msg_database = msg_insert_database
                                    )  


if __name__ == '__main__': 
    app.run(debug=True) 

 