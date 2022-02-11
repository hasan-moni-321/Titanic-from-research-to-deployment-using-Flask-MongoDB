import numpy as np 


# preprocessing imported pclass data 
def pclass_fea(pclass_data):
    return int(pclass_data) 

# preprocessing imported sex data 
def sex_fea(sex_data):  
    if sex_data == "Male":
        return  0
    else: 
        return 1

# preprocessing imported age data 
def age_fea(age_data):  
    if age_data <= 11: 
        return 0 
    elif age_data > 11 and age_data <= 18: 
        return 1 
    elif age_data > 18 and age_data <= 22:
        return 2  
    elif age_data > 22 and age_data <= 27: 
        return 3  
    elif age_data > 27 and age_data <= 33:
        return 4  
    elif age_data > 33 and age_data <= 40:  
        return 5 
    elif age_data > 40 and age_data <= 66: 
        return 6
    else: 
        return 7  

# preprocessing imported sibsp data 
def sibsp_fea(sibsp_data): 
    return int(sibsp_data)  

# preprocessing imported fare data 
def fare_fea(fare_data): 
    if float(fare_data) <= 7.91: 
        return 0 

    elif float(fare_data) > 7.91 and float(fare_data) <= 14.454: 
        return 1 

    elif float(fare_data) > 14.454 and float(fare_data) <= 31: 
        return 2 

    elif float(fare_data) > 31 and float(fare_data) <= 99:
        return 3 

    elif float(fare_data) > 99 and float(fare_data) <= 250: 
        return 4 
    else: 
        return 5

# preprocessing imported embarked data 
def embarked_fea(embarked_data): 
    if embarked_data == 'S': 
        return 0 
    elif embarked_data == "C": 
        return 1
    else: 
        return 2

# preprocessing imported parch data 
def parch_fea(parch_data): 
    return int(parch_data) 

# preprocessing imported deck data 
def deck_fea(deck_data): 
    if deck_data == "A": 
        return 1
    elif deck_data == 'B': 
        return 2 
    elif deck_data == 'C': 
        return 3 
    elif deck_data == 'D': 
        return 4 
    elif deck_data == 'E': 
        return 5
    elif deck_data == 'F': 
        return 6 
    elif deck_data == 'G': 
        return 7
    elif deck_data == 'U': 
        return 8 
    else: 
        return 0 

# preprocessing imported title data 
def title_fea(title_data): 
    if title_data == 'Mr': 
        return 1
    elif title_data == 'Miss': 
        return 2
    elif title_data == 'Mrs': 
        return 3
    elif title_data == 'Master': 
        return 4
    elif title_data == 'Rare': 
        return 5 
    else: 
        return 0 


def predicting(pclass_val, sex_val, age_val, sibsp_val, fare_val, embarked_val, relative_val, deck_val, title_val, age_class_val, fare_per_person_val, loaded_model):
    pre_data = np.array([pclass_val, sex_val, age_val, sibsp_val, fare_val, embarked_val, relative_val, deck_val, title_val, age_class_val, fare_per_person_val])
    pre_data_reshape = pre_data.reshape(1, -1) 
    pre_result = loaded_model.predict(pre_data_reshape)
    return int(pre_result[0])  

