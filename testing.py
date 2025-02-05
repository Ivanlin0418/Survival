import joblib
import numpy as np
from sklearn.impute import SimpleImputer

# Load the saved model
model = joblib.load(r'model\preloaded\previous_model.pkl')

# Function to preprocess user input
def preprocess_input(data):
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    return data

def get_user_input():
    pclass = int(input("Enter class of the ticket (1, 2, 3): "))
    sex = int(input("Enter sex (0=male, 1=female): "))
    age = float(input("Enter age: "))
    sibsp = int(input("Enter number of siblings/spouses aboard: "))
    parch = int(input("Enter number of parents/children aboard: "))
    fare = float(input("Enter fare: "))

    user_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
    user_data = preprocess_input(user_data)
    
    prediction = model.predict(user_data)
    print("Survived" if prediction[0] == 1 else "Did not survive")

if __name__ == "__main__":
    get_user_input()
