import joblib
import numpy as np
from sklearn.impute import SimpleImputer


"""
@title: Survival Generator
---------------------------------------------------------------------------------------------------------------------------------
This code snippet is used to generate a random set of values for the user input for a Titanic passenger that ensure they survive.

"""

MODEL_PATH = r'model/previous/previous_model.pkl'

model = joblib.load(MODEL_PATH)

# Function to preprocess user input
def preprocess_input(data):
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    return data

def random_value_dumping(model, user_data, num_simulations=1000):
    predictions = []
    for i in range(num_simulations):
        random_data = user_data + np.random.uniform(-1, 1, user_data.shape)  # Add random values
        random_data = preprocess_input(random_data) 
        prediction = model.predict(random_data)
        predictions.append(prediction[0])
    
    survival_probability = np.mean(predictions)
    return survival_probability

def run_until_survival(model, user_data):
    """
    @param model: The trained model
    @param user_data: The user data to predict survival
    this function runs the simulation until a survival probability of 0.5 or higher is achieved
    """
    survival = False
    while survival is False: 
        survival_probability = random_value_dumping(model, user_data)
        if survival_probability >= 0.5:
            print(f"Survival predicted with parameters: pclass={user_data[0][0]}, sex={user_data[0][1]}, age={user_data[0][2]:.2f}, sibsp={user_data[0][3]}, parch={user_data[0][4]}, fare={user_data[0][5]:.2f}")
            survival = True
            break

if __name__ == "__main__":
    # Randomized predefined values for the simulation
    pclass = np.random.choice([1, 2, 3])
    sex = np.random.choice([0, 1]) 
    age = np.random.uniform(1, 80)
    sibsp = np.random.randint(0, 10)
    parch = np.random.randint(0, 10)
    fare = np.random.uniform(0, 500)

    user_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
    user_data = preprocess_input(user_data)
    
    run_until_survival(model, user_data)
