import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import joblib 

DATASET_PATH = r"dataset/titanic3.xls"
"""
Parameters
----------
pclass : class of the ticket (1, 2, 3)
survived : whether the passenger survived (0=no, 1=yes)
name : name of the passenger
sex : Sex
age : Age
sibsp : Number of siblings/spouses aboard
parch : Number of parents/children aboard
ticket : Ticket number
fare : Passenger fare
cabin : Cabin
embarked : Port of embarkation
boat : Lifeboat
body : Body identification number
home.dest : Home/destination
"""


def load_data():
    data = pd.read_excel(DATASET_PATH)
    return data

def preprocess_data(data):
    data.drop(["name", "ticket", "cabin", "boat", "body", "home.dest", "embarked"], axis=1, inplace=True) # drop columns that are not useful
    data['sex'] = data['sex'].map({'male': 0, 'female': 1}) # convert sex to numerical values

def basic_statistics(data):
    print(data.describe())
    survival_rate = data['survived'].mean() * 100
    print(f"Overall chance of survival: {survival_rate:.2f}%")


class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Define functions to train and evaluate the model
def train_model(X_train, y_train, input_dim, epochs=100, lr=0.01):
    model = TitanicModel(input_dim) 
    criterion = nn.BCELoss() # binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32)) # create a dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(epochs): # loop over the dataset multiple times
        for inputs, labels in dataloader: # get the inputs; data is a list of [inputs, labels]
            optimizer.zero_grad() # zero the parameter gradients
            outputs = model(inputs).squeeze() # forward pass
            loss = criterion(outputs, labels) # compute the loss
            loss.backward() # backward pass
            optimizer.step() # optimize 
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    @param model: The trained model
    @param X_test: The test features
    @param y_test: The test labels
    @return: The accuracy of the model
    """
    with torch.no_grad():
        outputs = model(torch.tensor(X_test, dtype=torch.float32)).squeeze()
        predictions = (outputs >= 0.5).float()
        accuracy = (predictions == torch.tensor(y_test.values, dtype=torch.float32)).float().mean().item()
    return accuracy

def preprocess_data(data):
    data.drop(["name", "ticket", "cabin", "boat", "body", "home.dest", "embarked"], axis=1, inplace=True) # drop columns that are not useful
    data['sex'] = data['sex'].map({'male': 0, 'female': 1}) # convert sex to numerical values
    return data

def main():
    data = load_data()
    data = preprocess_data(data)

    X = data.drop("survived", axis=1)
    y = data["survived"]

    # Impute missing values and scale features
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)

    # Save the scaler
    joblib.dump(scaler, 'best_model.pkl')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model with hyperparameter tuning
    model = LogisticRegression(max_iter=9000) # create a logistic regression model
    param_grid = {'C': [0.1, 1, 10, 100]} # hyperparameters to tune
    grid_search = GridSearchCV(model, param_grid, cv=5) # 5-fold cross-validation
    grid_search.fit(X_train, y_train)

    # Save the best model
    joblib.dump(grid_search.best_estimator_, r'preloaded/best_model.pkl')

    # Make predictions
    y_pred = grid_search.best_estimator_.predict(X_test) # use the best model

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    main()
