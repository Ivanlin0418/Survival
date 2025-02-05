DATASET = r'dataset/titanic3.xls'

import numpy as np
import pandas as pd
import plotly.express as px

def load_data():
    return pd.read_excel(DATASET)

def preprocess_data(data):
    # Example preprocessing steps
    data = data.dropna(subset=['age', 'fare', 'survived', 'pclass'])
    return data

def plot_interactive_age_scatter(data):
    fig = px.scatter(data, x='age', y='pclass', color='pclass', 
                     labels={'age': 'Age', 'pclass': 'Passenger Class'},
                     title='Scatter plot of Age vs Passenger Class',
                     hover_data=['fare', 'survived', 'sex'])
    fig.show()

def plot_interactive_fare_scatter(data):
    fig = px.scatter(data, x='fare', y='pclass', color='pclass', 
                     labels={'fare': 'Fare', 'pclass': 'Passenger Class'},
                     title='Scatter plot of Fare vs Passenger Class',
                     hover_data=['age', 'survived', 'sex'])
    fig.show()

def plot_interactive_gender_scatter(data):
    fig = px.scatter(data, x='sex', y='pclass', color='pclass', 
                     labels={'sex': 'Gender', 'pclass': 'Passenger Class'},
                     title='Scatter plot of Gender vs Passenger Class',
                     hover_data=['age', 'fare', 'survived'])
    fig.show()

def plot_interactive_age_fare_scatter(data, filter_survived=None):
    if filter_survived is not None:
        data = data[data['survived'] == filter_survived]
    fig = px.scatter(data, x='age', y='fare', color='survived', 
                     labels={'age': 'Age', 'fare': 'Fare', 'survived': 'Survived'},
                     title='Scatter plot of Age vs Fare',
                     hover_data=['pclass', 'sex'])
    fig.show()

def plot_interactive_gender_fare_scatter(data, filter_survived=None):
    if filter_survived is not None:
        data = data[data['survived'] == filter_survived]
    fig = px.scatter(data, x='sex', y='fare', color='survived', 
                     labels={'sex': 'Gender', 'fare': 'Fare', 'survived': 'Survived'},
                     title='Scatter plot of Gender vs Fare',
                     hover_data=['age', 'pclass'])
    fig.show()

def basic_statistics(data):
    print(data.describe())

def main():
    data = load_data()
    data = preprocess_data(data)
    
    while True:
        print("Select an option:")
        print("1. Plot interactive age scatter")
        print("2. Plot interactive fare scatter")
        print("3. Plot interactive gender scatter")
        print("4. Plot interactive age vs fare scatter (both)")
        print("5. Plot interactive age vs fare scatter (survived only)")
        print("6. Plot interactive age vs fare scatter (not survived only)")
        print("7. Plot interactive gender vs fare scatter (both)")
        print("8. Plot interactive gender vs fare scatter (survived only)")
        print("9. Plot interactive gender vs fare scatter (not survived only)")
        print("10. Show basic statistics")
        print("11. Exit")
        
        choice = input("Enter choice: ")
        
        if choice == '1':
            plot_interactive_age_scatter(data)
        elif choice == '2':
            plot_interactive_fare_scatter(data)
        elif choice == '3':
            plot_interactive_gender_scatter(data)
        elif choice == '4':
            plot_interactive_age_fare_scatter(data)
        elif choice == '5':
            plot_interactive_age_fare_scatter(data, filter_survived=1)
        elif choice == '6':
            plot_interactive_age_fare_scatter(data, filter_survived=0)
        elif choice == '7':
            plot_interactive_gender_fare_scatter(data)
        elif choice == '8':
            plot_interactive_gender_fare_scatter(data, filter_survived=1)
        elif choice == '9':
            plot_interactive_gender_fare_scatter(data, filter_survived=0)
        elif choice == '10':
            basic_statistics(data)
        elif choice == '11':
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()