from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
import plotly.express as px
import os
from flask_caching import Cache

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Load the saved model and cache it
model = joblib.load(r'model/previous/previous_model.pkl')

# Function to preprocess user input
def preprocess_input(data):
    imputer = SimpleImputer(strategy='mean')
    data = imputer.fit_transform(data)
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])

    user_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
    user_data = preprocess_input(user_data)
    
    prediction = model.predict(user_data)
    result = "Survived" if prediction[0] == 1 else "Did not survive"
    
    return jsonify(result=result)

@app.route('/simulate', methods=['POST'])
def simulate():
    pclass = np.random.choice([1, 2, 3])
    sex = np.random.choice([0, 1])  # 0=male, 1=female
    age = np.random.uniform(1, 80)
    sibsp = np.random.randint(0, 10)
    parch = np.random.randint(0, 10)
    fare = np.random.uniform(0, 500)

    user_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
    user_data = preprocess_input(user_data)
    
    prediction = model.predict(user_data)
    result = f"Simulated with parameters: pclass={pclass}, sex={sex}, age={age:.1f}, sibsp={sibsp}, parch={parch}, fare={fare:.2f}\nResult: {'Survived' if prediction[0] == 1 else 'Did not survive'}"
    
    return jsonify(result=result)

@app.route('/scatterplot', methods=['POST'])
def scatterplot():
    # Generate random data for scatterplot
    num_points = 100
    pclass = np.random.choice([1, 2, 3], num_points)
    sex = np.random.choice([0, 1], num_points)  # 0=male, 1=female
    age = np.random.uniform(1, 80, num_points)
    sibsp = np.random.randint(0, 10, num_points)
    parch = np.random.randint(0, 10, num_points)
    fare = np.random.uniform(0, 500, num_points)

    user_data = np.column_stack((pclass, sex, age, sibsp, parch, fare))
    user_data = preprocess_input(user_data)
    
    predictions = model.predict(user_data)
    
    # Get the selected parameter for the scatterplot
    parameter = request.form['parameter']
    
    # Create scatterplot using Plotly
    if parameter == 'age':
        fig = px.scatter(x=age, y=predictions, labels={'x': 'Age', 'y': 'Survived'},
                         title='Scatter plot of Age with Survival Prediction')
    elif parameter == 'fare':
        fig = px.scatter(x=fare, y=predictions, labels={'x': 'Fare', 'y': 'Survived'},
                         title='Scatter plot of Fare with Survival Prediction')
    elif parameter == 'pclass':
        fig = px.scatter(x=pclass, y=predictions, labels={'x': 'Class', 'y': 'Survived'},
                         title='Scatter plot of Class with Survival Prediction')
    elif parameter == 'sex':
        fig = px.scatter(x=sex, y=predictions, labels={'x': 'Sex', 'y': 'Survived'},
                         title='Scatter plot of Sex with Survival Prediction')
    elif parameter == 'sibsp':
        fig = px.scatter(x=sibsp, y=predictions, labels={'x': 'Siblings/Spouses', 'y': 'Survived'},
                         title='Scatter plot of Siblings/Spouses with Survival Prediction')
    elif parameter == 'parch':
        fig = px.scatter(x=parch, y=predictions, labels={'x': 'Parents/Children', 'y': 'Survived'},
                         title='Scatter plot of Parents/Children with Survival Prediction')
    
    # Save plot to the visuals folder
    visuals_folder = 'visuals'
    if not os.path.exists(visuals_folder):
        os.makedirs(visuals_folder)
    fig_path = os.path.join(visuals_folder, 'scatterplot.png')
    fig.write_image(fig_path)
    
    # Cache the image path
    cache.set('scatterplot_path', fig_path)
    
    return jsonify(result="Scatterplot generated")

@app.route('/get_scatterplot')
def get_scatterplot():
    fig_path = cache.get('scatterplot_path')
    if fig_path is None or not os.path.exists(fig_path):
        return "No scatterplot available", 404
    return send_file(fig_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(debug=True)
