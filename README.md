# Titanic Survival Prediction

This project aims to predict the survival of Titanic passengers using some basic machine learning algorithms. The dataset used for this project is the famous Titanic dataset from Kaggle, which contains information about the passengers such as age, gender, class, and more.

## Project Structure

- `models/`: Saved machine learning models.
- `scripts/`: Python scripts for data preprocessing and model training.
- `README.md`: Project overview and instructions.

## Dataset

The dataset used in this project is available on [Kaggle](https://www.kaggle.com/c/titanic/data). It includes the following features:

- `PassengerId`: Unique ID for each passenger
- `Survived`: Survival status (0 = No, 1 = Yes)
- `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
- `Name`: Name of the passenger
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Ticket`: Ticket number
- `Fare`: Fare paid by the passenger
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/Titanic_survival.git
    cd Titanic_survival
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Preprocess the data:
    ```sh
    python scripts/preprocess_data.py
    ```

2. Train the model:
    ```sh
    python scripts/train_model.py
    ```

3. Make predictions:
    ```sh
    python scripts/predict.py
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Kaggle](https://www.kaggle.com/c/titanic) for providing the dataset.
- The open-source community for their valuable tools and libraries.
