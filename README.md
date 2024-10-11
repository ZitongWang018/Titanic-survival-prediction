# Titanic Survival Prediction Project

Result preview：
![e19192185bae9505c52d895d698a401](https://github.com/user-attachments/assets/e5bd480e-3d7a-47bc-b6d6-85c280aa9cb1)

## Introduction

This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset used is the famous Titanic dataset provided by Kaggle, containing information about passengers such as age, gender, class, and other relevant features. The goal is to build a predictive model that can determine the likelihood of survival for each passenger based on these features.

## Dataset

The dataset used in this project consists of two CSV files:

- **train.csv**: This file contains data for training the model, including features and the target variable (`Survived`).
- **test.csv**: This file is used for testing the model's predictions, containing the same features without the target variable.

The key features in the dataset are:

- **Pclass**: Ticket class (1st, 2nd, 3rd)
- **Name**: Name of the passenger
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **Parch**: Number of parents/children aboard
- **Fare**: Ticket fare
- **Embarked**: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

## Project Structure

The project is structured as follows:

1. **Data Loading**: The training and testing datasets are loaded using Pandas.
2. **Data Preprocessing**: Unnecessary columns such as `Ticket` and `Cabin` are dropped to simplify the model. Missing values are handled by filling them with appropriate values, such as median or mode, to avoid data leakage. Categorical features like `Sex` and `Embarked` are converted into numerical values using one-hot encoding, and numerical features are standardized.
3. **Exploratory Data Analysis (EDA)**: Visualization is performed using Seaborn and Matplotlib to understand the distribution of the data, relationships between features, and how different features affect survival.
4. **Feature Engineering**: Several new features are engineered to enhance model performance. For example:
   - **Age Binning**: Grouping `Age` into bins to reduce noise and help the model better capture relationships.
   - **Family Size**: Creating a new feature by combining `SibSp` and `Parch` to represent family size aboard the Titanic.
   - **IsAlone**: A binary feature indicating whether a passenger was alone or not.
   - **Title Extraction**: Extracting titles (e.g., Mr, Mrs, Miss) from passenger names to capture social status and its impact on survival.

5. **Model Training**: Multiple machine learning models are trained, including:
   - **Logistic Regression**: A baseline model for binary classification.
   - **Random Forest**: A tree-based model that captures non-linear relationships.
   - **Support Vector Machine (SVM)**: A model to test complex decision boundaries.

   The models are evaluated through cross-validation to ensure robustness and generalization. A full pipeline was created using scikit-learn to integrate data preprocessing, feature transformation, and model training into a single workflow.

6. **Model Evaluation**: The models are evaluated using various metrics including accuracy, precision, recall, and F1 score. Hyperparameter tuning is performed using GridSearchCV to optimize the model performance. The Random Forest model, with tuned hyperparameters, achieved an F1 score of 0.79 on the validation dataset.

7. **Prediction**: The selected model is used to make predictions on the test dataset. The pipeline ensures that all preprocessing steps are consistently applied to the test data before predictions are made.

## Dependencies

To run this project, you need the following Python libraries:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

You can install these dependencies using the following command:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## How to Run

1. **Clone the Repository**: Clone this project repository to your local machine.
2. **Install Dependencies**: Make sure all the required libraries are installed.
3. **Run the Notebook**: Open the Jupyter Notebook file (`titanic-survival-prediction.ipynb`) and run all cells to see the results.
4. **Make Predictions**: Use the trained model to make predictions on the test dataset.

## Results

The final model achieves an accuracy of approximately XX% on the training data. Feature importance analysis shows that `Sex`, `Pclass`, and `Age` are the most influential features for predicting survival. Engineered features like `Family Size` and `Title` also significantly contributed to improving model performance.

## Future Improvements

- **Hyperparameter Tuning**: Further optimization of model hyperparameters can improve performance.
- **Additional Features**: Incorporating additional features like social-economic status could provide better insights.
- **Ensemble Methods**: Using ensemble techniques such as bagging or boosting could enhance model accuracy.

## Acknowledgments

This project is based on the Titanic dataset provided by [Kaggle](https://www.kaggle.com/c/titanic). Special thanks to the Kaggle community for their insights and contributions.

## License

This project is licensed under the MIT License.
