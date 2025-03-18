## P03 Model Build - No-Show Prediction App

This project involves building and evaluating machine learning models to predict no-shows at a clinic. Additionally, a Streamlit app has been developed to provide real-time predictions based on clinic appointment data.

### Table of Contents
- Introduction
- Dataset
- Dependencies
- Project Structure
- Notebook Workflow
- Data Loading and Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Results Summary
- Streamlit App
- How to Run
- Usage
- Example Output
- Results
- Future Improvements

### Introduction
This project, a component of the P03 project, seeks to effectively predict if a patient is likely to attend their clinic appointment. The objective is to create machine learning models with high accuracy and reliability, incorporated in an interactive Streamlit web application for ease of use.

### Dataset
The dataset includes clinic appointment information, such as patient demographics, appointment details, and historical no-show records. Key features include:
- **Appointment ID (APPT_ID)**
- **Medical Record Number (MRN)**
- **Clinic name (CLINIC)**
- **Appointment date (APPT_DATE)**
- **Target variable: No-show status**

### Dependencies
Python libraries required to run the notebook and app are:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- streamlit
- pickle

### Project Structure
- `notebook/` - Jupyter notebooks for training and testing models
- `app/` - Streamlit application for real-time predictions
- `models/` - Machine learning models trained
- `data/` - Raw and preprocessed data sets
- `utils/` - Utility scripts and functions

### Notebook Workflow
1. **Data Loading and Preprocessing**
- Load the data set and examine its structure
- Handle missing values, outliers, and categorical variables
- Normalize and scale numeric variables

2. **Feature Engineering**
- Create new features based on domain knowledge (e.g., booking to appointment time)
- One-hot or label encode categorical features

3. **Model Training**
- Train different machine learning models, including Logistic Regression, Random Forest, Gradient Boosting, and SVM
- Use cross-validation to carry out hyperparameter tuning

4. **Model Evaluation**
- Evaluate models on metrics such as Accuracy, Precision, Recall, F1 Score, and AUC
- Compare models on performance and select the best model

5. **Summary of Results**
- Convert evaluation results to a DataFrame for easy comparison
- Select the best model based on AUC and F1 Score

### Streamlit App
The Streamlit app offers the capacity for the selection of a clinic and a range of dates in order to predict no-shows. It takes advantage of the best model during training.

#### How to Run
1. Clone the repo:
```bash
git clone https://github.com/yourusername/no-show-prediction.git
cd no-show-prediction
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the Streamlit app:
```bash
streamlit run app/app.py
```


#### Usage
- Select the clinic from the dropdown list
- Choose the start and end dates for the prediction
- Click the 'Get Predictions' button to view the results
- Optionally download the predictions as a CSV file

#### Example Output
The app displays a table with the columns MRN, APPT_ID, Clinic, Appointment Date, No-Show Probability, and Prediction (Yes/No).

### Results
The results of the model evaluations are displayed in a table, where each model's performance is compared across various metrics. The best model is selected based on the best F1 Score and AUC.

### Future Improvements
- Incorporate additional features to improve model performance
- Experiment with more complex models like XGBoost or deep learning
- Perform feature selection to reduce dimensionality and improve interpretability
- Expand the Streamlit app for comparing models