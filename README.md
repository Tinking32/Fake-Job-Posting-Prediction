Fake Job Posting Prediction 👨‍💻
## Objective
To develop and evaluate machine learning classifiers for detecting fraudulent job postings from a highly imbalanced dataset. This project implements an end-to-end pipeline from data preprocessing and feature engineering to model training and performance analysis.

## Technical Workflow & Results
Dataset: The project utilizes a dataset of ~18,000 job postings from a Kaggle competition, characterized by a significant class imbalance between fraudulent and non-fraudulent entries.

Preprocessing: Initial data cleaning involved null value imputation (backward-fill), duplicate removal, and feature consolidation by merging multiple text fields (description, requirements, company_profile) into a single corpus.

Feature Engineering: A ColumnTransformer pipeline was implemented to handle mixed data types:

Textual Data: Transformed using TfidfVectorizer with stop-word removal and max_features=10000.

Categorical Data: Transformed using OneHotEncoder with handle_unknown='ignore' for robustness.

Modeling: Three classification models were trained and evaluated. The class_weight='balanced' parameter was used in all models to counteract the dataset's class imbalance.

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Evaluation: The Random Forest Classifier was the top-performing model, achieving a final accuracy of ~98.4% on the test set. Performance was comprehensively assessed using precision, recall, and F1-score for both classes.

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Jupyter
