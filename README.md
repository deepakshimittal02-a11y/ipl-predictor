# 🏏 IPL Match Win Predictor

A machine learning web app that predicts the winner of an IPL match
based on historical data from 2008 to 2026.

Built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-Latest-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🤖 What It Does
- Predicts IPL match winner based on teams, venue, and toss
- Shows win probability percentage for both teams
- Displays match insights like venue win rate and recent form
- Trained on 1169 IPL matches (2008-2026)

## 🧠 ML Pipeline

### Data
- Raw IPL match data cleaned and preprocessed
- Feature engineering: venue win rate, recent form
- Label encoding for teams, venues, toss decision

### Models Compared
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 74.79% |
| Voting Classifier | 71.37% |
| Gradient Boosting | 70.09% |
| Random Forest | 68.80% |
| XGBoost | 65.81% |

Best Model: Logistic Regression with 74.79% accuracy

### Key Insight
Feature engineering improved accuracy from 54% to 74% — a 20% jump!
Better data beats better model.

## ⚙️ How to Run

1. Clone the repository
   git clone https://github.com/deepakshimittal02-a11y/ipl-predictor.git
   cd ipl-predictor

2. Install dependencies
   pip install -r requirements.txt

3. Run the app
   streamlit run app.py

4. Open browser at http://localhost:8501

## 📦 Project Structure
   ipl-predictor/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   ├── .gitignore
   ├── data/
   │   └── processed_matches.csv
   ├── model/
   │   ├── ipl_model.pkl
   │   ├── team_encoder.pkl
   │   ├── toss_encoder.pkl
   │   └── venue_encoder.pkl
   └── notebooks/
       ├── 01_eda.ipynb
       ├── 02_preprocessing.ipynb
       └── 03_model_training.ipynb

## 🛠️ Tech Stack
- Python 3
- Streamlit (Web App)
- Scikit-learn (ML Models)
- XGBoost
- Pandas (Data Processing)
- Pickle (Model Saving)

## 🧠 What I Learned
- End-to-end ML project development
- Feature engineering for sports data
- Comparing multiple ML models
- Building and deploying ML web apps with Streamlit
- Label encoding and data preprocessing

## 👤 Author
**Deepakshi** — [GitHub](https://github.com/deepakshimittal02-a11y)

*Personal ML Project*