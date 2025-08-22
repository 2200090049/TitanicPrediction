# 🚢 Titanic Survival Prediction Project

A comprehensive machine learning project that analyzes the Titanic passenger dataset and predicts survival outcomes using logistic regression.

## 📋 Project Overview

This project performs:
- **Exploratory Data Analysis (EDA)** on the Titanic dataset
- **Data cleaning and preprocessing** with missing value imputation
- **Feature engineering** to create meaningful predictors
- **Logistic regression modeling** for survival prediction
- **Model evaluation** with accuracy metrics and visualizations
- **Feature importance analysis** with comprehensive charts

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms
- **Jupyter** - Interactive development

## 📁 Project Structure

```
TitanicPrediction/
├── data/
│   └── titanic.csv                 # Dataset file
├── visualizations/
│   ├── exploratory_analysis.png    # EDA visualizations
│   ├── model_evaluation.png        # Model performance plots
│   └── feature_analysis.png        # Feature importance charts
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   └── model.py
├── main.py                         # Main execution script
├               
└── README.md                       # Project documentation
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- PyCharm IDE (recommended)
- Git (for version control)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/TitanicPrediction.git
   cd TitanicPrediction
   ```



2**Run the project:**
   ```bash
   python main.py
   ```

## 📊 Key Features

### Data Analysis
- **Missing value analysis** and intelligent imputation
- **Statistical summaries** and data exploration
- **Survival rate analysis** by different passenger attributes

### Visualizations
- **Bar charts** showing survival rates by gender, class, and age groups
- **Histograms** for age and fare distributions
- **Correlation heatmaps** to identify feature relationships
- **Confusion matrices** for model performance evaluation

### Model Performance
- **Logistic Regression** with feature scaling
- **Train/Test split** with stratified sampling
- **Accuracy metrics** and classification reports
- **Feature importance** rankings

## 📈 Results

The model achieves approximately **80-82% accuracy** on the test set, with key findings:

### Most Important Features:
1. **Gender (Sex)** - Women had significantly higher survival rates
2. **Passenger Class** - First-class passengers were more likely to survive
3. **Age** - Children and younger passengers had better survival chances
4. **Fare** - Higher fare correlates with better survival odds
5. **Family Size** - Optimal family size improves survival probability

### Key Insights:
- **Women survived at 74% vs men at 19%**
- **1st class: 63% survival, 2nd class: 47%, 3rd class: 24%**
- **Children under 12 had higher survival rates**
- **Passengers with moderate family sizes (2-4) survived more often**

## 🔧 Usage

Run the complete analysis:
```python
from main import TitanicSurvivalPredictor

# Initialize predictor
predictor = TitanicSurvivalPredictor()

# Run complete analysis pipeline
predictor.run_complete_analysis()
```

## 📊 Visualizations Generated

The project creates three main visualization files:

1. **exploratory_analysis.png** - EDA charts including survival rates, demographics
2. **model_evaluation.png** - Confusion matrix and feature importance
3. **feature_analysis.png** - Detailed feature correlation and coefficient analysis




## 📧 Contact

Your Name - your.email@example.com
Project Link: https://github.com/yourusername/TitanicPrediction

## 🙏 Acknowledgments

- Dataset provided by [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Inspiration from the data science community
- Thanks to all contributors and reviewers

---

**⭐ If you found this project helpful, please give it a star!**