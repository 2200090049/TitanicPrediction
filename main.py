"""
Titanic Survival Prediction Project
===================================
This project performs exploratory data analysis on the Titanic dataset,
cleans missing data, and builds a logistic regression model to predict survival outcomes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import os
import urllib.request
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')


class TitanicSurvivalPredictor:
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, file_path='data/titanic.csv'):
        """Load the Titanic dataset"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)

            # Download dataset if it doesn't exist
            if not os.path.exists(file_path):
                print("Downloading Titanic dataset...")
                url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                urllib.request.urlretrieve(url, file_path)
                print("Dataset downloaded successfully!")

            self.df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully! Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def explore_data(self):
        """Perform initial data exploration"""
        print("\n" + "=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)

        print("\n1. Dataset Info:")
        print("-" * 20)
        print(self.df.info())

        print("\n2. First 5 rows:")
        print("-" * 20)
        print(self.df.head())

        print("\n3. Statistical Summary:")
        print("-" * 20)
        print(self.df.describe())

        print("\n4. Missing Values:")
        print("-" * 20)
        missing_data = self.df.isnull().sum()
        missing_percent = 100 * missing_data / len(self.df)
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing Percentage': missing_percent
        })
        print(missing_df[missing_df['Missing Count'] > 0])

        print("\n5. Survival Rate:")
        print("-" * 20)
        survival_rate = self.df['Survived'].value_counts()
        print(f"Survived: {survival_rate[1]} ({survival_rate[1] / len(self.df) * 100:.1f}%)")
        print(f"Did not survive: {survival_rate[0]} ({survival_rate[0] / len(self.df) * 100:.1f}%)")

    def visualize_data(self):
        """Create visualizations for exploratory data analysis"""
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATIONS")
        print("=" * 50)

        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)

        # Set style for better looking plots
        plt.style.use('seaborn-v0_8')

        # 1. Survival Rate Bar Chart
        plt.figure(figsize=(15, 12))

        plt.subplot(2, 3, 1)
        survival_counts = self.df['Survived'].value_counts()
        bars = plt.bar(['Did not survive', 'Survived'], survival_counts.values,
                       color=['red', 'green'], alpha=0.7)
        plt.title('Overall Survival Rate', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Passengers')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

        # 2. Survival by Gender
        plt.subplot(2, 3, 2)
        gender_survival = pd.crosstab(self.df['Sex'], self.df['Survived'])
        gender_survival.plot(kind='bar', color=['red', 'green'], alpha=0.7)
        plt.title('Survival Rate by Gender', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Passengers')
        plt.legend(['Did not survive', 'Survived'])
        plt.xticks(rotation=0)

        # 3. Survival by Passenger Class
        plt.subplot(2, 3, 3)
        class_survival = pd.crosstab(self.df['Pclass'], self.df['Survived'])
        class_survival.plot(kind='bar', color=['red', 'green'], alpha=0.7)
        plt.title('Survival Rate by Passenger Class', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Passengers')
        plt.xlabel('Passenger Class')
        plt.legend(['Did not survive', 'Survived'])
        plt.xticks(rotation=0)

        # 4. Age Distribution
        plt.subplot(2, 3, 4)
        plt.hist(self.df[self.df['Survived'] == 1]['Age'].dropna(),
                 bins=30, alpha=0.7, label='Survived', color='green')
        plt.hist(self.df[self.df['Survived'] == 0]['Age'].dropna(),
                 bins=30, alpha=0.7, label='Did not survive', color='red')
        plt.title('Age Distribution by Survival', fontsize=14, fontweight='bold')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        plt.legend()

        # 5. Fare Distribution
        plt.subplot(2, 3, 5)
        plt.hist(self.df[self.df['Survived'] == 1]['Fare'].dropna(),
                 bins=30, alpha=0.7, label='Survived', color='green')
        plt.hist(self.df[self.df['Survived'] == 0]['Fare'].dropna(),
                 bins=30, alpha=0.7, label='Did not survive', color='red')
        plt.title('Fare Distribution by Survival', fontsize=14, fontweight='bold')
        plt.xlabel('Fare')
        plt.ylabel('Frequency')
        plt.legend()

        # 6. Correlation Heatmap
        plt.subplot(2, 3, 6)
        # Prepare numerical data for correlation
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5)
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('visualizations/exploratory_analysis.png', dpi=300, bbox_inches='tight')
        #plt.show()

        print("✅ Visualizations saved to 'visualizations/exploratory_analysis.png'")

    def clean_and_preprocess_data(self):
        """Clean missing data and preprocess features"""
        print("\n" + "=" * 50)
        print("DATA CLEANING & PREPROCESSING")
        print("=" * 50)

        # Make a copy of the original data
        self.df_processed = self.df.copy()

        # 1. Handle missing values
        print("\n1. Handling missing values...")

        # Fill missing Age with median age by Pclass and Sex
        age_median = self.df_processed.groupby(['Pclass', 'Sex'])['Age'].median()
        for pclass in [1, 2, 3]:
            for sex in ['male', 'female']:
                mask = (self.df_processed['Pclass'] == pclass) & \
                       (self.df_processed['Sex'] == sex) & \
                       (self.df_processed['Age'].isnull())
                if mask.any():
                    median_age = age_median.get((pclass, sex))
                    if pd.notna(median_age):
                        self.df_processed.loc[mask, 'Age'] = median_age

        # Fill remaining missing ages with overall median
        self.df_processed['Age'].fillna(self.df_processed['Age'].median(), inplace=True)

        # Fill missing Embarked with mode
        self.df_processed['Embarked'].fillna(self.df_processed['Embarked'].mode()[0], inplace=True)

        # Fill missing Fare with median fare by Pclass
        self.df_processed['Fare'].fillna(
            self.df_processed.groupby('Pclass')['Fare'].transform('median'), inplace=True
        )

        # Drop Cabin column due to too many missing values
        self.df_processed.drop('Cabin', axis=1, inplace=True)

        print(f"✅ Missing values handled. Remaining null values: {self.df_processed.isnull().sum().sum()}")

        # 2. Feature Engineering
        print("\n2. Feature Engineering...")

        # Create family size feature
        self.df_processed['FamilySize'] = self.df_processed['SibSp'] + self.df_processed['Parch'] + 1

        # Create age groups
        self.df_processed['AgeGroup'] = pd.cut(self.df_processed['Age'],
                                               bins=[0, 12, 18, 35, 60, 100],
                                               labels=['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior'])

        # Create fare groups
        self.df_processed['FareGroup'] = pd.qcut(self.df_processed['Fare'], q=4,
                                                 labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

        # Create title feature from Name
        self.df_processed['Title'] = self.df_processed['Name'].str.extract(' ([A-Za-z]+)\\.', expand=False)

        # Group rare titles
        rare_titles = self.df_processed['Title'].value_counts()[self.df_processed['Title'].value_counts() < 10].index
        self.df_processed['Title'] = self.df_processed['Title'].replace(rare_titles, 'Other')

        print("✅ New features created: FamilySize, AgeGroup, FareGroup, Title")

        # 3. Encode categorical variables
        print("\n3. Encoding categorical variables...")

        categorical_columns = ['Sex', 'Embarked', 'AgeGroup', 'FareGroup', 'Title']

        for col in categorical_columns:
            le = LabelEncoder()
            self.df_processed[col + '_encoded'] = le.fit_transform(self.df_processed[col].astype(str))
            self.label_encoders[col] = le

        print("✅ Categorical variables encoded")

        # 4. Select features for modeling
        feature_columns = [
            'Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 'Fare',
            'Embarked_encoded', 'FamilySize', 'AgeGroup_encoded',
            'FareGroup_encoded', 'Title_encoded'
        ]

        self.X = self.df_processed[feature_columns]
        self.y = self.df_processed['Survived']

        print(f"✅ Feature matrix prepared with {self.X.shape[1]} features")
        print(f"Features used: {feature_columns}")

    def train_model(self):
        """Train logistic regression model"""
        print("\n" + "=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")

        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # Train logistic regression model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(self.X_train_scaled, self.y_train)

        print("✅ Logistic Regression model trained successfully")

    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION")
        print("=" * 50)

        # Make predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)

        # Calculate accuracies
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)

        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy * 100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")

        # Classification report
        print("\nClassification Report:")
        print("-" * 30)
        print(classification_report(self.y_test, y_test_pred))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_test_pred)

        plt.figure(figsize=(12, 5))

        # Plot confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Did not survive', 'Survived'],
                    yticklabels=['Did not survive', 'Survived'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Feature importance
        plt.subplot(1, 2, 2)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': abs(self.model.coef_[0])
        }).sort_values('importance', ascending=True)

        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.title('Feature Importance (Absolute Coefficients)', fontsize=14, fontweight='bold')
        plt.xlabel('Absolute Coefficient Value')

        plt.tight_layout()
        plt.savefig('visualizations/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Model evaluation plots saved to 'visualizations/model_evaluation.png'")

        return test_accuracy

    def feature_analysis(self):
        """Analyze feature importance and relationships"""
        print("\n" + "=" * 50)
        print("FEATURE ANALYSIS")
        print("=" * 50)

        plt.figure(figsize=(15, 10))

        # 1. Feature Importance Bar Chart
        plt.subplot(2, 2, 1)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)

        bars = plt.bar(range(len(feature_importance)), feature_importance['importance'],
                       color='skyblue', alpha=0.8)
        plt.title('Feature Importance Rankings', fontsize=14, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Absolute Coefficient Value')
        plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=8)

        # 2. Survival Rate by Top Features
        plt.subplot(2, 2, 2)
        top_features = feature_importance.head(3)['feature'].tolist()
        if 'Sex_encoded' in self.X.columns:
            survival_by_sex = self.df_processed.groupby('Sex')['Survived'].mean()
            bars = plt.bar(survival_by_sex.index, survival_by_sex.values,
                           color=['lightcoral', 'lightgreen'], alpha=0.8)
            plt.title('Survival Rate by Gender', fontsize=14, fontweight='bold')
            plt.ylabel('Survival Rate')
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.3f}', ha='center', va='bottom')

        # 3. Correlation between top numerical features and survival
        plt.subplot(2, 2, 3)
        numerical_features = ['Age', 'Fare', 'FamilySize', 'Pclass']
        correlations = []
        for feature in numerical_features:
            if feature in self.df_processed.columns:
                corr = self.df_processed[feature].corr(self.df_processed['Survived'])
                correlations.append(corr)

        colors = ['red' if x < 0 else 'green' for x in correlations]
        bars = plt.bar(numerical_features, correlations, color=colors, alpha=0.7)
        plt.title('Feature Correlation with Survival', fontsize=14, fontweight='bold')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom' if height > 0 else 'top')

        # 4. Model Coefficients
        plt.subplot(2, 2, 4)
        coefficients = pd.DataFrame({
            'feature': self.X.columns,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient')

        colors = ['red' if x < 0 else 'green' for x in coefficients['coefficient']]
        plt.barh(range(len(coefficients)), coefficients['coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(coefficients)), coefficients['feature'])
        plt.title('Logistic Regression Coefficients', fontsize=14, fontweight='bold')
        plt.xlabel('Coefficient Value')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Feature analysis plots saved to 'visualizations/feature_analysis.png'")

        # Print top features
        print(f"\nTop 5 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
            print(f"{i}. {row['feature']}: {row['importance']:.4f}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚢 TITANIC SURVIVAL PREDICTION PROJECT")
        print("=" * 60)

        # Load data
        if not self.load_data():
            return

        # Explore data
        self.explore_data()

        # Create visualizations
        self.visualize_data()

        # Clean and preprocess
        self.clean_and_preprocess_data()

        # Train model
        self.train_model()

        # Evaluate model
        accuracy = self.evaluate_model()

        # Feature analysis
        self.feature_analysis()

        print("\n" + "=" * 60)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print(f"📊 Final Model Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print("📁 All visualizations saved in 'visualizations/' folder")
        print("=" * 60)


# Main execution
if __name__ == "__main__":
    # Create and run the Titanic survival predictor
    predictor = TitanicSurvivalPredictor()
    predictor.run_complete_analysis()