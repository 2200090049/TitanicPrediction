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
        """Create beautiful visualizations for exploratory data analysis"""
        print("\n" + "=" * 50)
        print("CREATING VISUALIZATIONS")
        print("=" * 50)

        # Create visualizations directory
        os.makedirs('visualizations', exist_ok=True)

        # Set style for professional looking plots
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with multiple subplots - better spacing
        fig = plt.figure(figsize=(20, 24))
        fig.suptitle('🚢 Titanic Dataset - Exploratory Data Analysis',
                     fontsize=24, fontweight='bold', y=0.98)

        # 1. Overall Survival Rate with improved styling
        ax1 = plt.subplot(4, 3, 1)
        survival_counts = self.df['Survived'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']  # Modern color palette
        wedges, texts, autotexts = ax1.pie(survival_counts.values,
                                          labels=['Did not survive', 'Survived'],
                                          colors=colors, autopct='%1.1f%%',
                                          startangle=90, explode=(0.05, 0.05),
                                          shadow=True)
        ax1.set_title('Overall Survival Rate', fontsize=16, fontweight='bold', pad=20)
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)

        # 2. Survival by Gender - Horizontal Bar Chart
        ax2 = plt.subplot(4, 3, 2)
        gender_survival = self.df.groupby('Sex')['Survived'].agg(['count', 'sum']).reset_index()
        gender_survival['survival_rate'] = gender_survival['sum'] / gender_survival['count'] * 100

        bars = ax2.barh(gender_survival['Sex'], gender_survival['survival_rate'],
                       color=['#FF9999', '#66B2FF'], alpha=0.8, height=0.6)
        ax2.set_title('Survival Rate by Gender (%)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Survival Percentage')
        ax2.grid(axis='x', alpha=0.3)

        # Add percentage labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', fontweight='bold')

        # 3. Survival by Passenger Class - Enhanced
        ax3 = plt.subplot(4, 3, 3)
        class_survival = pd.crosstab(self.df['Pclass'], self.df['Survived'], normalize='index') * 100
        class_survival.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'],
                           alpha=0.8, width=0.7)
        ax3.set_title('Survival Rate by Passenger Class', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Survival Percentage')
        ax3.set_xlabel('Passenger Class')
        ax3.legend(['Did not survive', 'Survived'], loc='upper right')
        ax3.set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
        ax3.grid(axis='y', alpha=0.3)

        # 4. Age Distribution by Survival - Violin Plot
        ax4 = plt.subplot(4, 3, 4)
        age_data = []
        age_labels = []
        for survival in [0, 1]:
            ages = self.df[self.df['Survived'] == survival]['Age'].dropna()
            age_data.append(ages)
            age_labels.append('Did not survive' if survival == 0 else 'Survived')

        violin_parts = ax4.violinplot(age_data, positions=[0, 1], showmeans=True, showmedians=True)
        ax4.set_title('Age Distribution by Survival', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Age')
        ax4.set_xticks([0, 1])
        ax4.set_xticklabels(age_labels)
        ax4.grid(axis='y', alpha=0.3)

        # Color the violin plots
        colors = ['#FF6B6B', '#4ECDC4']
        for pc, color in zip(violin_parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # 5. Fare Distribution - Box Plot
        ax5 = plt.subplot(4, 3, 5)
        fare_data = [self.df[self.df['Survived'] == i]['Fare'].dropna() for i in [0, 1]]
        box_plot = ax5.boxplot(fare_data, labels=['Did not survive', 'Survived'],
                              patch_artist=True, showfliers=False)
        ax5.set_title('Fare Distribution by Survival', fontsize=16, fontweight='bold')
        ax5.set_ylabel('Fare (£)')
        ax5.grid(axis='y', alpha=0.3)

        # Color the box plots
        colors = ['#FF6B6B', '#4ECDC4']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # 6. Embarked Port Analysis
        ax6 = plt.subplot(4, 3, 6)
        embark_survival = pd.crosstab(self.df['Embarked'], self.df['Survived'], normalize='index') * 100
        embark_survival.plot(kind='bar', ax=ax6, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax6.set_title('Survival Rate by Embarked Port', fontsize=16, fontweight='bold')
        ax6.set_ylabel('Survival Percentage')
        ax6.set_xlabel('Embarked Port')
        ax6.legend(['Did not survive', 'Survived'])
        ax6.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'], rotation=45)
        ax6.grid(axis='y', alpha=0.3)

        # 7. Family Size Impact
        ax7 = plt.subplot(4, 3, 7)
        family_size = self.df['SibSp'] + self.df['Parch'] + 1
        family_survival = pd.crosstab(family_size, self.df['Survived'], normalize='index') * 100
        family_survival[1].plot(kind='line', ax=ax7, marker='o', linewidth=3,
                               markersize=8, color='#4ECDC4')
        ax7.set_title('Survival Rate by Family Size', fontsize=16, fontweight='bold')
        ax7.set_ylabel('Survival Percentage')
        ax7.set_xlabel('Family Size')
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(0, 100)

        # 8. Age Groups Analysis
        ax8 = plt.subplot(4, 3, 8)
        age_groups = pd.cut(self.df['Age'], bins=[0, 12, 18, 30, 50, 80],
                           labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
        age_group_survival = pd.crosstab(age_groups, self.df['Survived'], normalize='index') * 100
        age_group_survival[1].plot(kind='bar', ax=ax8, color='#4ECDC4', alpha=0.8)
        ax8.set_title('Survival Rate by Age Group', fontsize=16, fontweight='bold')
        ax8.set_ylabel('Survival Percentage')
        ax8.set_xlabel('Age Group')
        ax8.set_xticklabels(ax8.get_xticklabels(), rotation=45)
        ax8.grid(axis='y', alpha=0.3)

        # 9. Correlation Heatmap - Enhanced
        ax9 = plt.subplot(4, 3, 9)
        numeric_df = self.df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Show only lower triangle
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                    square=True, linewidths=0.5, ax=ax9, fmt='.2f')
        ax9.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')

        # 10. Survival by Gender and Class
        ax10 = plt.subplot(4, 3, 10)
        gender_class = pd.crosstab([self.df['Sex'], self.df['Pclass']],
                                   self.df['Survived'], normalize='index') * 100
        gender_class[1].plot(kind='bar', ax=ax10, color='#4ECDC4', alpha=0.8)
        ax10.set_title('Survival Rate by Gender & Class', fontsize=16, fontweight='bold')
        ax10.set_ylabel('Survival Percentage')
        ax10.set_xlabel('Gender & Class')
        ax10.set_xticklabels([f'{gender}\nClass {pclass}' for gender, pclass in gender_class.index],
                            rotation=45, ha='right')
        ax10.grid(axis='y', alpha=0.3)

        # 11. Missing Data Analysis
        ax11 = plt.subplot(4, 3, 11)
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=True)
        if len(missing_data) > 0:
            bars = ax11.barh(range(len(missing_data)), missing_data.values,
                            color='#FF6B6B', alpha=0.8)
            ax11.set_yticks(range(len(missing_data)))
            ax11.set_yticklabels(missing_data.index)
            ax11.set_title('Missing Values by Feature', fontsize=16, fontweight='bold')
            ax11.set_xlabel('Number of Missing Values')
            ax11.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax11.text(width + 5, bar.get_y() + bar.get_height()/2,
                         f'{int(width)}', ha='left', va='center', fontweight='bold')

        # 12. Fare vs Age Scatter Plot
        ax12 = plt.subplot(4, 3, 12)
        survived = self.df[self.df['Survived'] == 1]
        not_survived = self.df[self.df['Survived'] == 0]

        ax12.scatter(not_survived['Age'], not_survived['Fare'],
                    alpha=0.6, c='#FF6B6B', label='Did not survive', s=30)
        ax12.scatter(survived['Age'], survived['Fare'],
                    alpha=0.6, c='#4ECDC4', label='Survived', s=30)
        ax12.set_title('Fare vs Age by Survival', fontsize=16, fontweight='bold')
        ax12.set_xlabel('Age')
        ax12.set_ylabel('Fare (£)')
        ax12.legend()
        ax12.grid(True, alpha=0.3)

        # Adjust layout to prevent overlapping
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig('visualizations/exploratory_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()  # Close the figure to free memory

        print("✅ Enhanced visualizations saved to 'visualizations/exploratory_analysis.png'")

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

        plt.figure(figsize=(15, 6))

        # Plot confusion matrix
        plt.subplot(1, 2, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Did not survive', 'Survived'],
                    yticklabels=['Did not survive', 'Survived'])
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
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
        plt.title('Feature Importance (Absolute Coefficients)', fontsize=16, fontweight='bold')
        plt.xlabel('Absolute Coefficient Value')

        plt.tight_layout()
        plt.savefig('visualizations/model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Model evaluation plots saved to 'visualizations/model_evaluation.png'")

        return test_accuracy

    def feature_analysis(self):
        """Analyze feature importance and relationships"""
        print("\n" + "=" * 50)
        print("FEATURE ANALYSIS")
        print("=" * 50)

        plt.figure(figsize=(18, 12))

        # 1. Feature Importance Bar Chart
        plt.subplot(2, 2, 1)
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)

        bars = plt.bar(range(len(feature_importance)), feature_importance['importance'],
                       color='skyblue', alpha=0.8)
        plt.title('Feature Importance Rankings', fontsize=16, fontweight='bold')
        plt.xlabel('Features')
        plt.ylabel('Absolute Coefficient Value')
        plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=45)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        # 2. Survival Rate by Top Features
        plt.subplot(2, 2, 2)
        if 'Sex_encoded' in self.X.columns:
            survival_by_sex = self.df_processed.groupby('Sex')['Survived'].mean()
            bars = plt.bar(survival_by_sex.index, survival_by_sex.values,
                           color=['lightcoral', 'lightgreen'], alpha=0.8)
            plt.title('Survival Rate by Gender', fontsize=16, fontweight='bold')
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
        plt.title('Feature Correlation with Survival', fontsize=16, fontweight='bold')
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
        plt.title('Logistic Regression Coefficients', fontsize=16, fontweight='bold')
        plt.xlabel('Coefficient Value')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig('visualizations/feature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

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