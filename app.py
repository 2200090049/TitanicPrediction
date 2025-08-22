"""
Simple Streamlit wrapper for existing Titanic project
Just displays the results of your existing main.py code
"""

import streamlit as st
import sys
import os
import subprocess

from io import StringIO
import contextlib

# Page config
st.set_page_config(
    page_title="🚢 Titanic Survival Analysis",
    page_icon="🚢",
    layout="wide"
)

# Title
st.title("🚢 Titanic Survival Prediction Analysis")
st.markdown("### Results from Machine Learning Analysis")

# Add your existing main.py to the path
sys.path.append('.')


# Function to capture print output
@contextlib.contextmanager
def capture_output():
    old_stdout = sys.stdout
    stdout_buffer = StringIO()
    try:
        sys.stdout = stdout_buffer
        yield stdout_buffer
    finally:
        sys.stdout = old_stdout


# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["📊 Analysis Results", "🤖 Run Analysis", "📈 Visualizations"])

with tab1:
    st.markdown("""
    ## 📋 Project Overview

    This project performs comprehensive analysis on the Titanic dataset:

    - **🔍 Data Exploration**: Statistical analysis and missing value assessment
    - **🧹 Data Cleaning**: Intelligent missing value imputation
    - **🔨 Feature Engineering**: Creation of new predictive features
    - **🤖 Model Training**: Logistic regression for survival prediction
    - **📊 Visualization**: Multiple charts and analysis plots
    - **📈 Evaluation**: Accuracy metrics and feature importance

    ### Key Findings:
    - **Overall Survival Rate**: ~38% of passengers survived
    - **Gender Impact**: Women had 74% survival rate vs men at 19%
    - **Class Matters**: 1st class: 63%, 2nd class: 47%, 3rd class: 24%
    - **Age Factor**: Children had higher survival rates
    - **Model Accuracy**: Achieved 80-82% prediction accuracy
    """)

with tab2:
    st.markdown("## 🚀 Run Complete Analysis")

    if st.button("▶️ Execute Titanic Analysis", type="primary"):
        with st.spinner("🔄 Running complete Titanic survival analysis..."):
            try:
                # Import and run your existing main.py code
                import main

                # Capture the output
                with capture_output() as output:
                    predictor = main.TitanicSurvivalPredictor()
                    predictor.run_complete_analysis()

                # Display the captured output
                st.text_area("📋 Analysis Output", value=output.getvalue(), height=400)

                st.success("✅ Analysis completed successfully!")
                st.info("📁 Check the 'visualizations' folder for generated plots")

            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")

    st.markdown("""
    ### 📊 What happens when you run the analysis:
    1. **Downloads Titanic dataset** automatically
    2. **Performs data exploration** and shows statistics
    3. **Creates visualizations** (saved to visualizations folder)
    4. **Cleans and preprocesses** the data
    5. **Trains logistic regression** model
    6. **Evaluates performance** with accuracy metrics
    7. **Analyzes feature importance**
    """)

with tab3:
    st.markdown("## 📈 Generated Visualizations")

    # Check if visualization files exist and display them
    viz_files = [
        ("exploratory_analysis.png", "🔍 Exploratory Data Analysis"),
        ("model_evaluation.png", "🤖 Model Performance Evaluation"),
        ("feature_analysis.png", "📊 Feature Importance Analysis")
    ]

    for filename, title in viz_files:
        filepath = f"visualizations/{filename}"
        if os.path.exists(filepath):
            st.markdown(f"### {title}")
            st.image(filepath, use_column_width=True)
        else:
            st.info(f"📁 {title} will appear here after running the analysis")

# Sidebar
st.sidebar.markdown("""
## 🧭 Navigation

### 📚 About This Project
This web app runs your existing Titanic survival prediction code without any modifications.

### 🔧 How It Works
- Uses your original `main.py` file
- Displays results in a web interface  
- Shows generated visualizations
- No code changes needed!

### 📊 Dataset
- **Source**: Kaggle Titanic Competition
- **Size**: 891 passengers, 12 features
- **Goal**: Predict survival (0 or 1)

### 🤖 Model
- **Algorithm**: Logistic Regression
- **Features**: 11 engineered features
- **Accuracy**: ~80-82%

### 📁 Files Generated
- `data/titanic.csv`
- `visualizations/*.png`
""")

# Footer
st.markdown("---")
st.markdown("**🚢 Titanic Survival Prediction** | Built with Streamlit | Powered by Machine Learning")