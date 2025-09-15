# Housing Price Prediction Project

## Overview
In this project, you'll apply your knowledge of linear regression to build a model that predicts house prices based on various features. You'll work with a synthetic dataset containing information about houses, including their size, number of bedrooms, location, and other relevant features.

## Dataset
The dataset is provided in the `datasets/` directory:
- `housing.db`: SQLite database with housing data
- `housing_data_extended.csv`: CSV version of the dataset

The dataset contains the following features:
- `price`: Sale price of the house (target variable)
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `sqft_living`: Square footage of the home
- `sqft_lot`: Square footage of the lot
- `floors`: Number of floors
- `waterfront`: Whether the property has a waterfront view (0/1)
- `view`: Quality of the view (0-4 scale)
- `condition`: Overall condition of the house (1-5 scale)
- `grade`: Overall grade given to the housing unit (1-13 scale)
- `sqft_above`: Square footage of house apart from basement
- `sqft_basement`: Square footage of the basement
- `yr_built`: Year the house was built
- `yr_renovated`: Year the house was last renovated (0 if never)
- `zipcode`: Zip code of the house
- `lat`: Latitude coordinate
- `long`: Longitude coordinate

## Project Requirements

### 1. Data Exploration and Preprocessing (20%)
- Load and explore the dataset
- Handle missing values (if any)
- Perform feature engineering (create new features if needed)
- Handle categorical variables appropriately
- Split the data into training and test sets

### 2. Model Building (40%)
- Implement a simple linear regression model from scratch
- Implement a multiple linear regression model from scratch
- Use scikit-learn's implementation to compare results
- Implement polynomial regression to capture non-linear relationships
- Apply regularization techniques (Ridge and Lasso)

### 3. Model Evaluation (20%)
- Evaluate models using appropriate metrics (R², MSE, RMSE)
- Analyze the bias-variance tradeoff
- Use learning curves to diagnose model performance
- Compare different models and select the best one

### 4. Feature Importance and Interpretation (10%)
- Identify the most important features
- Interpret the model coefficients
- Discuss the business implications of your findings

### 5. Documentation and Code Quality (10%)
- Write clean, well-documented code
- Include comments explaining your approach
- Create visualizations to support your analysis

## Deliverables
1. Jupyter notebook containing your analysis and code
2. Python scripts for your implementations
3. A short report (2-3 pages) summarizing your findings
4. Presentation slides (5-7 slides) for a 10-minute presentation

## Evaluation Criteria
- **Code Quality (20%)**: Clean, efficient, and well-documented code
- **Methodology (30%)**: Appropriate use of techniques and algorithms
- **Analysis (30%)**: Depth of analysis and insights
- **Presentation (20%)**: Clarity and professionalism of report and presentation

## Getting Started
1. Create a new directory for your project
2. Set up a virtual environment
3. Install the required packages from `requirements.txt`
4. Start by exploring the data and planning your approach

## Tips
- Start with simple models before moving to complex ones
- Visualize your data at each step
- Document your thought process and decisions
- Use version control (Git) to track your changes

## Submission
Submit your work as a zip file containing all your code, data, and documentation by the due date.
