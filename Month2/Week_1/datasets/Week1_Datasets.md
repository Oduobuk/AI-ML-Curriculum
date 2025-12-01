# Month 2, Week 1: Datasets for Linear Regression

This week, as we focus on Linear Regression, we'll be looking at datasets where the goal is to predict a continuous numerical value. These datasets typically contain one or more independent variables (features) and a single dependent variable (target) that we want to model.

Here are some classic and commonly used datasets suitable for practicing linear regression:

## 1. Boston Housing Dataset

*   **Description:** This dataset contains information about housing in the Boston area. The goal is to predict the median value of owner-occupied homes (MEDV) based on various features of the neighborhood.
*   **Features:** Includes attributes like crime rate (CRIM), proportion of residential land zoned for lots over 25,000 sq.ft. (ZN), proportion of non-retail business acres per town (INDUS), Charles River dummy variable (CHAS), nitric oxides concentration (NOX), average number of rooms per dwelling (RM), proportion of owner-occupied units built prior to 1940 (AGE), weighted distances to five Boston employment centres (DIS), accessibility to radial highways (RAD), full-value property-tax rate per $10,000 (TAX), pupil-teacher ratio by town (PTRATIO), 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town (B), and % lower status of the population (LSTAT).
*   **Target:** Median value of owner-occupied homes in $1000s (MEDV).
*   **Use Case:** A very popular dataset for practicing multiple linear regression, feature engineering, and understanding the impact of different features on housing prices.

## 2. Advertising Dataset

*   **Description:** This dataset consists of the sales of a product in 200 different markets, along with advertising budgets for three different media: TV, radio, and newspaper.
*   **Features:** Advertising budgets for TV, radio, and newspaper (in thousands of dollars).
*   **Target:** Sales (in thousands of units).
*   **Use Case:** Excellent for understanding the relationship between advertising spending and sales, and for practicing multiple linear regression. It can also be used to explore feature importance and interaction effects.

## 3. California Housing Dataset

*   **Description:** This dataset contains aggregated data from the 1990 California census. The goal is to predict the median house value for California districts.
*   **Features:** Includes attributes like median income, housing median age, average number of rooms, average number of bedrooms, population, average occupancy, latitude, and longitude.
*   **Target:** Median house value.
*   **Use Case:** A larger and more complex dataset than Boston Housing, suitable for more robust linear regression models and exploring geographical influences on housing prices.

## 4. Auto MPG Dataset

*   **Description:** This dataset contains information about cars from the 1970s and 1980s. The goal is to predict the fuel efficiency (miles per gallon) of a car.
*   **Features:** Includes attributes like cylinders, displacement, horsepower, weight, acceleration, model year, origin.
*   **Target:** Miles Per Gallon (MPG).
*   **Use Case:** Good for practicing linear regression and understanding how different car attributes affect fuel efficiency. It also presents opportunities for handling categorical features (like 'origin').

## 5. Simple Synthetic Datasets

*   **Description:** For initial learning and testing, it's often beneficial to create your own simple synthetic datasets. These datasets have a known underlying linear relationship, allowing you to easily verify if your linear regression implementation is working correctly.
*   **Features:** Typically one or two features.
*   **Target:** A continuous numerical value with some added noise.
*   **Use Case:** Perfect for implementing linear regression from scratch, understanding gradient descent, and visualizing the regression line.

These datasets are readily available through libraries like `scikit-learn` or can be found on platforms like Kaggle, making them easy to load and experiment with.
