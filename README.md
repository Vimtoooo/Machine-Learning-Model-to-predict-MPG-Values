# Machine Learning Model to Predict MPG!

In this project, I will be covering a brief summary of each incremented step, leading to the complete models of prediction and statistical diagrams, such as a Scatter Plot graph. This is also my first M.L. Model build, so along the way, I will be refining and adjusting the program and the documentation for complete precision. Also, feel free to comment on how or what i should include to upgrade on this repository!

I will be utilizing the **Car Information Dataset** found in kaggle:
[car information dataset](https://www.kaggle.com/datasets/tawfikelmetwally/automobile-dataset/data)

For this task, we will utilize the `scikit-learn` library and the coding environment will be within VS Code!

## Libraries:

- `scikit-learn`: For data separation (training and test), training, building these models and predicting their values;
- `pandas`: For data visualization and analysis;
- `numpy`: For the trend line;
- `matplotlib`: For constructing the scatter graph;
- `ipykernel`: Utilization of the `ipynb` files;
- `openpyxl`: Engine for the excel spreadsheet selection and for reading, writing, creating, and modifying Excel files.

## Documentation Steps:

### 1. Load the Data into a DataFrame:

We will load the data by importing `pandas`, addressing the path to the `.xlsx` file and the index of the specific spreadsheet within that same file that we want to retrieve, and then we create a DataFrame.

### 2. Prepare the Data:

We'll prepare the data by separating the df into the `x` and `y` variables, noting that the `y` variable holds the DataFrame of only the **mpg** field, where in the other hand, the `x` stores the entire df, but not including the **mpg** field.

Now, import the `scikit-learn` module to utilize the `train_test_split()` function and pass the appropriate variables to contain these models such as: `x_train`, `x_test`, `y_train` and `y_test`.

> [!NOTE]
> It is fundamental to split the data into different sections so that your results can be reproduced into a more modular representation of our predicted model!

### 3. Begin Constructing the Model:

For the next task, we will make use of other several import statements for when we start constructing, training and predicting the values of **mpg** from the models, and for this reason, we will be using the **regression models**. The main reason is that, with regression models, you can work with **quantitative numerical values**, such as our `y` variable that holds the **mpg field**, meaning that it represents **numerical data**, which will be utilized to product a **numerical output** that will illustrate a **prediction of continuous values (regression)**.

But in the other hand, if we were to deal with **categorical data**, then we would build a **classification model**.

#### Linear Regression:

The **linear regression** is a **statistical technique** used to **find the relationship between variables** (predicting a continuous variable, such as the dependent variable), by modeling the linear relationship between that variable and one or more predictor variables (also referred to as the independent variables).

##### Training the Model:

Make sure that we import the `LinearRegression()` function so that the models can be trained efficiently. However, this will be empty at first, so it its key to use the `fit()` method to pass the required arguments (which will be the `x_train` and the `y_train` models). Down below is the following syntax:

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
```

##### Apply the Predictions:

After training the models, we can smoothly move on to the next phase, where we will **apply the trained models to make their predictions!**
But, throughout the program, it is important to keep naming conventions short, understandable and non-redundant so that we don't mix up these variables.

Wielding with the `predict()` method will allow us to predict both the `x_train` and `x_test` models, storing their values inside adequate variables like `y_lr_train_prediction`.

```python
y_lr_train_prediction = lr.predict(x_train)
y_lr_test_prediction = lr.predict(x_test)
```

- `y`: Contains only the **mpg** field, but not the entire df, while the `x` would be the opposite!
- `lr`: Stands for Linear Regression;
- `train` or `test`: Indicates the type of model to predict/train;

##### Evaluate the Model Performance:

First of all, exhibit both the `y_train` and `y_lr_train_prediction` variables to quickly analyze the pattern of numbers compared to the original `y_train`. The results that have been generated from the `y_lr_train_prediction` variable should **most likely be relatively close to the original values** in the `y_train` variable (the non-trained model). But if the generated prediction highlights a **significant gap between the non-trained model or baseline model and the trained model**, there are other steps that can be made such as **understanding the discrepancy** and **addressing the discrepancy**, but for our case, we won't have to go through this topic.

Import the `mean_squared_error` and `r2_score` methods from the `sklearn.metrics` module to calculate the **mean squared error and the squared correlation coefficient**, the results generated from these metrics will be crucial for model evaluation in regression tasks:

```python
from sklearn.metrics import mean_squared_error, r2_score

# Results for the training set
lr_train_mse = mean_squared_error(y_train, y_lr_train_prediction)
lr_train_r2 = r2_score(y_train, y_lr_train_prediction)

# Results for the test set
lr_test_mse = mean_squared_error(y_test, y_lr_test_prediction)
lr_test_r2 = r2_score(y_test, y_lr_test_prediction)
```

- `mean_squared_error`: Calculates the average of the squared differences between the actual and predicted values, while also being commonly used as a **loss function**during model training;
- `r2_score`: It provides a **relative measure of the model's "goodness of fit"** compared to a simple baseline model (which just predicts the mean of the actual values).

> [!NOTE]
> Why do we create evaluation results for both the training and test models?
> Well, we are not only diagnosing the models, but also addressing the critical model performance issues, if it may be **overfitting or underfitting**, and to ensure that the model can generalize to new, unseen data!

###### Organize the Results:

This part of the documentation is **OPTIONAL** only if you would like to compare each model technique with their corresponding results to the training and testing models, but for best practices, I would recommend doing this complementary step.

We'll extract the results into a DataFrame called `results_df`, storing them in specific fields with adequate names like "Training MSE", "Training R2" and so on. Utilize the `DataFrame()` method from the `pandas` library when instantiating the DF.

```python
df_results = pd.DataFrame(['Linear Regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()

df_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
```

- `transpose()`: Transports all of the data into the same record but in distinct fields, rather than applying them to be stored in the same column;
- `columns`: Adds a name to each column (the order of how you passed the name of columns matter and will impact the exhibition of the results!).

This makes it more convenient to visualize the generated results from each and every statistical technique and model, **allowing easy modifications and insertions** as you continue to create and utilize more techniques. Up next will be the Random Forest!

#### Random Forest Regression:

Slightly similar to the previous statistical technique, the overrated **random Forest Regression** wields with an **ensemble of decision trees to predict a continuous value**. It builds various individual decision trees on random subsets of the data, averaging their predictions to then, product a singular, modular and reliable result than a single tree could provide. It is worth noting that this particular approach will allow to reduce overfitting and variance, greatly improving the model performance!

We will reapply the exact same prior steps of training the model, applying the predictions, evaluating and organizing our results. Furthermore, only altering the naming convention from `lr` to `rf`, meaning the **random forest** model prediction.

```python
# Train the model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# Apply the predictions
y_rf_train_prediction = rf.predict(x_train)
y_rf_test_prediction = rf.predict(x_test)

# Evaluate the model performance by calculating the MSE and r2
# Training set results
rf_train_mse = mean_squared_error(y_train, y_rf_train_prediction)
rf_train_r2 = r2_score(y_train, y_rf_train_prediction)

# Test set results
rf_test_mse = mean_squared_error(y_test, y_rf_test_prediction)
rf_test_r2 = r2_score(y_test, y_rf_test_prediction)

# Build the DF
rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()

# Name each field correctly
rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']
```

##### Important Methods and Optional Keywords:

- `RandomForestRegressor()`: Instantiates the random forest instance to construct the model;
- `max_depth` and `random_state`: Optional keyword arguments that will remain with the same values as the `LinearRegressor()` model to keep the model modular;

### 4. Justify a Model Comparison:

One of the main reasons why we include a model comparison is because of how we can evaluate objectively how each well each model performed during specific tasks, then we can identify the best model for deployment, but also, to understand the **strengths and weaknesses** to determine the optimal resource allocation, which helps determine which model provides the best trade-off between performance and efficiency.

We will concatenate both, the **linear regression** and the **random forest models** into a singular DataFrame for simple data analysis, while also fixing some issues under the way:

```python
df_models = pd.concat([df_results, rf_results], axis=0)

df_models.reset_index(drop=True) # This resets and fixes the index exhibition
```

- `concat()`: Merges two or more DataFrames into a single DF, while the `axis=0` will be row based!
- `reset_index()`: Resets the index column at the left side (indicates the number of records in the DF, in indexes), noting that the `drop=True` will eliminate the `index` column name to be set above the indexing.

#### Model Performance Evaluation:

During the comparison of the evaluation, we may encounter various lesser or rather, greater **gaps of results** when comparing two or multiple records of the same filed, but from different statistical techniques (such as the linear and random forest regressors). The difference in their **generalization performance** on unseen data is often referred to as the **performance gap** or **model performance difference**. Let's break this down into difference segments.

In the summoning of results in specific metrics can most likely vary in value, definition and purpose, so this is one way to judge a model on how well they have performed, based on its value:

##### Metrics where a Smaller Value is Better:

For metrics that are focussed on measuring **error, loss or distance** are optimized by **minimizing their value**, where a result closer to zero is considered better:

- **Mean Squared Error (MSE)**;
- **Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE)**.

In our merged DF, the training MSE for `linear regression` has prompted a slight smaller result compared to the `random forest regression`, with a difference of approximately ~

##### Metrics where a Larger Value is Better:

In the other hand, metrics that measure **goodness-of-fit, accuracy, or explanatory power** are optimized by **maximizing their value**. A result closer to one (or 100%) is better:

- **R-Squared Score**;
- **Accuracy**;
- **Area Under the ROC Curve (AUCROC)**.

##### Comparing the Test and Training set Results:

Verifying both the test and training set results can also be considered, a way to determine how well a model has performed. By analyzing the scores of these sets can help indicate and diagnose the **performance issues**, if it's overfitting or underfitting, ensuring that the model is likely to perform reliably on new, unseen data in the real world.
