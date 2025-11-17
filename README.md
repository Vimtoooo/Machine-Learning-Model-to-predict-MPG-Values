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

This part of the documentation is **OPTIONAL** if you would like to compare each model technique with their corresponding results to the training and testing models, but for best practices, I would recommend doing this complementary step.
