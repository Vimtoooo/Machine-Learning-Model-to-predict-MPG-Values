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

Make sure that we import the `LinearRegression()` function so that the models can be trained efficiently. However, this will be empty at first, so it its key to use the `fit()` method to pass the required arguments (which will be the `x_train` and the `y_train` models). Down below is the following syntax:

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)
```

After training the models, we can smoothly move on to the next phase, where we will **apply the trained models to make their predictions!**
But, throughout the program, it is important to keep naming conventions short, understandable and non-redundant so that we don't mix up these variables.

Wielding with the `predict()` method will allow us to predict both the `x_train` and `x_test` models, storing their values inside adequate variables like `y_lr_train_prediction`.

- `lr`: Stands for Linear Regression;
- `train` or `test`: Indicates the type of model to predict/train;
- `y`: Contains only the **mpg** field, but not the entire df, while the `x` would be the opposite!
