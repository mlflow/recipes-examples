# Binary classification: Is this bottle of wine red or white?
This is the root directory for an example project for the
[MLflow Classification Recipe](https://mlflow.org/docs/latest/recipes.html#classification-recipe).
Follow the instructions [here](../README.md) to set up your environment first,
then use this directory to create a classifier and evaluate its performance,
all out of box!

In this example, we demonstrate how to use MLflow Recipes to build a model to predict whether
a bottle of wine is red or white based on the wine's physicochemical properties.
The example uses a dataset from the UCI Machine Learning Repository,
presented in Modeling wine preferences by data mining from physicochemical properties
[Cortez et al., 2009](https://www.sciencedirect.com/science/article/abs/pii/S0167923609001377).

In this [notebook](notebooks/jupyter.ipynb) ([the Databricks version](notebooks/databricks.py)),
we show how to build and evaluate a very simple classifier step by step,
following the best practices of machine learning engineering.
By the end of this example,
you will learn how to use MLflow Recipes to
- Ingest the raw source data.
- Splits the dataset into training/validation/test.
- Create an identity transformer and transform the dataset.
- Train a linear model (classifier) to tell if a bottle of wine is red.
- Evaluate the trained model, and improve it by iterating through the `transform` and `train` steps.
- Register the model for production inference.

All of these can be done with Jupyter notebook or on the Databricks environment.
Finally, challenge yourself to build a better model. Try the following:
- Find a better data source with more training data and more raw feature columns.
- Clean the dataset to make it less noisy.
- Find better feature transformations.
- Fine tune the hyperparameters of the model.
