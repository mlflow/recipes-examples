# Predict the NYC taxi fare with an ML regressor
This is the root directory for an example project for the
[MLflow Regression Recipe](https://mlflow.org/docs/latest/recipes.html#regression-recipe).
Follow the instructions [here](../README.md) to set up your environment first,
then use this directory to create a linear regressor and evaluate its performance,
all out of box!

In this example, we demonstrate how to use MLflow Recipes
to predict the fare amount for a taxi ride in New York City,
given the pickup and dropoff locations, trip duration and distance etc.
The original data was published by the [NYC gov](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

In this [notebook](notebooks/jupyter.ipynb) ([the Databricks version](notebooks/databricks.py)),
we show how to build and evaluate a very simple linear regressor step by step,
following the best practices of machine learning engineering.
By the end of this example,
you will learn how to use MLflow Recipes to
- Ingest the raw source data.
- Splits the dataset into training/validation/test.
- Create a feature transformer and transform the dataset.
- Train a linear model (regressor) to predict the taxi fare.
- Evaluate the trained model, and improve it by iterating through the `transform` and `train` steps.
- Register the model for production inference.

All of these can be done with Jupyter notebook or on the Databricks environment.
Finally, challenge yourself to build a better model. Try the following:
- Find a better data source with more training data and more raw feature columns.
- Clean the dataset to make it less noisy.
- Find better feature transformations.
- Fine tune the hyperparameters of the model.
