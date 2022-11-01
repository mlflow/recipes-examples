# Databricks notebook source

# MAGIC %md
# MAGIC # MLflow Regression Recipe Databricks Notebook
# MAGIC This notebook runs the MLflow Regression Recipe on Databricks and inspects its results.
# MAGIC
# MAGIC For more information about the MLflow Regression Recipe, including usage examples,
# MAGIC see the [Regression Recipe overview documentation](https://mlflow.org/docs/latest/recipes.html#regression-recipe)
# MAGIC and the [Regression Recipe API documentation](https://mlflow.org/docs/latest/python_api/mlflow.recipes.html#module-mlflow.recipes.regression.v1.recipe).

# COMMAND ----------

# MAGIC %pip install mlflow
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

from mlflow.recipes import Recipe

r = Recipe(profile="databricks")

# COMMAND ----------

r.clean()

# COMMAND ----------

r.inspect()

# COMMAND ----------

r.run("ingest")

# COMMAND ----------

r.run("split")

# COMMAND ----------

r.run("transform")

# COMMAND ----------

r.run("train")

# COMMAND ----------

r.run("evaluate")

# COMMAND ----------

r.run("register")

# COMMAND ----------

r.inspect("train")

# COMMAND ----------

training_data = r.get_artifact("training_data")
training_data.describe()

# COMMAND ----------

trained_model = r.get_artifact("model")
print(trained_model)
