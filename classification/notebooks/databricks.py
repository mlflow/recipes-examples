# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Classification Recipe Databricks Notebook
# MAGIC This notebook runs the MLflow Classification Recipe on Databricks and inspects its results.
# MAGIC 
# MAGIC For more information about the MLflow Classification Recipe, including usage examples,
# MAGIC see the [Classification Recipe overview documentation](https://mlflow.org/docs/latest/recipes.html#classification-recipe)
# MAGIC and the [Classification Recipe API documentation](https://mlflow.org/docs/latest/python_api/mlflow.recipes.html#module-mlflow.recipes.classification.v1.recipe).

# COMMAND ----------

# MAGIC %pip install -r ../../requirements.txt
# MAGIC %pip install git+https://github.com/mlflow/mlflow@sunish-worst-examples-df

# COMMAND ----------

# MAGIC %md ### Start with a recipe:

# COMMAND ----------

from mlflow.recipes import Recipe

r = Recipe(profile="databricks")

# COMMAND ----------

r.clean()

# COMMAND ----------

# MAGIC %md ### Inspect recipe DAG:

# COMMAND ----------

r.inspect()

# COMMAND ----------

# MAGIC %md ### Ingest the dataset:

# COMMAND ----------

r.run("ingest")

# COMMAND ----------

# MAGIC %md ### Perform some EDA on the ingested dataset

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

ingested_data = r.get_artifact("ingested_data")

dims = (3, 4)

f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in ingested_data.columns:
    if col == "is_red":
        continue  # Box plots cannot be used on indicator variables
    sns.boxplot(
        x=ingested_data["is_red"], y=ingested_data[col], ax=axes[axis_i, axis_j]
    )
    axis_j += 1
    if axis_j == dims[1]:
        axis_i += 1
        axis_j = 0

# COMMAND ----------

# MAGIC %md ### Split the dataset into train, validation and test:

# COMMAND ----------

r.run("split")

# COMMAND ----------

r.run("transform")

# COMMAND ----------

# MAGIC %md ### Train the model:

# COMMAND ----------

# MAGIC %

# COMMAND ----------

r.run("train")

# COMMAND ----------

r.get_artifact("predicted_training_data")

# COMMAND ----------

# MAGIC %md ### Evaluate the model:

# COMMAND ----------

r.run("evaluate")

# COMMAND ----------

# MAGIC %md ### Register the model:

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

# COMMAND ----------

import pandas as pd

mydict = [{'a': 1, 'b': 2, 'c': 3, 'd': 4},
          {'a': 100, 'b': 200, 'c': 300, 'd': 400},
          {'a': 1000, 'b': 2000, 'c': 3000, 'd': 4000 }]
df = pd.DataFrame(mydict)
df

# df.iloc[0:, 0].values

# COMMAND ----------

df['a'].values

# COMMAND ----------


