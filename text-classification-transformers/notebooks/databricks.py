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
# MAGIC %pip install -r ../requirements.txt
# MAGIC %pip install -U sentence-transformers
# MAGIC 
# MAGIC %pip install git+https://github.com/mshtelma/mlflow.git@multiclassclassification

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

# MAGIC %md ### Split the dataset into train, validation and test:

# COMMAND ----------

r.run("split")

# COMMAND ----------

df = r.get_artifact("training_data").sample(10)
display(df)


# COMMAND ----------

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

#Our sentences we like to encode
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.',
    'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
embeddings = model.encode(df["Consumer_complaint_narrative"].values)
embeddings

# COMMAND ----------

r.run("transform")

# COMMAND ----------

# MAGIC %md ### Train the model:

# COMMAND ----------

r.run("train")

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


