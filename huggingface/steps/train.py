"""
This module defines the following routine to be used by the 'train' step:
- ``trainer_fn``: Returns a ``Trainer`` object for training a HF model.
"""
from typing import Dict, Any
from transformers import DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


def trainer_fn(estimator_params: Dict[str, Any]):
    """
    Returns an *untrained* HF trainer here.

    Input estimator_params is a dictionary of parameters passed to the estimator.
    It contains the following keys:
      'train_dataset': A ``datasets.Dataset`` object for training.
      'validation_dataset': A ``datasets.Dataset`` object for validation.
      'output_dir': A string containing the path to the cache directory.
    """
    training_args = TrainingArguments(
        output_dir=estimator_params["output_dir"],
        evaluation_strategy="steps",
        eval_steps=2,
        save_steps=5,
        max_steps=20,
        log_level="warning",
    )
    # Model name
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_fast=True)
    # Set PAD token to EOS token for variable length sequences.
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    # We resize the embeddings only when necessary to avoid index errors.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorForLanguageModeling(
        tokenizer,
        mlm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=estimator_params["train_dataset"],
        eval_dataset=estimator_params["validation_dataset"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer
