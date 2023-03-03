"""
This module defines the following routine to be used by the 'train' step:
- ``trainer_fn``: Returns a ``Trainer`` object for training a HF model.
"""
from typing import Dict, Any, List, Tuple
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    DistilBertForMaskedLM,
    Seq2SeqTrainer,
    TrainingArguments,
)


def trainer_fn(estimator_params: Dict[str, Any]):
    """
    Returns an *untrained* HF trainer here.

    Input estimator_params is a dictionary of parameters passed to the estimator.
    It contains the following keys:
      'train_dataset': A ``datasets.Dataset`` object for training.
      'cache_dir': A string containing the path to the cache directory.
    """
    training_args = TrainingArguments(output_dir=estimator_params["cache_dir"])
    # Model name
    model_name = "distilbert-base-uncased"
    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=training_args.output_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=training_args.output_dir,
        use_fast=True,
    )
    model = DistilBertForMaskedLM.from_pretrained(
        model_name,
        config=config,
        cache_dir=training_args.output_dir,
    )
    # We resize the embeddings only when necessary to avoid index errors.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    train_dataset = estimator_params["train_dataset"]
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer
