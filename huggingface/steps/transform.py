"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
from transformers import AutoTokenizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


def transformer_fn():
    """
    Returns a function to process input examples and generate model inputs via tokenizer.
    """
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Global parameters for tokenizer and model.
    max_seq_length = tokenizer.model_max_length

    def preprocess_examples(examples):
        model_inputs = tokenizer(
            examples["text"],
            max_length=max_seq_length,
            truncation=True,
            padding="max_length",
        )
        model_inputs["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in model_input]
            for model_input in model_inputs["input_ids"]
        ]
        model_inputs["labels"] = model_inputs["input_ids"]
        model_inputs["decoder_input_ids"] = model_inputs["input_ids"]
        return model_inputs

    return Pipeline(
        steps=[
            (
                "process_examples",
                FunctionTransformer(preprocess_examples),
            )
        ]
    )
