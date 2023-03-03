"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import FunctionTransformer
from transformers import AutoTokenizer


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )
    # Global parameters for tokenizer and model.
    max_seq_length = min(128, tokenizer.model_max_length)
    padding = "max_length"
    question_column = "character"
    answer_column = "speech"

    def preprocess_squad_batch(
        examples,
        question_column: str,
        answer_column: str,
    ) -> Tuple[pd.Series, pd.Series]:
        questions = examples[question_column]
        answers = examples[answer_column]
        return questions, answers

    def preprocess_examples(examples):
        inputs, targets = preprocess_squad_batch(
            examples, question_column, answer_column
        )
        model_inputs = tokenizer(
            inputs.tolist(),
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
        )
        labels = tokenizer(
            targets.tolist(),
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
        )
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_input_ids"] = labels["input_ids"]
        return model_inputs

    return FunctionTransformer(preprocess_examples)
