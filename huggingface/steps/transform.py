"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
from typing import Dict, Any, List, Tuple
from transformers import AutoTokenizer


def transformer_fn():
    """
    Returns a function to process input examples and generate model inputs via tokenizer.
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
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        answers = examples[answer_column]
        return questions, answers

    def preprocess_examples(examples):
        inputs, targets = preprocess_squad_batch(
            examples, question_column, answer_column
        )
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
        )
        labels = tokenizer(
            targets,
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

    return preprocess_examples
