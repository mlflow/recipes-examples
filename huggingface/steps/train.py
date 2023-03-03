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
        model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True
    )
    model = DistilBertForMaskedLM.from_pretrained(
        model_name,
        config=config
    )
    # Global parameters for tokenizer and model.
    max_seq_length = min(128, tokenizer.model_max_length)
    padding = "max_length"
    question_column = "character"
    answer_column = "speech"

    # We resize the embeddings only when necessary to avoid index errors.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def preprocess_squad_batch(
        examples,
        question_column: str,
        answer_column: str,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        answers = examples[answer_column]
        pairs = [[q, a] for q, a in zip(questions, answers)]
        return questions, answers, pairs

    def preprocess_examples(examples):
        inputs, targets, pairs = preprocess_squad_batch(
            examples, question_column, answer_column
        )
        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            targets,
            max_length=max_seq_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["decoder_input_ids"] = labels["input_ids"]
        return model_inputs

    train_dataset = estimator_params["train_dataset"]
    # Create train feature from dataset
    train_dataset = train_dataset.map(
        preprocess_examples,
        batched=True,
        num_proc=1,
        load_from_cache_file=True,
        remove_columns=[question_column, answer_column],
        desc="Running tokenizer on train dataset",
    )

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
