"""
Below is an example adapted from the `run_seq2seq.py` script in the `transformers` library. See
https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_seq2seq_qa.py

This module defines the following routine to be used by the 'train' step:
- ``trainer_fn``: Returns a ``Trainer`` object for training a HF model.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)

_logger = logging.getLogger(__name__)


@dataclass
class TrainingArgs:
    model_name: str = "distilbert-base-uncased"
    tokenizer_name: str = "distilbert-base-uncased"
    cache_dir: str = ""
    question_column: str = "character"
    answer_column: str = "speech"
    num_workers: int = 1
    max_seq_length: int = 384
    max_answer_length: int = 128
    padding: str = "max_length"
    ignore_pad_token_for_loss: bool = True


def trainer_fn(estimator_params: Dict[str, Any]):
    """
    Returns an *untrained* HF trainer here.

    Input estimator_params is a dictionary of parameters passed to the estimator.
    It contains the following keys:
      'train_dataset': A ``datasets.Dataset`` object for training.
      'cache_dir': A string containing the path to the cache directory.
    """
    training_args = TrainingArgs(cache_dir=estimator_params["cache_dir"])
    config = AutoConfig.from_pretrained(
        training_args.model_name,
        cache_dir=training_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        training_args.tokenizer_name,
        cache_dir=training_args.cache_dir,
        use_fast=True,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        training_args.model_name,
        config=config,
        cache_dir=training_args.cache_dir,
    )
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    if training_args.max_seq_length > tokenizer.model_max_length:
        _logger.warning(
            f"The max_seq_length passed ({training_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(training_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_squad_batch(
        examples,
        question_column: str,
        answer_column: str,
    ) -> Tuple[List[str], List[str]]:
        questions = examples[question_column]
        answers = examples[answer_column]

        def generate_input(_question, _context):
            return " ".join(["character:", _question.lstrip()])

        inputs = [generate_input(question) for question in questions]
        targets = [
            answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers
        ]
        return inputs, targets

    def preprocess_examples(examples):
        inputs, targets = preprocess_squad_batch(
            examples, training_args.question_column, training_args.answer_column
        )

        model_inputs = tokenizer(
            inputs,
            max_length=max_seq_length,
            padding=training_args.padding,
            truncation=True,
        )
        # Tokenize targets with text_target=...
        labels = tokenizer(
            text_target=targets,
            max_length=training_args.max_answer_length,
            padding=training_args.padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if (
            training_args.padding == "max_length"
            and training_args.ignore_pad_token_for_loss
        ):
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label]
                for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = estimator_params["train_dataset"]
    # Create train feature from dataset
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_examples,
            batched=True,
            num_proc=training_args.num_workers,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )

    # Data collator
    label_pad_token_id = (
        -100 if training_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer
