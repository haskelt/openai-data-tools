This package provides a set of classes for processing data using calls to OpenAI models.

## DataProcessor

Allows you to take a list of input values, the have an OpenAI model process each one to preturn a list of output values. See the `date_formatting`, `info_extraction`, and `name_matching` demos for examples of how to use it.

## Scorer

A class for comparing model output values with target values, useful for evaluating model performance when the correct output value is known. See the `date_formatting`, `info_extraction`, and `name_matching` demos for examples of how to use it.

## RunAggregator

A class for taking output lists from multiple runs of a model and producing a single output list with the most common value returned. See the `name_matching` demo for an example.

## DataCoder

A specialized version of `DataProcessor` for making yes/no judgments about whether to apply a given code to each data point in a list. See the `coding` demo for an example.

## ClassificationScorer

A specialized version of `Scorer` that can be used when the model output is a yes/no answer. See the `coding` demo for an example.

## Chat

A class that simulates an interactive chat with a model. See the `chat` demo for an example.
