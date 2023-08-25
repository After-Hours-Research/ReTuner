# ReTuner (Retrieval Tuning)

Everyone uses sentence transformers to retrieve document chunks given a question. This isn't optimal, there are probably elements of the query we should be boosting, and elements we should be dampening.

Let's do it!

This repo is a WIP - theres still lots to work on. Including documentation!
## Status

- [x] Data Loading
- [x] Question Generation
- [x] Baselines
- [x] Model Tuning
- [x] MixUp
- [x] InfoNCE
- [x] Smart Chunking


## Installation

```bash
pip install poetry
poetry install
```

## Basics

### Hydra

You can run experiments with hydra: `retuner/cli/data_cli.py` to generate data, and `retuner/cli/tune_cli.py` to add an adapter / train.

We'll add better entrypoints soon.


### Python API

Alternatively, the Python API:

#### Generating Data

```python
QuestionGenerator.from_hf_dataset(...)
```
or
```python
QuestionGenerator.from_directory(...)
```

#### Training

```python
    tuner_model = TuningModule(base_model, other args, ...)
    tuner_trainer = TunerTrainer(...)
    tuner_trainer.fit(tuner_model, ...)
```


