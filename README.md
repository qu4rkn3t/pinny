# PINNY

PINNY is a physics informed neural network solving framework for ordinary differential equations. The framework has capabilities to:

- Define ODEs as Python classes.
- Implement multiple neural network architectures.
- Run single experiments with pre-defined parameters.
- Conduct comprehensive hyperparameter searches on specific ODEs.
- Generate customized visualizations for training runs and model evaluation.

## Project Structure

`config.yaml`: configuration for a SINGLE run.

`hyperparameters.py`: logic for generating hyperparameter search runs.

`main.py`: main entry point for the user (run or search).

`options.yaml`: all available options and the hyperparameter search space.

`run.py`: core logic for executing a single training run.

`engine/`

- `model.py`: implementations of neural network architectures (MLP, ResNet).
- - `train.py`: the main training loop for the PINN model.

`problems/`

- `base.py`: abstract base class for ODE problems.

- `problems.py`: implementations of specific ODE problems.

`utils/`

- `metrics.py`: functions to compute performance metrics (RMSE, etc.).

`visualizations/`

Contains functions for charting results of training and testing.

## Usage

To run a single experiment, you define the parameters in `config.yaml` and:

```
python main.py run
```

You can also point to a different configuration file

```
python main.py run --config other.yaml
```

To run a hyperparameter search, you define the search space in `options.yaml` and:

```
python main.py search
```

### Configuration Details

`options.yaml`

- `available_components`: lists the string names of all models, problems, and activations that can be used. If you add a new model, you must add its class name here.
- `global_settings`: default parameters (like epochs) that apply to all runs unless a more specific value is provided in `config.yaml` or the `search_space`.
- `search_space`: possible values for each parameter

`config.yaml`

- `run_params`: specify the exact `problem_class`, `model_class`, and hyperparameter values for one run. These values will override any defaults from `global_settings` in o`ptions.yaml`.

## Extending the Framework

### Adding a New Problem

To add a new problem, create a new class in the `problems.py` file. Implement the `residual`, `ic_loss`, and `exact` methods. Add the class name to the problem classes list in `options.yaml` and create a map in `run.py`.

## Adding a New Model Architecture

Create a new class that inherits from torch.nn.Module. It should accept the standard parameters (`input_dim`, `output_dim`, `width`, etc.) in its `__init__` method. Implement the forward method. Add the new model's class name to the `model_classes` list in `options.yaml` and add a map in `run.py`.

## Adding a New Visualization

Create a new file in `visualizations/` and implement the code in a function. Expose the function in `visualizations/__init__.py`.
