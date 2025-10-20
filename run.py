import yaml
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

import engine.model as model
import problems.problems as problems
from engine.train import train_pinn, PinnResult

PROBLEM_MAP = {
    "ExpDecay": problems.ExpDecay,
    "LogisticGrowth": problems.LogisticGrowth,
    "HarmonicOscillator": problems.HarmonicOscillator,
}
MODEL_MAP = {
    "MLP": model.MLP,
    "ResNetMLP": model.ResNetMLP,
}
ACTIVATION_MAP = {
    "Tanh": nn.Tanh,
    "SiLU": nn.SiLU,
    "ReLU": nn.ReLU,
}


def get_problem_instance(problem_name: str, **kwargs) -> problems.BaseProblem:
    if problem_name not in PROBLEM_MAP:
        raise ValueError(f"Problem '{problem_name}' not found in PROBLEM_MAP.")
    return PROBLEM_MAP[problem_name](**kwargs)


def get_model_instance(params: Dict[str, Any]) -> nn.Module:
    model_name = params["model_class"]
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model '{model_name}' not found in MODEL_MAP.")

    activation_str = params.get("activation", "Tanh")
    if activation_str not in ACTIVATION_MAP:
        raise ValueError(f"Activation '{activation_str}' not found in ACTIVATION_MAP.")

    return MODEL_MAP[model_name](
        input_dim=params["input_dim"],
        output_dim=params["output_dim"],
        width=params["width"],
        hidden_dim=params["hidden_dim"],
        activation=ACTIVATION_MAP[activation_str],
    )


def get_evaluation_tensors(params: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    problem = get_problem_instance(params["problem_class"])
    t_eval = torch.linspace(params["start"], params["stop"], 500).reshape(-1, 1)
    y_true = problem.exact(t_eval)
    return t_eval, y_true


def execute_run(
    config_source: Any, is_dict: bool = False, silent: bool = False
) -> Tuple[PinnResult, Dict[str, Any]]:
    if is_dict:
        params = config_source.get("run_params", {})
    else:
        with open(config_source, "r") as f:
            config = yaml.safe_load(f)
        params = config.get("run_params", {})

    with open("options.yaml", "r") as f:
        options = yaml.safe_load(f)

    final_params = options.get("global_settings", {}).copy()
    final_params.update(params)

    if not silent:
        print("\n" + "-" * 25 + " RUN CONFIGURATION " + "-" * 25)
        for key, val in final_params.items():
            print(f"  {key:<20}: {val}")
        print("-" * 70 + "\n")

    problem_instance = get_problem_instance(final_params["problem_class"])
    model_instance = get_model_instance(final_params)

    result = train_pinn(
        model_instance, problem_instance, final_params, track_residuals=True
    )

    if not silent:
        print("\n" + "-" * 30 + " RUN RESULTS " + "-" * 31)
        for key, value in result.metrics.items():
            print(f"  {key:<10}: {value:.6f}")
        print("-" * 70)

    return result, final_params
