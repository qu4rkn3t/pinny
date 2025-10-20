import yaml
import itertools
from typing import Dict, Any, Generator


def generate_hyperparameter_configs(
    options_path: str,
) -> Generator[Dict[str, Any], None, None]:
    with open(options_path, "r") as f:
        options = yaml.safe_load(f)

    search_space = options.get("search_space", {})
    global_settings = options.get("global_settings", {})

    if not search_space:
        print(
            "[WARNING] 'search_space' not defined in options.yaml. No search will be run."
        )
        return

    keys, values = zip(*search_space.items())
    hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for hparams in hyperparam_combinations:
        run_params = global_settings.copy()
        run_params.update(hparams)
        yield run_params
