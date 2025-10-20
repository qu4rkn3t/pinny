import os
import argparse
from tqdm import tqdm
from datetime import datetime

import pandas as pd

import run as run
import hyperparameters as hp
import visualizations as vis


def main():
    parser = argparse.ArgumentParser(
        description="PINNY",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["run", "search"],
        help="Command to execute:\n"
        "  run     - Execute a single experiment using 'config.yaml'.\n"
        "  search  - Run a full hyperparameter search defined in 'options.yaml'.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration file for a single run (default: config.yaml).",
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_output_dir = os.path.join("results", timestamp)
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"[INFO] Results will be saved in: {base_output_dir}")

    if args.command == "run":
        print("\n[INFO] Executing a single run...")
        if not os.path.exists(args.config):
            print(f"[ERROR] Config file not found at '{args.config}'. Exiting.")
            return

        result, params = run.execute_run(args.config)

        single_run_dir = os.path.join(base_output_dir, "single_run_visuals")
        os.makedirs(single_run_dir, exist_ok=True)

        t_eval, y_true = run.get_evaluation_tensors(params)

        print("[INFO] Generating visualizations for the single run...")
        vis.plot_solution_comparison(
            t_eval.detach().cpu().numpy(),
            y_true.detach().cpu().numpy(),
            result.y_pred,
            params["problem_class"],
            single_run_dir,
        )
        vis.plot_loss_curve(
            result.loss_history, params["problem_class"], single_run_dir
        )
        if result.residual_history is not None:
            vis.plot_residual_distribution_over_time(
                result.residual_history, params, params["problem_class"], single_run_dir
            )
        print(f"[SUCCESS] Visualizations saved to {single_run_dir}")

    elif args.command == "search":
        print("\n[INFO] Initializing hyperparameter search...")

        all_results = []
        search_configs = list(hp.generate_hyperparameter_configs("options.yaml"))
        total_runs = len(search_configs)
        print(f"[INFO] Found {total_runs} hyperparameter combinations to run.")

        pbar = tqdm(
            enumerate(search_configs), total=total_runs, desc="Hyperparameter Search"
        )
        for i, run_params in pbar:
            temp_config = {"run_params": run_params}

            try:
                result, params = run.execute_run(temp_config, is_dict=True, silent=True)

                run_info = {"run_id": i, **params, **result.metrics}
                all_results.append(run_info)
                pbar.set_postfix({"last_rmse": f"{result.metrics.get('RMSE', 0):.4f}"})

            except Exception as e:
                tqdm.write(f"[WARNING] Run {i + 1} failed: {e}. Skipping.")

        if not all_results:
            print("[ERROR] No runs completed successfully. Exiting.")
            return

        final_df = pd.DataFrame(all_results)
        search_summary_dir = os.path.join(base_output_dir, "search_summary_visuals")
        os.makedirs(search_summary_dir, exist_ok=True)
        final_df.to_csv(
            os.path.join(search_summary_dir, "hyperparameter_search_summary.csv"),
            index=False,
        )

        print("\n[INFO] Generating summary visualizations for the search...")
        vis.plot_hyperparameter_correlation_matrix(final_df, search_summary_dir)

        for problem_name in final_df["problem_class"].unique():
            problem_df = final_df[final_df["problem_class"] == problem_name]
            problem_dir = os.path.join(search_summary_dir, problem_name)
            os.makedirs(problem_dir, exist_ok=True)

            vis.plot_architecture_comparison(
                problem_df, "RMSE", f"RMSE: {problem_name}", problem_dir
            )
            vis.plot_hyperparameter_heatmap(
                problem_df,
                "hidden_dim",
                "activation",
                "R2",
                f"R² Score: {problem_name}",
                problem_dir,
            )

        print(f"[SUCCESS] Summary visualizations saved to {search_summary_dir}")

        print("\n" + "=" * 80)
        print("                  BEST MODEL CONFIGURATIONS PER PROBLEM")
        print("=" * 80)
        best_models = final_df.loc[final_df.groupby("problem_class")["RMSE"].idxmin()]
        for _, row in best_models.iterrows():
            print(f"\n[ PROBLEM: {row['problem_class']} ]")
            print(f"  > Best Architecture : {row['model_class']}")
            print(f"  > Best RMSE         : {row['RMSE']:.6f}")
            print("  > Hyperparameters:")
            print(f"    - Learning Rate   : {row['learning_rate']:.5f}")
            print(f"    - Width x Layers  : {row['width']} x {row['hidden_dim']}")
            print(f"    - Activation      : {row['activation']}")
            print(f"    - ODE Loss Weight : {row['ode_loss_weight']}")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
