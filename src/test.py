import json
import time
from pathlib import Path
from copy import deepcopy
from init_pyspark_session import init_spark  # your Spark init function
from run_job_module import run_job           # your pre-defined run_job function

def run_experiment(
    config_list_json="output/test_config_list.json",
    task_path="data/clusterdata-2011-2/task_events/part-00160-of-00500.csv",
    job_path="data/clusterdata-2011-2/job_events/part-00160-of-00500.csv",
    usage_path="data/clusterdata-2011-2/task_usage/part-00160-of-00500.csv",
    output_file="output/experiment_results.json"
):
    """
    Run a Spark job for each config in the config list and measure execution time.

    Parameters:
    - config_list_json: str, path to JSON file containing list of Spark configs
    - task_path, job_path, usage_path: str, paths to CSV data files
    - output_file: str, path to save JSON results

    Each config differs by only one parameter (ablation study style).
    """

    # Load config list
    with open(config_list_json, "r") as f:
        config_list = json.load(f)

    results = []

    for i, cfg in enumerate(config_list):
        print(f"\n=== Running experiment {i+1}/{len(config_list)} ===")

        # Initialize Spark session with this config
        spark = init_spark(
            app_name=cfg.get("app_name", "GoogleClusterAnalysis"),
            master=cfg.get("master", "local[*]"),
            shuffle_partitions=cfg.get("shuffle_partitions", 200),
            executor_memory=cfg.get("executor_memory", "4g"),
            driver_memory=cfg.get("driver_memory", "4g"),
            arrow_enabled=cfg.get("arrow_enabled", True),
            serializer=cfg.get("serializer", "org.apache.spark.serializer.KryoSerializer"),
            extra_conf=cfg.get("extra_conf", {})
        )

        # Determine whether to cache (optional config)
        cache_flag = cfg.get("cache", False)

        # Run the pre-defined job and measure execution time
        try:
            elapsed_time = run_job(
                spark=spark,
                task_path=task_path,
                job_path=job_path,
                usage_path=usage_path,
                cache=cache_flag
            )
            print(f"Execution time: {elapsed_time:.2f}s")
        except Exception as e:
            print(f"Error running job: {e}")
            elapsed_time = None

        # Save result
        results.append({
            "config_index": i,
            "config": cfg,
            "elapsed_time": elapsed_time
        })

        # Stop Spark to avoid memory leaks
        spark.stop()

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Save all results to JSON
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nAll experiments completed. Results saved to {output_file}")
