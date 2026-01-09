# Spark default configuration
SPARK_DEFAULTS = {
    "app_name": "GoogleClusterAnalysis",
    "master": "local[*]",
    "shuffle_partitions": 200,
    "executor_memory": "4g",
    "driver_memory": "4g",
    "arrow_enabled": True,
    "serializer": "org.apache.spark.serializer.KryoSerializer",
    "extra_conf": {
        "spark.executor.cores": 4,
        "spark.sql.autoBroadcastJoinThreshold": 10485760  # 10 MB
    }
}

# Spark Test Strategies
STRATEGIES = {
    "shuffle_partitions": [50, 100, 200, 400, 800],
    "executor_memory": ["2g", "4g", "8g"],
    "driver_memory": ["2g", "4g"],
    "spark.executor.cores": [2, 4, 8],
    "spark.sql.autoBroadcastJoinThreshold": [-1, 10485760, 52428800],
    "arrow_enabled": [True, False],
    "spark.sql.adaptive.enabled": [True, False],
    "cache": [True, False]
}

# File paths
DATA_DIR = "data/clusterdata-2011-2/"
OUTPUT_DIR = "output/"

# Other default settings
DEFAULTS = {
    "log_level": "INFO",       # Spark log level
    "max_partition_size": "128m"  # optional, can be used in extra_conf
}
