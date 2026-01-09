# Init pyspark session
from pyspark.sql import SparkSession

def init_spark(app_name="GoogleClusterAnalysis",
               master="local[*]",
               shuffle_partitions=200,
               executor_memory="4g",
               driver_memory="4g",
               arrow_enabled=True,
               serializer="org.apache.spark.serializer.KryoSerializer",
               # log_level="INFO",
               extra_conf=None):
    """
    Initialize and return a Spark session with configurable settings.

    Parameters:
    - app_name: str, Spark application name
    - master: str, Spark master URL
    - shuffle_partitions: int, number of shuffle partitions
    - executor_memory: str, memory allocated for executors
    - driver_memory: str, memory allocated for driver
    - arrow_enabled: bool, enable Arrow optimization for PySpark
    - serializer: str, Spark serializer class
    - extra_conf: dict, any additional Spark config as key-value pairs

    Returns:
    - SparkSession object
    """
    builder = SparkSession.builder.appName(app_name).master(master)
    builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    builder = builder.config("spark.executor.memory", executor_memory)
    builder = builder.config("spark.driver.memory", driver_memory)
    builder = builder.config("spark.serializer", serializer)
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", str(arrow_enabled).lower())
    # builder = builder.config("spark.log.level", str(log_level).upper())

    # Apply any extra Spark config passed in
    if extra_conf:
        for k, v in extra_conf.items():
            builder = builder.config(k, v)

    spark = builder.getOrCreate()
    print(f"Spark session initialized. Version: {spark.version}")
    return spark
