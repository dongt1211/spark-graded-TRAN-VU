from pyspark.sql.functions import count, sum as spark_sum
import time

def run_job(spark, task_path, job_path, usage_path, cache=False):
    start = time.time()

    df_task = spark.read.csv(task_path, header=False, inferSchema=True)
    df_job  = spark.read.csv(job_path, header=False, inferSchema=True)
    df_usage = spark.read.csv(usage_path, header=False, inferSchema=True)

    # task_events ⨝ job_events
    df1 = df_task.join(
        df_job,
        df_task._c0 == df_job._c0
    )

    # (task_events ⨝ job_events) ⨝ task_usage
    df2 = df1.join(
        df_usage,
        df_task._c0 == df_usage._c0
    )

    if cache:
        df2 = df2.cache()

    # filter + heavy aggregation
    df_agg = (
        df2
        .filter(df_task._c2.isin([1, 2, 3]))   # event types
        .groupBy(df_task._c0)                  # job_id
        .agg(
            count("*").alias("num_records"),
            spark_sum(df_usage._c5).alias("total_cpu"),
            spark_sum(df_usage._c6).alias("total_memory")
        )
    )

    df_agg.count()  # action → trigger full execution

    return time.time() - start
