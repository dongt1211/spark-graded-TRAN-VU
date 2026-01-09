import json
import pandas as pd
import configs as conf

def parse_schema_to_dict(schema_file="schema.csv", output_file="schema_dict.json"):
    """
    Parse the Google Cluster schema CSV file and convert it into a structured
    Python dictionary grouped by file pattern, then save the result as a JSON file.

    The input schema file is expected to contain the following columns:
    - 'file pattern' : name or prefix of the data file
    - 'content'      : semantic meaning of each column
    - 'format'       : data type or format of each column
    - 'mandatory'    : whether the column is mandatory ('YES' or 'NO')

    Parameters:
    - schema_file (str): Path to the schema CSV file.
    - output_file (str): Path where the generated JSON file will be saved.

    Returns:
    - dict: A nested dictionary mapping each file pattern to its column
            contents, formats, and mandatory flags.
    """
    df = pd.read_csv(schema_file)    
    schema_dict = {}

    for _, row in df.iterrows():
        file_pattern = row["file pattern"].split("/")[0]
        if file_pattern not in schema_dict:
            schema_dict[file_pattern] = {
                "col_content": [],
                "col_format": [],
                "col_mandatory": []
            }

        schema_dict[file_pattern]["col_content"].append(row["content"])
        schema_dict[file_pattern]["col_format"].append(row["format"])
        schema_dict[file_pattern]["col_mandatory"].append(1 if row["mandatory"] == "YES" else 0)

    # Save to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schema_dict, f, ensure_ascii=False, indent=4)

    print(f"Schema dictionary has been saved to '{output_file}'")
    return schema_dict


import copy

def strategies_to_config_list(strategy: dict = conf.STRATEGIES, defaults: dict = conf.SPARK_DEFAULTS, output_file="test_conf_list.json"):
    """
    Generate a list of full Spark configs.
    Each config differs from defaults by ONLY ONE parameter.
    """
    configs = []

    for key, values in strategy.items():
        for v in values:
            cfg = copy.deepcopy(defaults)

            # nested key: extra_conf.xxx
            if key.startswith("extra_conf."):
                sub_key = key.replace("extra_conf.", "")
                cfg["extra_conf"][sub_key] = v
            else:
                cfg[key] = v

            configs.append(cfg)

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(configs, f, indent=2)
            
    return configs



if __name__ == "__main__":
    # import configs as conf
    # import os
    # os.makedirs("output", exist_ok=True)
    # schema_file = conf.DATA_DIR + "schema.csv"
    # schema_json = conf.OUTPUT_DIR + "schema_dict.json"
    # parse_schema_to_dict(schema_file, schema_json)

    test_conf_path = conf.OUTPUT_DIR + "test_conf_list.json"
    conf_list = strategies_to_config_list(output_file=test_conf_path)
    