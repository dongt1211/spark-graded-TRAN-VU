1. Run docker environment:

```
docker compose up -d
```

2. Download data from gs cloud

```
bash scripts/download-data.sh
```

**IMPORTANT**: `gsutil` is required. If `gsutil` is not yet installed on local machine, use docker instead by executing docker container `python-env`

```
docker exec -it python-env /bin/bash
```

then run the downloading script above again.

3. Analyse with Spark using JupyterNotebook

First, active docker container `python-env`

```
docker exec -it python-env /bin/bash
```

then run the script for active jupyter environment

```
bash scripts/active-jupyter.sh
```
