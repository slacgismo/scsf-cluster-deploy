# scsf-cluster-deploy

Code for running SCSF on an AWS cluster


## Setup

```bash
pip install -r requirements.txt
```

To set up remote computational nodes, run the following:

```bash
node-1> ./ppserver.py

node-2> ./ppserver.py

node-3> ./ppserver.py
```

The server setup script `ppserver.py` is installed when running the `pip` command above. If `pip` is run inside a virtual environment named `env1`, then `ppserver.py` is located here:

```bash
~/.virtualenvs/env1/bin/ppserver.py
```

Otherwise, try searching on your Python path. Note, working inside a virtual environment is always recommended.

[Quick start guide on clusters](https://www.parallelpython.com/content/view/15/30/#QUICKCLUSTERS)