We recommend using the `Azure ML compute for job submission` to run large-scale experiments.

(run/azureml-compute)=
### AzureML Compute

#### Using EnCortex to run on AML

First, run `encortex_setup` command and provide the prompted keys and paths related to AML, Blob storage and Docker. This creates a `.encortex` folder in the home directory of your current userspace with the file `config.encortex` holding this information.

Once EnCortex is installed and AML related configuration is set, use the bundled CLI command `encortex_aml` that can run a script on AML. The snippet below shows how to use encortex_aml.

```bash
Usage: encortex_aml [OPTIONS] COMPUTE_CLUSTER_NAME ENVIRONMENT_NAME
                    SCRIPT_DIR_PATH SCRIPT_NAME EXPERIMENT_NAME

Arguments:
  COMPUTE_CLUSTER_NAME  [required]
  ENVIRONMENT_NAME      [required]
  SCRIPT_DIR_PATH       [required]
  SCRIPT_NAME           [required]
  EXPERIMENT_NAME       [required]

Options:
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.

  --help                Show this message and exit.
```

Example:

```bash
encortex_aml t-vballoli2 EnCortexConda ./script test_script.py test
```

The above command runs the `test_script.py` in the folder `./script` on compute instance `t-vballoli2` using Environment `EnCortexConda` with the experiment name being `test`.

#### Custom Job Submission

AzureML Compute Instances and Cluster helps you run your scripts that use `EnCortex` on the hosted compute instances/clusters seamlessly from your local notebook/development environment(please follow the instructions to [setup your compute instance/cluster](setup/azureml/compute)). In order to do this, you need two files. One file, let's call it `custom_scenario.py` consists of your implementation of a scenario using `EnCortex`. The other file, let's call it `run_on_aml.py` takes the `custom_scenario.py`, uses the `EnCortex` docker image and submits the job to AML(see directory tree below).

```
encortex_on_aml/
├── datasets
├── run_on_aml.py
└── custom_scenario.py
```

For example, the code below contains contents on the `run_on_aml.py` file(alternatively, the following code can also be run inside a Jupyter Notebook(best viewed using VSCode)).

````{margin}
```{list-table}
:header-rows: 1

* - Variable
  - Value
* - compute_cluster
  - Name of the compute cluster from [compute resource](setup/azureml/compute)
* - config_path
  - Path to `config.json` from [AML setup](setup/azureml)
* - script_dir
  - Directory containing the `custom_scenario.py` code
* - script_path
  - `custom_scenario.py` - `EnCortex` code
```
````

```python
# Import AzureML components
from azureml.core import Workspace, Experiment, Environment,ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.widgets import RunDetails

# Import EnCortex
import encortex

#Instantiate compute, workspace, environment, experiment_name and script to run
compute_cluster = 'encortexcompute'
config_path = 'config.json' #Obtained from AzureML resource in Azure ML subscription
environment_name = "EnCortex"
script_dir = "." #Path to the directory of the file custom_scenario.py.
script_path = "custom_scenario.py" #Name of the file you want to run on AML compute
experiment_name = "encortex_experiment"
remote = True #If outside of AML compute(which is True in this case)


#Run the script
if remote:
    ws = Workspace.from_config(path=config_path)
else:
    ws = Workspace.from_config()
cluster = ComputeTarget(workspace=ws, name=compute_cluster)
env = ws.environments[environment_name]

src = ScriptRunConfig(source_directory=script_dir,
                    script=script_path,
                    environment=env,
                    compute_target=cluster
                   )

run = Experiment(ws, experiment_name).submit(src)

print(RunDetails(run))
RunDetails(run).show() #If run in a jupyter notebook(recommended inside VSCode), an AML widget shows up tracking progress
```
**Importantly, `script_dir` points towards the directory containing `custom_scenario.py`.**

> Modify the variables `compute_cluster`, `config_path`, `environment_name`, `script_dir`, `script_path`, `experiment_name` based on your experiments.

## Building your own environment

1. Create the wheel files for your local EnCortex
   1. Navigate to the root of your local EnCortex
   2. Run `python setup.py bdist_wheel --universal`. The latest wheel will be available in the `dist` folder
   3. Copy the latest EnCortex wheel, the DFDB wheel to another folder and the `environment.yml` file to a different folder.
   4. In the newly created folder with the above files, create a python file called `encortex_env.py`
   5. Add the following code to `encortex_env.py`. This creates an environment named `EnCortexConda`
   6. ```python
      from azureml.core import Environment
      from azureml.core import Workspace
      ml_client = Workspace.from_config()

      env = Environment.from_conda_environment('EnCortexConda', 'environment.yml')
      dfdb = Environment.add_private_pip_wheel(workspace=ml_client,file_path = "dist/encortex-0.1.21-py2.py3-none-any.whl", exist_ok=True)
      enc = Environment.add_private_pip_wheel(workspace=ml_client,file_path = "dfdb-0.0.2-py2.py3-none-any.whl", exist_ok=True)
      deps = env.python.conda_dependencies
      deps.add_pip_package(enc)
      deps.add_pip_package(dfdb)
      env.python.conda_dependencies = deps
      env.build(ml_client)
      ```
   7. Run `python encortex_env.py`

(setup/aml-job-results)=
#### See AML Job results

To view the results of the experiment:
1. Head on to your AzureML studio
2. Click on `Jobs` from the sidebar
3. Click on the experiment name(`encortex_experiment` from above)
4. Click on the job to view logs and metrics.

(run/local)=
### Local Development

1. Clone the repository from [DevOps](https://dev.azure.com/MSREnergy/EnCortex/_git/EnCortex-Release)
2. Change your working directory to the root of the `EnCortex-Release` folder
3. Run `python3 -m pip install -e .` on your terminal to install directly from source
4. Run `python3 -m pip install dfdb-0.0.2-py2.py3-none-any.whl`
5. `import encortex` should run from any script on your local machine.
6. Run any example from the `examples` directory(for example: `python examples/microgrid_example.py`). Note: ensure appropriate data files exist before running your experiments
