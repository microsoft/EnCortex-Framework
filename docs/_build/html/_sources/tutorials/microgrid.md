# MicroGrid

![](../_static/microgrid.png)

A microgrid (MG) is a local, self-sufficient energy system that allows you to generate your own electricity along with control capabilities.
- In this scenario, we consider each MG is associated with a solar farm, battery storage, and consumer demand (an industrial setup).
- The objective is to __efficiently utilize the MG resources__ to __satisfy consumer demand__ while __maximizing cost savings__.

## Brief look into EnCortex implementation details

1. The Entities used here are `Battery`, `Solar`, `Consumer`, `MicroGrid` and `Grid`, which is an umbrella term for a utility grid. The solar, consumer and grid entities require data. The solar entity uses data in the format of 2 columns specifying __timestamps__ and __generation__. The consumer entity also requires __timestamps__ and __consumption__ corresponding to the required timeslots. The grid entity involves using data with fields of __timestamps__, __prices_buy__ and __prices_sell__ information,specifically the prices at which the transactions are made in the energy market.
2. The *action space* or the *decision* in this scenario is to either charge the battery, discharge the battery or letting it stay idle.
3. The supported optimizers are `Mixed-Integer Linear Programming`, `Reinforcement Learning`, `Simulated Annealing`


For more details, please refer to `Section 5.2` in our paper linked below:
```{button-link} ../_static/nsdi23fall-paper705.pdf
:color: primary
:outline:

EnCortex paper
```

Here, we aim to use the below objective function:

````{eval-rst}
.. math::
    :label: Objective function for Energy Arbitrage

    max \sum_{t=0}^{T}-(Price_{t}E_{t}^{Ugrid})
````

where $Price_{t}$ is the time of use price value and $E_{t}^{Ugrid}$ is the energy supplied by the main Utility Grid. The $T$ corresponds to the horizon over which the objective function is maximized.

## Running Micro Grid

To run the MicroGrid experiment, you'll need a specific `config.yaml` file. Create the `config.yml` in your root folder of your AML workspace(as shown below).
<details open>
  <summary><b>microgrid.yml</b>(click to open/close)</summary>

  ```yaml
  #Li-Ion Battery Parameters
  storage_capacity: 10.
  efficiency: 1.
  soc_initial: 1.
  depth_of_discharge: 90.
  soc_minimum: 0.1
  time_unit: 15
  milp_flag: true
  solver: "ort"
  test_flag: false
  min_battery_capacity_factor: 0.8
  battery_cost_per_kWh: 20.
  reduction_coefficient: 0.99998
  degradation_period: 7
  weight_emission: 1.0
  weight_degradation: 0.0
  weight_price: 1.0
  degradation_flag: false
  seed: 40
  ```
</details>

<details>
  <summary><b>Explanation of the above configuration values</b>(click to open)</summary>

  ```{list-table}
  :header-rows: 1

  * - Parameter
    - Description
    - Value
  * - storage_capacity
    - Battery Storage capacity(kWh)
    - 10.
  * - efficiency
    - Battery Charge/Discharge efiiciency
    - 1.
  * - soc_initial
    - SoC initial(Between 0-1)
    - 1.
  * - depth_of_discharge(%)
    - Depth of Discharge
    - 90.
  * - soc_minimum(between 0-1)
    - SoC minimum
    - 0.1
  * - time_unit
    - Minimum time unit in mins
    - 15
  * - milp_flag
    - Use MILP
    - true
  * - solver
    - Solver to use['ort','dqn'(RL)]
    - "ort"
  * - test_flag
    - False
    - false
  * - min_battery_capacity_factor
    - Minimum Battery Capacity Factor
    - 0.8
  * - battery_cost_per_kWh
    - Battery Cost per kWh
    - 20.
  * - reduction_coefficient
    - Battery reduction coefficient
    - 0.99998
  * - degradation_period
    - Degradation period in days
    - 7
  * - weight_emission
    - Emission saving weightage(between 0-1)
    - 1.0
  * - weight_degradation
    - Degradation weightage(between 0-1)
    - 0.0
  * - weight_price
    - Cost saving weightage(between 0-1)
    - 1.0
  * - degradation_flag
    - Flag to enable battery degradation
    - false
  * - seed
    - Seed of the experiment
    - 40
  ```
</details>

### Running via CLI

To run locally, run the following command on your `EnCortex` docker container `bash`(if using [Docker container](setup/docker)) or your terminal/jupyter notebook cell(if installed via [pip](setup/pip))

```bash
python examples/microgrid_example.py
```

> Visualizations through the CLI can only be viewed when run locally(through Streamlit). The final numbers are visible in the logs.


Other variations of the configuration can be found below. In the tab mentioned `Degradation(enabled)`, the degradation of a battery during charge/discharge cycles is considered and is enabled by the setting flag `degradation_flag` as `true`.

````{eval-rst}
.. tabs::

   .. tab:: Degradation(disabled)
      **config.yml**

      .. code:: yaml

        storage_capacity: 10.
        efficiency: 1.
        soc_initial: 1.
        depth_of_discharge: 90.
        soc_minimum: 0.1
        time_unit: 15
        milp_flag: true
        solver: "ort"
        test_flag: false
        min_battery_capacity_factor: 0.8
        battery_cost_per_kWh: 20.
        reduction_coefficient: 0.99998
        degradation_period: 7
        weight_emission: 1.0
        weight_degradation: 0.0
        weight_price: 1.0
        degradation_flag: true
        seed: 40
        train_path: "data/train.csv"
        test_path: "data/test.csv"

   .. tab:: Degradation(enabled)
      **config.yml**

      .. code:: yaml

        storage_capacity: 10.
        efficiency: 1.
        soc_initial: 1.
        depth_of_discharge: 90.
        soc_minimum: 0.1
        time_unit: 15
        milp_flag: true
        solver: "ort"
        test_flag: false
        min_battery_capacity_factor: 0.8
        battery_cost_per_kWh: 20.
        reduction_coefficient: 0.99998
        degradation_period: 7
        weight_emission: 1.0
        weight_degradation: 0.0
        weight_price: 1.0
        degradation_flag: false
        seed: 40
        train_path: "data/train.csv"
        test_path: "data/test.csv"
````
## Running on an [AML notebook](run/azureml-notebooks)

We also provide a notebook to ease modification and help visualize your results(click on the button below).

```{button-link} ../notebooks/microgrid.ipynb
:color: primary
:shadow:

MicroGrid Notebook
```

To run the notebook:

1. Download the notbook from this documentation(download button {octicon}`download;1em;sd-text-info` on the top of the page) and upload the notebook to AML studio
2. Connect to your compute instance(see [setup](setup/azureml/compute))
3. Select `Python3.8 - AzureML` as your environment
4. Run the pip instructions to install `EnCortex` (see [setup](setup/pip))

> Results and visualizations can be viewed directly on the notebook.

````{margin}
```{note}
   This is a one-time setup. As long as the compute environment is not changed, this step can be ignored after the first installation.
```
````

