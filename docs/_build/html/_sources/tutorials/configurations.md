# Configurations

In this section, we show how you can play with the different configurations in the pre-defined `Energy Arbitrage`  and `MicroGrid` Scenario. Before proceeding, we highly recommend going through the [Energy Arbitrage AML Notebook](../notebooks/scenario1RL.ipynb) and the [MicroGrid AML Notebook](../notebooks/scenario2.ipynb). Here, we define all the configurable parameters mentioned in the different steps of the notebook.


## 1. Optimization Algorithms :

As mentioned in the notebook in Step 2, we support three different class of algorithms. Deterministic solutions are attained using Mixed Integer Linear Programming (MILP). Simulated Annealing (SA) and Deep Reinforcement Learning (DRL) algorithms provide stochastic solutions. Each of these optimization techniques comes with different solvers and algorithms. Our `EnCortex` framework supports :

- __MILP__ : OR-Tools ("ort"), Gurobi ("grb"), Cplex ("cpx"), CyLP ("clp"), ECOS ("eco"), MOSEK ("msk"). Provide the short-hand solver names mentioned in brackets to solve the problem using MILP.

- __DRL__ : Advantage Actor Critic ("a2c"), Proximal Policy Optimization ("PPO"), Deep Q Network("dqn"), if the problem is solved for discrete action spaces. To have continuous action spaces, we recommend using Deep Deterministic Policy Gradient ("ddpg")  and Twin Delayed DDPG ("td3"). Also, our framework supports the use of a2c,PPO and dqn on continuous action spaces as well by discretizing the continuous actions into different bins. Use the shot-hand notations as the solver name to work with DRL based algorithms

- __SA__ : We support the SciPy based Dual Simulated Annealing for the problem. Users can play with the different hyperparameters provided. Here, there is no need to provide names for solver.

## 2. Choosing Importance Weights in Objectives :

Users can play with the importance weights of `carbon emissions`, `cost` and `degradation` to attain an optimal solution handling a joint optimization. We recommend to plug in normalizad weights for the objectives having a value in the range of 0 to 1. For maximum monetary profit, put $\omega_{cost}$ as 1.0 and $\omega_{carbon}$ as 0.0. To have maximum carbon savings, just interchange the weights of the two. In all practical simulations, it's advisable to account for the degradation of the battery to formulate correct arbitrage opportunities. Hence, this provides an easy plug and play use-case to switch between single objectives to multi-objectives. Please, go through our [paper](../_static/nsdi23fall-paper705.pdf), where we show how Pareto Optimization plot is used to determine the point of optimality for joint optimization.

In the Micro Grid scenario, due to lack of carbon emissions data from the grid, we donot provide Carbon Optimization as an optimization option. This is seen from step 2 of the AML Notebook.

## 3. Varying Battery Configurations and use of multiple batteries :

`EnCortex` models battery as an electrical entity by taking in various configurable parameters required for its operations. Sending multiple jobs to the AML Compute CLuster with varying battery configurations can help determine the accurate battery sizing required to operate in the power system so as to maximize the objective. Following are the parameters provided by the EnCortex based Li-ion Battery model :
- Battery Specifications :
  - __storage_capacity :__ the battery capacity (in kWh)
  - __efficiency :__ here, charging and discharging efficiency (in %) taken the same/ if different take it differently
  - __depth_of_discharge :__ the maximum discharge (in %) percentage that can happen at a time, here 90%
  - __soc_minimum :__ the minimum state of charge of the battery, below which the battery should not be explored
  - __timestep :__ battery decision time steps
  - __action :__ battery actions -  Here the battery can take 3 different actions {Charge/Discharge/Stay idle} at the mentioned rates. (battery datasheet max rate specifications used for the purpose, can be tweaked to have either `multiple discrete` action spaces > 3 or even continuous action spaces by using `gym.spaces.Continuous`)
- Degradation based parameters :
  - __degradation_flag :__ whether to have degradation model in place or not for the batteries
  - __min_battery_capacity_factor :__ the battery capacity reduction percentage due to degradation, below which if capacity reduces due to overuse, battery doesnot stay at good optimal health
  - __battery_cost_per_kWh :__ the battery replacement cost (in $/kWh)
  - __reduction_coefficient :__ after every charge-discharge cycles over a certain period, the battery capacity reduced by the reduction coefficienct
  - __degradation_period_in_days :__ the period after which battery degrades (Here we take a period of 7 days)
- Initialization parameters :
  - __soc_initial :__ initial state of charge of the battery to run the test experiments
  - __test_flag :__ the flag initiates random initial state of charge of the battery during training runs/experiments to avoid overfitting

The AML notebook takes in all the parameters in the form of a list as shown in step 2. Feeding in n elements in the list for all the parameters, denote having n batteries in the microgrid each with different configurations.


## 4. Changing Main Utility Grid Data - indicating a different market price and carbon emission evaluation :

For the `Energy Arbitrage` scenario, we provide support for publicly available `California (CAISO)` and `UK` [data](https://microsoftapc-my.sharepoint.com/:f:/g/personal/t-vballoli_microsoft_com/Evd4JIo7F4hFjI9Y_MJPVEYBYc4iP2i-OND1gfoCx3xiIQ?e=hhzg4M) which comprises of `market prices` and `carbon emissions` along with `timestamps`. Any of these grid data can be interchangeably used for the purpose. Users can also __download data__ from other publicly available sites, upload the file into the `data` folder in the AML notebook directory space, reformat it to the required `csv` format as mentioned [here](../tutorials/battery_arbitrage.md), split the data into training and testing portions (here, already split) and use it to solve the energy arbitrage problem. __DataLoaders__ provided with the framework can also serve the purpose. Since only reinforcement learning requires training, and because of it  uncertain, susceptibility to noisiness and variability, we recommend using larger splits for training to have sufficient data to get trained on so as to produce optimal policies that can translate to the test dataset/environment.

As for the `MicroGrid` scenario, we provide [data]() for multiple (71) industrial sites recognized as microgrid with each site containing solar and Li-ion battery storage facilities meeting the consumer demands. A common pricing structure is used for all the sites. Data can be __directly downloaded__ from there and stored in the required `folder structure` and `csv` format in the AML Notebook directory. The inbuilt-__Dataloaders__ can also be used for the purpose. Properly split the data before being used for training in reinforcement learning based algorithms. Splittinf has been done for one site (site 71), use a similar 60:40 split for the other sites before feeding it to the algorithms

Step 3 in the notebook dives deeper into how to use them.

## 5. Use of Different Forecasts :

Decisions within the framework are based on data received from forecasting models/services. We have developed a unified forecasting module, that supports numerous state-of-the-art time-series forecasting techniques and also operators can bring their own forecaster model and integrate it. We support a wide range of techniques from simple heuristics to state-of-the-art algorithms, i.e.,
- __Noise forecast ("noise")__: gaussian noise is added to actual data} ,
- __Yesterday's forecast ("yesterdays")__: previous day's actual data as forecast for the next day,
- __Mean ("meanprev")__: mean of actual data for previous N days as forecast,
- __LightGBM ("lgbm")__, a decision tree based algorithm that builds a tree-like structure to perform regression and
- __N-BEATS ("nbeats")__, a purely neural network-based model.

Provide the shorthand notations while using the dataloaders for the `for_type` argument to use the respective forecaster. Step 3 in the notebook accounts for this.

## 6. Varying the time variables in the environment and set it up for the optimizer :

The environment takes in 3 different time variables for its use and later passes on the values to the optimizer.
- __timestep__ /__timeunit__ :This can be directly inferred from the data based on the data granularity. This signifies the granularity of taking actions for a defined time period.

- __horizon__ : Forecasts values are fed to the environment based on a given horizon. Commonly 24 hrs horizon is used for the experiments.

- __step_time_difference__ : This shows when to take the next optimization step. Lower step_time_difference generally produces optimal results. In RL, its recommended to use step_time_difference of the minimum time unit, so that more frequent decisions can be taken adding to the frameowk extensibility.

Vary these time variables from Step 6 in the notebook.

## 7. Visualization Parameters :

As a part of visualization feature, users can compare across different optimizers by choosing from a multi-select option, and select a day from a slider to delve depper into the decisions made for each of the time slots for that particular day. This feature is provided in Step 9 of the notebook.