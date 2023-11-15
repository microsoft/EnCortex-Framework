# EnCortex - Synapse Setup

Azure Synapse Analytics helps bring together data integration and big data analytics - all in one place. With `EnCortex` installed within Azure Synapse, this helps you analyse, explore and run experiments on your own data without any hassle.

## Deployment - Azure Synapse

1. Create your Azure Synapse workspace by following the instructions [here](https://docs.microsoft.com/en-us/azure/synapse-analytics/quickstart-create-workspace#create-a-synapse-workspace)
2. Open Synapse studio - login via your azure resource admin account and assign the relevant people working on this as `Synapse Administrator/Contributor`

## Creating Spark pool

1. Create a new Apache Spark pool within the Synapse workspace under `Manage -> Apache Spark Pool -> +New`
2. Review the compute type under basic settings based on the need of the experiment
3. Under advanced settings, `enable Allow Session Level Packages`
4. Review and Create the Spark Pool
5. Open the package settings of the newly created spark pool - upload the encortex wheel attached, click on apply and you're good to go!

## Tutorials

Visit the tutorials section
