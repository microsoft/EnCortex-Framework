# EnCortex - AzureML

Empower data scientists and developers to build, deploy, and manage high-quality models faster and with confidence. Accelerate time to value with industry-leading machine learning operations (MLOps), open-source interoperability, and integrated tools. Innovate on a secure, trusted platform designed for responsible AI applications in machine learning.

## Deployment - AzureML studio and workspace

1. Visit [here](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources) to create your AzureML workspace
2. Depending on your computational needs, create a compute instance/cluster on your AzureML studio.
3. Download the `config.json` from the Azure portal of your Azure ML resource
4. Create your environment: Install EnCortex from <!--TODO: VB Add accessible link to how to install here-->
5. Use MLFlow to track your experiments: `encortex.logger.get_experiment_logger - use "mlflow" as logger_name`
6. In case of framework changes, use `build_and_push.sh` to rebuild and push the `encortex` docker image and can directly be used in the Azure ML environment as shown in the [azure-ml.ipynb](../notebooks/azure-ml.ipynb)

## Tutorial

Follow the tutorial noteboks to run `EnCortex` experiments on Azure ML