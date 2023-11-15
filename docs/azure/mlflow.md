# EnCortex with MLFlow

MLFlow is our preferred model tracking and registry tool, in integration with Azure ML.

For plain MLFlow:

1. `pip install mlflow`
2. `from encortex.loggers import get_experiment_logger`
3. `experiment_logger = get_experiment_logger('mlflow', experiment_name='encortex-experiment-1') #Add more arguments as per your MLFlow configuration`

For MLFlow with Azure ML:

1. `pip install mlflow azureml-mlflow`
2. `from encortex.loggers import get_experiment_logger`
3. `mlflow.set_tracking_url(ws.get_mlflow_tracking_uri()) #If working on Azure ML hosted notebooks`
4. `experiment_logger = get_experiment_logger('mlflow', experiment_name='encortex-experiment-1') #Add more arguments as per your MLFlow configuration`

`experiment_logger` can `log_metics` for training metrics, `log_hyperparams` for experiment hyperparameters, `log_artifacts` for hosting images, multimedia, files, etc. and `log_hyperparams.

For model registry related functionality, use `experiment_logger.experiment.create_registered_model(...)`
