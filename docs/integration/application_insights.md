# Azure Monitor

`EnCortex` uses python's logging module to log information, warnings and errors. Thus, we natively support integrating with [Azure Monitor](https://learn.microsoft.com/en-us/azure/azure-monitor/app/app-insights-overview?tabs=net). To integrate with Azure Monitor:

1. Create an Azure Monitor resource on the Azure Portal
2. Fetch your instrumentation key from the portal
3. Run the following command in your development shell: `python -m pip install opencensus-ext-azure`
4. Add the following code:
```python
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=00000000-0000-0000-0000-000000000000')
)
```