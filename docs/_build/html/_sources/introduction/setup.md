## Setup
<!-- In this section, we first cover setting up your [Azure Synapse workspace](https://azure.microsoft.com/en-in/products/synapse-analytics/) (see [below](setup/azure-synapse)). -->
<!--[Azure ML workspace](setup/azureml). Then, we show two([Docker](setup/docker),[Pip](setup/pip)) approaches to install `EnCortex` in your workspace. -->
In this section, we cover setting up your [AzureML](https://azure.microsoft.com/en-us/products/machine-learning/#product-overview) environment to run `EnCortex`. AzureML empowers data scientists and developers to build, deploy, and manage high-quality models faster and with confidence. AML also helps accelerate time to value with industry-leading machine learning operations (MLOps), open-source interoperability, and integrated tools. Hence, we recommend running `EnCortex` on Azure ML.

In the next few pages and sections, we cover:
1. We cover [setting up your AzureML studio](setup/azureml).
2. Setup [AzureML notebooks](run/azureml-notebooks)
3. [Installing](setup/pip) `EnCortex`
4. [Tutorials](../tutorials/index)


Please proceed next if this is your first time using `EnCortex` or you can head onto the [Advanced section](../advanced/index) to start developing with `EnCortex`.


(setup/azureml)=
### AzureML(AML)

Setup your AzureML studio. To do this, you'll need access to your resource group in Azure. In this setup, we cover setting up AzureML through the portal, but other setup methods can be found [here](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment).

1. Sign in to your [Azure Portal](https://ms.portal.azure.com) -> `+ Create a Resource` -> Search "Azure Machine Learning" and follow the instructions to setup your AzureML studio.
2. From the **Overview** tab on the sidebar of your now created Azure ML workspace, click on **Download config.json** button to download your AML `config.json` file.
3. Head onto [ml.azure.com](https://ml.azure.com) and sign in to your account to view the AML studio. Alternatively, you can directly access your studio from the **Overview** tab by clicking on the **Studio Web URL**.


(setup/azureml/compute)=
#### Compute Resources

Now that you've setup your AML studio, we now move on to creating a compute resource. A compute resource is VM-like entity that allows you to run code. Having a compute resource is mandatory to run any `EnCortex` experiment. Click on the **Compute** tab from the sidebar inside your AML studio and click on `+ New` button of the type of compute resource you want to spin up. We recommend selecting the following compute resources based on requirements:

1. For simple EDA and small workloads, we recommend a CPU Virtual Machine. From the sidebar, Compute -> *Compute Instance* -> Enter Compute Name -> CPU -> *STANDARD_D13_V2*
2. For larger workloads involving multiple parallel experiments, we recommend a GPU Virtual Machine.  From the sidebar, Compute -> *Compute Cluster* -> Enter Cluster name -> Dedicated tier -> *Standard_DS3_v2* .


You're now ready to setup AML notebooks.

(run/azureml-notebooks)=
### AzureML notebooks

AzureML notebooks provides a jupyter notebook interface to play and explore `EnCortex`. To run `EnCortex` on AzureML notebooks, follow the instructiosn below:

1. Make sure your compute is instantiated(see [previous section](setup/azureml/compute))
2. From your studio homepage, navigate to the `Notebooks` tab and click on the `+` icon to create a notebook.
3. Connect to the above created compute on the top of notebook through the dropdown titled `Compute Instance` and select the `Python 3.8 - AzureML` environment from the far-right dropdown.
4. Run the the installation instructions mentioned in [Setup through Pip](setup/pip) and you're ready to run EnCortex.


## Installation
Your AML notebook environment is now set. Now, in order to install `EnCortex`, we recommend the following approach:

(setup/pip)=
### Approach 1: Pip(AML Notebook)

>**Accessible by @Microsoft email aliases only.**

1. Download the latest wheel files from [here](https://microsoftapc-my.sharepoint.com/:f:/g/personal/t-vballoli_microsoft_com/Em4E46gRGyhCuORVvEzGf5MBESMfXKmbiGzCtXlf-nVCyQ?e=lDW1Bj)(Go to the `wheels` folder)
2. Create a folder on AML called `encortex_wheels` through the UI.
3. Upload the downloaded wheels to the `encortex_wheels` folder.
4. Run the following command in an AML notebook cell:

```bash
!pip install encortex_wheels/*.whl
```

To verify installation, run the following code in your AML notebook cell and it should give out a string like `0.1.23`:

```bash
import encortex
print(encortex.__version__)
```

You're now ready to use AzureML notebooks. To run pre-defined scenarios, head on to [Tutorials](../tutorials/index.md) and follow the AML Notebook instructions for each of the scenarios there.

<!-- (setup/azure-synapse)=
### Azure Synapse

Azure Synapse Analytics is a limitless analytics service that brings together data integration, enterprise data warehousing and big data analytics. It gives you the freedom to query data on your terms, using either serverless or dedicated options—at scale. Azure Synapse brings these worlds together with a unified experience to ingest, explore, prepare, transform, manage and serve data for immediate BI and machine learning needs.

Through Synapse, you can integrate any data from all your databases and use them directly within `EnCortex`.


#### Create Azure Synapse Worksace
1. Connect to your [Azure Portal](
    https://ms.portal.azure.com) -> Create a Resource -> Search for Synapse(Click on Azure Synapes Analytics) -> Fill the details -> Click on Create.
2. Head on to your newly created resource. In the overview tab on the sidebar, click on the `Workspace web URL` to access your Synapse Workspace Studio.

#### Create Apache Spark Pool

Now, let's create an Apache Spark Pool that acts as a compute to run `EnCortex` code.
1. From the sidebar, click on Manage -> Apache Spark Pools -> Click on `+New` to create.
2. For small scale experiments, we recommend `Memory Optimized` node family, `Medium` size node. After entering the Apache Spark Pool name, head onto `Additional Settings` on the next tab and enable `Session Level Packages`.
3. Click `Review + Create` and wait for a few minutes while the Spark Pool is created.

#### Install EnCortex

To install `EnCortex`, we recommend installing through Synapse's Workspace packages.
1. Head on to `Manage`->`Workspace Package`->`Upload .whl` file. Download the `encortex` and `dfdb` file from here and upload them.
2. Now, click on `Apache Spark pools` within the `Manage` tab from the sidebar and click on the `...` button when you hover over the Spark pool you created in the previous step.
3. Now, create your first notebook by heading onto `Develop` tab from the sidebar and click on the `+` -> `Notebook` on top to create a notebook or upload one from the [tutorials](../tutorials/index.md).

You can now head onto [tutorials](../tutorials/index.md) run any of the pre-defined scenarios. -->
