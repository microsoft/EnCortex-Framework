# This Early-Access License Agreement (“Agreement”) is an agreement between the party receiving access to the Software Code, as defined below (such party, “you” or “Customer”) and Microsoft Corporation (“Microsoft”). It applies to the Hourly-Matching Solution Accelerator software code, which is Microsoft pre-release beta code or pre-release technology and provided to you at no charge (“Software Code”). IF YOU COMPLY WITH THE TERMS IN THIS AGREEMENT, YOU HAVE THE RIGHTS BELOW. BY USING OR ACCESSING THE SOFTWARE CODE, YOU ACCEPT THIS AGREEMENT AND AGREE TO COMPLY WITH THESE TERMS. IF YOU DO NOT AGREE, DO NOT USE THE SOFTWARE CODE.
#
# 1.	INSTALLATION AND USE RIGHTS.
#    a)	General. Microsoft grants you a nonexclusive, perpetual, royalty-free right to use, copy, and modify the Software Code. You may not redistribute or sublicense the Software Code or any use of it (except to your affiliates and to vendors to perform work on your behalf) through network access, service agreement, lease, rental, or otherwise. Unless applicable law gives you more rights, Microsoft reserves all other rights not expressly granted herein, whether by implication, estoppel or otherwise.
#    b)	Third Party Components. The Software Code may include or reference third party components with separate legal notices or governed by other agreements, as may be described in third party notice file(s) accompanying the Software Code.
#
# 2.	USE RESTRICTIONS. You will not use the Software Code: (i) in a way prohibited by law, regulation, governmental order or decree; (ii) to violate the rights of others; (iii) to try to gain unauthorized access to or disrupt any service, device, data, account or network; (iv) to spam or distribute malware; (v) in a way that could harm Microsoft’s IT systems or impair anyone else’s use of them; (vi) in any application or situation where use of the Software Code could lead to the death or serious bodily injury of any person, or to severe physical or environmental damage; or (vii) to assist or encourage anyone to do any of the above.
#
# 3.	PRE-RELEASE TECHNOLOGY. The Software Code is pre-release technology. It may not operate correctly or at all. Microsoft makes no guarantees that the Software Code will be made into a commercially available product or offering. Customer will exercise its sole discretion in determining whether to use Software Code and is responsible for all controls, quality assurance, legal, regulatory or standards compliance, and other practices associated with its use of the Software Code.
#
# 4.	AZURE SERVICES.  Microsoft Azure Services (“Azure Services”) that the Software Code is deployed to (but not the Software Code itself) shall continue to be governed by the agreement and privacy policies associated with your Microsoft Azure subscription.
#
# 5.	TECHNICAL RESOURCES.  Microsoft may provide you with limited scope, no-cost technical human resources to enable your use and evaluation of the Software Code in connection with its deployment to Azure Services, which will be considered “Professional Services” governed by the Professional Services Terms in the “Notices” section of the Microsoft Product Terms (available at: https://www.microsoft.com/licensing/terms/product/Notices/all) (“Professional Services Terms”). Microsoft is not obligated under this Agreement to provide Professional Services. For the avoidance of doubt, this Agreement applies solely to no-cost technical resources provided in connection with the Software Code and does not apply to any other Microsoft consulting and support services (including paid-for services), which may be provided under separate agreement.
#
# 6.	FEEDBACK. Customer may voluntarily provide Microsoft with suggestions, comments, input and other feedback regarding the Software Code, including with respect to other Microsoft pre-release and commercially available products, services, solutions and technologies that may be used in conjunction with the Software Code (“Feedback”). Feedback may be used, disclosed, and exploited by Microsoft for any purpose without restriction and without obligation of any kind to Customer. Microsoft is not required to implement Feedback.
#
# 7.	REGULATIONS. Customer is responsible for ensuring that its use of the Software Code complies with all applicable laws.
#
# 8.	TERMINATION. Either party may terminate this Agreement for any reason upon (5) business days written notice. The following sections of the Agreement will survive termination: 1-4 and 6-12.
#
# 9.	ENTIRE AGREEMENT. This Agreement is the entire agreement between the parties with respect to the Software Code.
#
# 10.	GOVERNING LAW. Washington state law governs the interpretation of this Agreement. If U.S. federal jurisdiction exists, you and Microsoft consent to exclusive jurisdiction and venue in the federal court in King County, Washington for all disputes heard in court. If not, you and Microsoft consent to exclusive jurisdiction and venue in the Superior Court of King County, Washington for all disputes heard in court.
#
# 11.	DISCLAIMER OF WARRANTY. THE SOFTWARE CODE IS PROVIDED “AS IS” AND CUSTOMER BEARS THE RISK OF USING IT. MICROSOFT GIVES NO EXPRESS WARRANTIES, GUARANTEES, OR CONDITIONS. TO THE EXTENT PERMITTED BY APPLICABLE LAW, MICROSOFT EXCLUDES ALL IMPLIED WARRANTIES, INCLUDING MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT.
#
# 12.	LIMITATION ON AND EXCLUSION OF DAMAGES. IN NO EVENT SHALL MICROSOFT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE BY CUSTOMER, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. TO THE EXTENT PERMITTED BY APPLICABLE LAW, IF YOU HAVE ANY BASIS FOR RECOVERING DAMAGES UNDER THIS AGREEMENT, YOU CAN RECOVER FROM MICROSOFT ONLY DIRECT DAMAGES UP TO U.S. $5,000.
#
# This limitation applies even if Microsoft knew or should have known about the possibility of the damages.

import glob
import os
from pathlib import Path

try:
    pass

    from tqdm.rich import tqdm
except ImportError:
    # Rich not installed, we only throw an error
    # if the progress bar is used
    from tqdm import tqdm

import mlflow
from azure.storage.blob import BlobServiceClient
from azureml.core import Dataset, Workspace
from azureml.data import TabularDataset

from encortex.azure import get_config

import typer

azure_app = typer.Typer()


def get_connection_string(connection_string: str) -> str:
    if connection_string is None:
        if connection_string is None:
            config = get_config()
            connection_string = config["blob_storage"]
    return connection_string


@azure_app.command()
def download_from_azure(
    out_path: str,
    container_name: str,
    blob_name: str,
    connection_string: str = None,
):
    assert container_name is not None
    assert blob_name is not None
    connection_string = get_connection_string(connection_string)
    assert connection_string is not None
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string
    )

    blob_client_instance = blob_service_client.get_container_client(
        container=container_name
    )

    blob_data = blob_client_instance.download_blob(blob_name)
    data = blob_data.readall()

    with open(os.path.join(out_path, blob_name), "wb") as f:
        f.write(data)


@azure_app.command()
def upload_to_azure(
    file_path: str,
    container_name: str,
    connection_string: str = None,
) -> str:
    connection_string = get_connection_string(connection_string)
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string
    )

    file_paths = glob.glob(file_path)
    for file_path in file_paths:
        blob_name = Path(str(file_path)).name
        blob_client_instance = blob_service_client.get_blob_client(
            container_name, blob_name, snapshot=None
        )
        size = os.stat(file_path).st_size
        with tqdm.wrapattr(open(file_path, "rb"), "read", total=size) as data:
            blob_client_instance.upload_blob(data)


@azure_app.command()
def version_dataset(
    path: str,
    azure_path: str,
    dataset_name: str,
    dataset_description: str,
    create_new_version: bool,
    subscription_id: str = None,
    resource_group: str = None,
    workspace_name: str = None,
    load_from_config: str = "",
) -> TabularDataset:
    if load_from_config != "":
        ws = Workspace.from_config(load_from_config)
    else:
        assert all(
            (subscription_id, resource_group, workspace_name)
        ), f"Expected Subscription ID, Resource Group and Workspace name to be non empty, received SID: {subscription_id}, RG: {resource_group}, WN: {workspace_name}"
        ws = Workspace(subscription_id, resource_group, workspace_name)

    datastore = ws.get_default_datastore()
    upload_to_azure(path, azure_path, dataset_name, get_connection_string(None))

    try:
        dataset = Dataset.Tabular.from_delimited_files(
            path=(datastore, azure_path + dataset_name)
        )
    except ImportError as e:
        print(
            "Run pip install azureml-dataset-runtime --upgrade to use this functionality."
        )

    dataset = dataset.register(
        workspace=ws,
        name=dataset_name,
        description=dataset_description,
        create_new_version=create_new_version,
    )
    return dataset


@azure_app.command()
def version_model(
    filepath: str,
    model_name: str,
    load_from_config: str,
    subscription_id: str = None,
    resource_group: str = None,
    workspace_name: str = None,
):
    if load_from_config != "":
        if load_from_config is None:
            load_from_config = get_config()["aml_config"]
        ws = Workspace.from_config(load_from_config)
    else:
        assert all(
            (subscription_id, resource_group, workspace_name)
        ), f"Expected Subscription ID, Resource Group and Workspace name to be non empty, received SID: {subscription_id}, RG: {resource_group}, WN: {workspace_name}"
        ws = Workspace(subscription_id, resource_group, workspace_name)

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    return mlflow.register_model(filepath, model_name)


def load_model(
    uri: str,
    subscription_id: str = None,
    resource_group: str = None,
    workspace_name: str = None,
    load_from_config: str = "",
):
    if load_from_config != "":
        ws = Workspace.from_config(load_from_config)
    else:
        assert all(
            (subscription_id, resource_group, workspace_name)
        ), f"Expected Subscription ID, Resource Group and Workspace name to be non empty, received SID: {subscription_id}, RG: {resource_group}, WN: {workspace_name}"
        ws = Workspace(subscription_id, resource_group, workspace_name)

    mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

    model = mlflow.pyfunc.load_model(model_uri=uri)
    return model
