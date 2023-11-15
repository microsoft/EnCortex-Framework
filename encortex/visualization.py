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

import argparse
import json
import os
import typing as t
from copy import deepcopy

import pandas as pd
import streamlit as st
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

CONTAINERS = {
    "container": st.container(),
    "expander": st.expander,
    "columns": st.columns,
    "sidebar": st.sidebar,
    "tabs": st.tabs,
    "plot": None,
}


def render(dir_path, container_config: t.Dict):
    # print("Container config: ", container_config)
    container_config = DictConfig(
        container_config, flags={"allow_objecs": True}
    )
    container_config = OmegaConf.to_container(container_config)
    element = container_config["_target_"]
    # container_config_copy = DictConfig(container_config, flags={'allow_objecs': True})
    # for k, v in container_config.items():
    #     container_config_copy[k] = v
    container_config_copy = deepcopy(container_config)

    for k in container_config.keys():
        if "_path" in k:
            assert "kwargs" in container_config.keys()
            if k.split("_")[-1] == "path":
                df = pd.read_csv(os.path.join(dir_path, container_config[k]))
                print(
                    "Key: ",
                    container_config["_".join(k.split("_")[:-1]) + "_col"],
                )
                container_config_copy[str("_".join(k.split("_")[:-1]))] = df[
                    container_config["_".join(k.split("_")[:-1]) + "_col"]
                ]

    container_config = deepcopy(container_config_copy)
    del container_config_copy
    kwargs = DictConfig({}, flags={"allow_objects": True})
    if "kwargs" in container_config.keys():
        if not container_config["single_arg"]:
            for k in container_config["kwargs"]:
                kwargs[k] = container_config[k]
        else:
            kwargs["_args_"] = [
                container_config[list(container_config["kwargs"].keys())[0]]
            ]
        kwargs["_target_"] = element
    else:
        kwargs = container_config

    instantiate(kwargs)


def parse_config(config: t.Union[str, t.Dict]):
    """A visualization config parser for automatically generating plots for the experiments run on the framework.

    config structure
    1. Level 1 - section wise dashboard structure
    2. Level 2 - If any, horizontal stacking
    3. Level 3-n - Plots

    dir path: path to directory containing everything.

    Args:
        config(t.Dict): The configuration to be parsed
    """
    print("In parse config")
    assert config is not None
    if isinstance(config, str):
        with open(config, "r") as f:
            extension = config.split(".")[-1]
            if extension == "json":
                config = json.load(f)
            elif extension in ["yaml", "yml"]:
                config = OmegaConf.load(f)
                print("Config: ", config)
            else:
                raise NotImplementedError(
                    f"{extension} not supported for encortex visualizer"
                )

    assert (
        "dir_path" in config.keys()
    ), f"Config should contain a path pointing towards all the data. Keys found: {config.keys()}"
    dir_path = config["dir_path"]

    for k in config.keys():
        if k not in ["dir_path"]:
            assert isinstance(config[k], (dict)) or OmegaConf.is_dict(
                config[k]
            ), f"{k} should be of type Dict, received {config[k]} of type {type(config[k])}"
            container_type = k.split("_")[1]
            assert (
                container_type in CONTAINERS.keys()
            ), f"Key {k} of type {container_type} not supported. Please provide one of {CONTAINERS.keys()}"

            if container_type in ["tabs"]:
                sub_containers = CONTAINERS[container_type](
                    list(config[k].keys())
                )

                for sub_container_config, sub_container in zip(
                    list(config[k].keys()), sub_containers
                ):
                    with sub_container:
                        render(dir_path, config[k][sub_container_config])
            elif container_type in ["columns"]:
                sub_containers = CONTAINERS[container_type](
                    len(list(config[k].keys()))
                )

                for sub_container_config, sub_container in zip(
                    list(config[k].keys()), sub_containers
                ):
                    with sub_container:
                        render(dir_path, config[k][sub_container_config])
            elif container_type in ["expander"]:
                with CONTAINERS[container_type](config[k]["header"]) as c:
                    for sub_k in config[k].keys():
                        if isinstance(
                            config[k][sub_k], (dict)
                        ) or OmegaConf.is_dict(config[k][sub_k]):
                            render(dir_path, config[k][sub_k])
            elif container_type in ["plot"]:
                render(dir_path, config[k])
            else:
                with CONTAINERS[container_type] as c:
                    render(dir_path, config[k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)

    args = parser.parse_args()
    config_file = args.config_file
    parse_config(config_file)
