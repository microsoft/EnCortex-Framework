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

"""Action transform related utilities for different scaling and transform attributes
"""
import typing as t

import numpy as np
import pandas as pd
from windpowerlib import WindTurbine


def irradiance_to_generation(
    global_horizontal_irradiance: np.float32,
    day_of_the_year: int,
    latitude: np.float32,
    area: np.float32 = None,
    scaling_factor: np.float32 = 1.0,
) -> np.float32:
    """Irradiance to solar generation conversion

    \delta = 23.45 * \sin(360 * (284 + day_of_the_year) / 365)
    \alpha = 90 - latitude + delta
    \beta = latitude
    Power = Horizontal Irradiance * sin(\alpha+\beta) / \sin(\alpha) * Area * Scaling Factor

    Args:
        global_horizontal_irradiance (np.float32): Horizontal Irradiance at a particular area
        day_of_the_year (int): Day of the year (1-365/366)
        latitude (np.float32): Latitude of the area of solar panels(degrees)
        area (np.float32-Optional): Area of the solar panels in m^2

    Returns:
        np.float32: theoretical power generation in watts if area is provided or in watts/m^2 if area is not provided
    """
    delta = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_the_year) / 365))
    alpha = 90 - latitude + delta
    beta = latitude

    global_perpendicular_solar_irradiance = (
        global_horizontal_irradiance
        * np.sin(np.deg2rad(alpha + beta))
        / np.sin(np.deg2rad(alpha))
    )
    if area is None:
        return global_perpendicular_solar_irradiance * scaling_factor
    else:
        return global_perpendicular_solar_irradiance * area * scaling_factor


def wind_speed_to_generation(
    power_curve: pd.DataFrame,
    value: np.float32,
    WIND_SPEED: str = "wind_speed",
    POWER: str = "power",
):
    """Wind speed to generation conversion

    Args:
        power_curve (pd.DataFrame): Wind power curve
        value (np.float32): Wind speed in m/s

    Returns:
        np.float32: theoretical power generation in watts
    """
    return np.interp(value, power_curve[POWER], power_curve[POWER])


def get_power_curve(
    manufacturer: str = "Vestas",
    turbine_type: str = "V100/1800",
    *args,
    **kwargs
):
    """Get wind power curve. https://openenergy-platform.org/dataedit/view/supply/wind_turbine_library

    Args:
        manufacturer (str): Wind turbine manufacturer
        turbine_type (str): Wind turbine type

    Returns:
        pd.DataFrame: Wind power curve
    """
    turbine = WindTurbine(manufacturer, turbine_type, *args, **kwargs)
    return pd.DataFrame(
        {
            "wind_speed": turbine.power_curve.wind_speed,
            "power": turbine.power_curve.value,
        }
    )


def vectorize_dict(input_dict: t.Dict) -> np.ndarray:
    def _extract_items(input_d: t.Dict, l: t.List = None):
        if l is None:
            l = []
        for k, v in input_d.items():
            if isinstance(v, t.Dict):
                l = _extract_items(v, l)
            else:
                l.append(v)
        return l

    l = _extract_items(input_dict)
    return np.hstack(l)
