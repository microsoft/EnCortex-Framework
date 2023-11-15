from encortex.sources.battery import Battery


def test_battery():
    battery = Battery(
        timestep=(15, "m"),
        name="Battery",
        id=2,
        description="Battery",
        storage_capacity=1100.0,
        charging_efficiency=0.95,
        discharging_efficiency=0.95,
        soc_initial=0.5,
        depth_of_discharge=90,
        soc_minimum=0.1,
        degradation_flag=False,
        min_battery_capacity_factor=0.8,
        battery_cost_per_kWh=200.0,
        reduction_coefficient=0.99998,
        degradation_period=7.0,
        test_flag=False,
        schedule="*/15 * * * *",
    )

    import numpy as np

    battery.act(
        np.datetime64("2020-01-01T00:00:00"),
        {"volume": [0, {"Ct": 0, "Dt": 1}]},
        False,
    )
    battery.act(
        np.datetime64("2020-01-01T00:15:00"),
        {"volume": [0, {"Ct": 1, "Dt": 0}]},
        False,
    )
    battery.act(
        np.datetime64("2020-01-01T00:30:00"),
        {"volume": [0, {"Ct": 1, "Dt": 0}]},
        False,
    )
    battery.act(
        np.datetime64("2020-01-01T00:45:00"),
        {"volume": [0, {"Ct": 1, "Dt": 0}]},
        False,
    )

    assert (
        battery.current_soc == 0.6497820688822623
    ), f"Mismatch in battery SOC: expected: 0.6497820688822623, received battery.current_soc"
