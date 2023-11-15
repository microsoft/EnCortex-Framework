import numpy as np
from rsome import ro

from encortex.contract import Contract
from encortex.decision_unit import DecisionUnit
from encortex.entity import Entity


def test_decision_unit():
    e1 = Entity((15, "m"), "Test Entity", 1, "", schedule="*/15 * * * *")
    e2 = Entity((15, "m"), "Test Entity", 2, "", schedule="*/15 * * * *")
    e3 = Entity((15, "m"), "Test Entity", 3, "", schedule="*/15 * * * *")
    e4 = Entity((15, "m"), "Test Entity", 5, "", schedule="*/15 * * * *")

    c1 = Contract(e1, e2, id=10, bidirectional=False)
    c2 = Contract(e3, e2, id=11, bidirectional=False)
    c3 = Contract(e4, e2, id=12, bidirectional=True)

    assert len(c1.edge) == 1, f"Edge length should be 1, {c1.edge}"
    assert len(c3.edge) == 2, f"Edge length should be 2, {c3.edge}"

    du = DecisionUnit([c1, c2, c3])
    assert du.timestep == np.timedelta64(15, "m"), "Timestep mismatch"

    # du.visualize_graph()
    model = ro.Model()
    schedule = du.get_schedule(
        np.datetime64("2020-01-01T00:00:00"),
        np.datetime64("2020-01-02T00:00:00"),
        model,
    )
    assert list(schedule[c1.id].keys())[0] == np.datetime64(
        "2020-01-01T00:00:00"
    ).astype("datetime64[m]")
    assert list(schedule[c1.id].keys())[-1] == np.datetime64(
        "2020-01-01T23:45:00"
    ).astype("datetime64[m]")
