import numpy as np
from rsome import ro

from encortex.contract import Contract
from encortex.entity import Entity


def test_contract():
    e1 = Entity((15, "m"), "Test Entity", 1, "", schedule="*/15 * * * *")
    e2 = Entity((15, "m"), "Test Entity", 2, "", schedule="*/15 * * * *")

    c1 = Contract(e1, e2, bidirectional=False)
    assert len(c1.edge) == 1, f"Edge length should be 1, {c1.edge}"

    c2 = Contract(e1, e2, bidirectional=True)
    assert len(c2.edge) == 2, f"Edge length should be 2, {c2.edge}"

    model = ro.Model()
    c1.get_schedule(
        np.datetime64("2020-01-01T00:00:00"),
        np.datetime64("2020-01-01T00:00:00"),
        model,
    )

    model = ro.Model()
    variables = c1.get_target_variables(
        np.datetime64("2020-01-01T00:00:00"),
        np.datetime64("2020-01-02T00:00:00"),
        model,
    )
    assert len(list(variables[0])) == 24 * 4, f"Variables: {len(variables)}"
    assert len(list(variables[1])) == 24 * 4, f"Variables: {len(variables)}"
