# Schedule

2 broad classes of market

1. DAM style = DAM, Term Ahead (T-start/end_time)

DAM [T - 12 <-> T - 10] for T to T+24*60 and 24 time slots
Term Ahead [T - 14*24 <-> T -  13*24] and  24 time slots
RTM[T - (60+15) <-> T - (60)] for T to T+15*60 and 1 timeslot


DSM - 45 mins prior


algo.optimize_full(forecast_price) -> {bid_volume, bid_price, solar.cap, }
"""
Satisfy the earliest guarantee required (for example for DAM and RTM, DAM is the earliest guarantee required)
"""

algo.optimize_partial()
"""
Satisfy partial market since the other part remains fixed (Only RTM in terms of DAM)
"""

algo.optimize_all()
"""
Provide the cutting planes and values || provide a range of values and map the action to each value
"""


DAM and RTM

1. optimize_full  between 12 PM and 2 PM on day zero:
2. optimize_partial

Schedule table

12:30 PM - optimize_full
11:30 PM - optimize_partial (freeze all the promised values at this time and optimize for the rest)
11:45 PM - optimize_all(get all possible values with minimum penalties)
12:30 AM - optimize_partial (freeze all the promised values at this time and optimize for the rest)
12:45 AM - optimzize_all(get all possible values with minimum penalties)


DAM, RTM and Term Ahead

1. optimize_full

# Gym

1. Bid full
2. Simulate closing (for research)
3. Bid partial
4. Simulate closing (for research)
5. Fetch promised values
6. Optimize for actual value
7. Calculate metrics

Example:

env.register([DAM( * args), RTM( * args)], DSM( * args), [ Solar(*kwargs), Wind( *args) ]

```python
from collections import OrderedDict
import typing as t

# def f(time: int) -> t.List[dict]:
#     if is_time_in_window1(time): #window 1 for DAM, Term Ahead



class DAM:

    def __init__(self) -> None:
        self.abstract_deadline = OrderedDict(
            #"2PM":"buy/sell for 96 slots"
            "14": [(i, v[i], p[i]) for i in range(96)],
            "23:30": [(time("01:00"), dsm_volume)]   ,
        )

        self.frequency = "daily"
        self.horizon = "24"
        self.timesteps = "4"
        self.num_timesteps = self.horizon * self.timesteps


class Env:
    def dsm_deadline():
        pass
"""
DAM() 12-2PM, RTM(), TAM() 1-2PM

1. Consistency check: Common bidding windows
"""

```