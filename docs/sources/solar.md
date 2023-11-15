# Solar

## Parameters

Max Capacity
Current Capacity

## Action space

Format = (t, action) - execute an action at time t

0 <= Action <= 1 || Discrete values between 0 and 1. These essentially cap the capacity at any point of time relative to the max capacity.

## Extendibility

1. Custom action space
2. Callbacks to intervene before and after events.
