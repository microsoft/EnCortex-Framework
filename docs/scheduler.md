# Scheduler

The principle behind the need of a scheduler is to:

1. Schedule optimization phases as per the participating market deadlines
2. To define the requirement of data for each phase of the optimization.

## Design

1. Each market has a deadline and all the deadlines are compiled to give a phase wise rollout as a Tree/DAG.
2. Extract data requirements from each node of the DAG and give it to the data provider.

## Format and fields associated

1. Bid Deadline
2. Data timeline requirements: with reference to current day 12 AM
3. Commit slots: Time and slots

### Schedule Generation:

Sort Bid deadline + Data timeline: the first one is phase 1, rest if phase 2.

## Example

1. TAM 1: 1-2 PM deadline everyday with bid time +2 days 12AM to +10 11:45PM
2. TAM 2: 3-4 PM deadline every Wed,Thu,Fri with bid time +4d days 12AM to +10 days 11:45 PM

Monday: Phase 2 at 2PM (We assume that phase 1 has already occurred)
Tuesday: Phase 2 at 2PM
Wednesday: Phase 2 at 2PM, Phase 1 at 4PM
Thursday: Phase 2 at 2PM, Phase 1 at 4PM
Friday: Phase 2 at 2PM, Phase 1 at 4PM

![Schedule generation](./EnCortex7.jpg)


