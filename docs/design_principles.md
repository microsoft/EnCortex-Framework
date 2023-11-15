# Design Principles

This document contains some of the key design principles of various components in the framework - including the way they are intended to be used, how to currently use them and more.

# Components

1. Data Access
2. Component behaviour
3. Environment design
4. Algorithm design
5. Compute structure

# Design Principles of the Framework

1. Be both research and production ready - support things that would continually work(for production) and support iterative development(for research)
2. Configurable via yaml files
3. Scalability for high-compute algorithms
4. Universal environment design

# Key comments

1. Python based frameworks helps us scale to both quick and easy array processing while being able to orchestrate everything as a yaml file.
2. We use PostgresQL for our production-level data handling and sqlite3's in-memory database to handle training/simulation loads where the data is not dynamic
3. Gym based design because there's a clear flow of actions from time to time
4. Since we're supporting various class of algorithms, we will be providing first class support to all the algorithms.

