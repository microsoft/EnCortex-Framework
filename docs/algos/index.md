# Developer Concepts

![EnCortex Architecture Diagram](../_static/EnCortex%20Architecture%20Diagram.jpg)

The above diagram represents the role `EnCortex` plays in the overall pipeline. Incoming streaming data is stored systematically in Azure Storage containers like Blob Storages and through
EnCortex, we can run different experiments and deploy trained algorithms using AzureML.

In this part of the documentation, we introduce some concepts in `EnCortex`, outline some of the algorithms and procedures to efficiently use `EnCortex`.

::::{card-carousel} 3

:::{card} Basics
:link: ./basics
:link-type: doc

Introduces Entity, Contract, Decision Unit and various other parts of EnCortex.
:::

:::{card} Adding new components to the framework
:link-type: doc
:link: ./new_components


In this page, we cover adding new entities to the framework.
:::

:::{card} Implementing an Environment
:link: ./env
:link-type: doc

This page covers step-by-step implementation of a new framework.
:::

::::