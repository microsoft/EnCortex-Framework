# Known Issues

1. Crashing on AML notebooks

When running on AML notebooks, importing `EnCortex` crashes. This happens only when `rsome` is imported after `pytorch_lightning`. A simple fix is reversing the order.
Since [pre-commit](https://pre-commit.com/) runs `isort` (formatts the imports in the right order), use the comment `# isort: skip` to skip automatic sorting of `rsome` and `pytorch_lightning`.