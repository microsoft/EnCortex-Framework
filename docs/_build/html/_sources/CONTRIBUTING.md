# Contributing

## Install pre-commit

1. `pip install pre-commit`
2. `pre-commit install`

## Adding a new file

Ensure the `LICENSE` is correctly added to the start of the file. To setup this tool,

1. Pull the docker image: `docker pull ghcr.io/google/addlicense:latest`
2. Run the following command from the root repository of EnCortex - `docker run -it -v ${PWD}:/src ghcr.io/google/addlicense -f LICENSE encortex/`

## Creating a new branch and pushing changes

After cloning the latest main commit,

1. `git switch -C alias/feature`
2. `git add <file/folder name>`
3. `git commit -m "<Message>"`
4. In the case of failed pre-commit runs/tests, the logs of these contains the files/folders causing the fails - `git add` these files/folders again and the pre-commit runs should be successful.
5. `git push origin alias/feature`
6. `Submit a PR`

While pushing [docs](contributing/docs), make sure to run `make clean && make html` inside the `docs/` folder before pushing.

(contributing/docs)=
## Writing Docs

We use [Sphinx](https://www.sphinx-doc.org/) and the [Executable Books Project](https://ebp.jupyterbook.org/) to build our docs. [**This link**](https://sphinx-book-theme.readthedocs.io/en/stable/) contains how to get stared with writing docs and a [kitchen-sink](https://sphinx-book-theme.readthedocs.io/en/stable/reference/kitchen-sink/index.html) with [other theme-specific elements](https://sphinx-book-theme.readthedocs.io/en/stable/reference/special-theme-elements.html) and [other extensions](https://sphinx-book-theme.readthedocs.io/en/stable/reference/extensions.html) to make the docs more user-friendly.

### Build Docs

1. Install the required libraries to build the docs : `pip install -r docs/requirements.txt`.
2. `make html` inside the `docs/` folder builds the documentation locally. To view the docs, install the [Live Server extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) on VSCode to browse through it.