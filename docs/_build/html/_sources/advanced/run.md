## Develop locally(Internal-use only)

### Approach 1: Pip


1. Run `pip install artifacts-keyring` and setup `pip.conf` file in in your root folder where the pip commands will be run from
2. In the `pip.conf`, setup your Personal Access Token([docs](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=Linux#create-a-pat)) and your `pip.conf` should look like
```
[global]
extra-index-url=https://encortex-feed:<YOUR_PAT>@pkgs.dev.azure.com/MSREnergy/EnCortex/_packaging/encortex-feed/pypi/simple
```

3. Install via pip: `pip install encortex` or `pip install encortex --upgrade`
4. If this doesn't work, `pip install encortex --index-url https://pkgs.dev.azure.com/MSREnergy/EnCortex/_packaging/encortex-feed/pypi/simple` and proceed to login

In case of an existing installation, upgrade to the latest wheel by running:

```bash
pip install encortex --index-url https://pkgs.dev.azure.com/MSREnergy/EnCortex/_packaging/encortex-feed/pypi/simple
```


(setup/docker)=
### Approach 2: Docker

To work locally, follow the instructions below to use our Docker image:

1. Pull the image from `EnCortex` Azure Container Registry by running the following commands in your terminal:

```bash
docker pull b2fb764f7f0b4a5d979cd3ed8d5ba0db.azurecr.io/encortex
docker tag b2fb764f7f0b4a5d979cd3ed8d5ba0db.azurecr.io/encortex encortex
```

2. To access the container's REPL(terminal) of the docker container running `EnCortex`, run the following command on your terminal:

````{margin}
```{note}
The state of this docker container is not saved once you exit. Remove `--rm` if you want the state of the container to be saved on exit. Remember to access it again by the container id and not image name.
```
````

```bash
docker run --rm --entrypoint bash -v ./:/workspace encortex
```

````{eval-rst}
.. note::

   This command mounts your current working directory to the encortex workspace. You can replace the `./` with another directory path to mount a different directory.
````

1. Run any of the pre-installed environments from the REPL(terminal), for example(more tutorials can be found [here](../tutorials/index.md):

```bash
encortex_run_ba config.yaml
```
