# Working with Notebooks on Azure ML

## Installing EnCortex

1. Run `pip install artifacts-keyring` and setup `pip.conf` file in in your root folder where the pip commands will be run from
2. In the `pip.conf`, setup your Personal Access Token([docs](https://docs.microsoft.com/en-us/azure/devops/organizations/accounts/use-personal-access-tokens-to-authenticate?view=azure-devops&tabs=Linux#create-a-pat)) and your `pip.conf` should look like
```
[global]
extra-index-url=https://encortex-feed:<YOUR_PAT>@pkgs.dev.azure.com/MSREnergy/EnCortex/_packaging/encortex-feed/pypi/simple
```

3. Install via pip: `pip install encortex` or `pip install encortex --upgrade`
4. If this doesn't work, `pip install encortex --index-url https://pkgs.dev.azure.com/MSREnergy/EnCortex/_packaging/encortex-feed/pypi/simple` and proceed to login

## Running code on AML

1. Connect to Jupyter notebook through the browser or through Azure ML extension on VSCode.
2. Connect to the relevant compute instance to run your notebook code and code away!