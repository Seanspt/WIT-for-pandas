# wit-for-pandas
WIT-for-pandas is a wrapper of [whatif-tool](https://github.com/tensorflow/tensorboard/tree/master/tensorboard/plugins/interactive_inference) for non-tensorflow users.

# Install dependencies

WIT can be installed with the following commands.
``` bash
pip install witwidget
jupyter nbextension install --py --symlink --sys-prefix witwidget
jupyter nbextension enable --py --sys-prefix witwidget

# if you have a gpu card
pip install widwidget-gpu
jupyter nbextension install --py --symlink --sys-prefix witwidget
jupyter nbextension enable --py --sys-prefix witwidget

```

# Install start script
Copy the script to your ipython startup dir, create a new or restart a running kernel. An example can be found in the demo notebook.
``` bash
mkdir -p ~/.ipython/profile_default/startup/
cp whatif_wrapper.py ~/.ipython/profile_default/startup/
```