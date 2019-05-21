# Install dependencies

``` bash
conda install tensorflow

pip install witwidget
jupyter nbextension install --py --symlink --sys-prefix witwidget
jupyter nbextension enable --py --sys-prefix witwidget

# if you have a gpu card
pip install widwidget-gpu
jupyter nbextension install --py --symlink --sys-prefix witwidget
jupyter nbextension enable --py --sys-prefix witwidget

```

# Install start script
``` bash
# if there is no ~/.ipython, feel free to create one!
cp whatif_wrapper.py ~/.ipython/profile_default/startup/
```