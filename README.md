# df2onehot

[![Python](https://img.shields.io/pypi/pyversions/df2onehot)](https://img.shields.io/pypi/pyversions/df2onehot)
[![PyPI Version](https://img.shields.io/pypi/v/df2onehot)](https://pypi.org/project/df2onehot/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/df2onehot/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/df2onehot/month)](https://pepy.tech/project/df2onehot/month)
[![Downloads](https://pepy.tech/badge/df2onehot)](https://pepy.tech/project/df2onehot)
[![DOI](https://zenodo.org/badge/245003302.svg)](https://zenodo.org/badge/latestdoi/245003302)
[![Sphinx](https://img.shields.io/badge/Sphinx-Docs-Green)](https://erdogant.github.io/df2onehot/)
<!---[![BuyMeCoffee](https://img.shields.io/badge/buymea-coffee-yellow.svg)](https://www.buymeacoffee.com/erdogant)-->
<!---[![Coffee](https://img.shields.io/badge/coffee-black-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)-->

``df2onehot`` is a Python package to convert unstructured DataFrames into structured dataframes, such as one-hot dense arrays.

# 
**⭐️ Star this repo if you like it ⭐️**
#

#### Install df2onehot from PyPI

```bash
pip install df2onehot
```

#### Import df2onehot package

```python
from df2onehot import df2onehot
```
# 


### [Documentation pages](https://erdogant.github.io/df2onehot/)

On the [documentation pages](https://erdogant.github.io/df2onehot/) you can find detailed information about the working of the ``df2onehot`` with many examples. 

<hr> 

### Examples

```python
results = df2onehot(df)
```

```python
# Force features (int or float) to be numeric if unique non-zero values are above percentage.
out = df2onehot(df, perc_min_num=0.8)
```

```python
# Remove categorical features for which less then 2 values exists.
out = df2onehot(df, y_min=2)
```

```python
# Combine two rules above.
out = df2onehot(df, y_min=2, perc_min_num=0.8)
```


# 
* [Example: Process Mixed dataset](https://erdogant.github.io/df2onehot/pages/html/Examples.html#)
# 
* [Example: Extracting nested columns](https://erdogant.github.io/df2onehot/pages/html/Examples.html#extracting-nested-columns)
# 
* [Example: Setting custom dtypes](https://erdogant.github.io/df2onehot/pages/html/Examples.html#custom-dtypes)
#

<hr>

#### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://www.buymeacoffee.com/erdogant">Coffee</a> for this work, it is very appreciated :)
