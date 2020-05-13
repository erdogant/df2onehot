# df2onehot

[![Python](https://img.shields.io/pypi/pyversions/df2onehot)](https://img.shields.io/pypi/pyversions/df2onehot)
[![PyPI Version](https://img.shields.io/pypi/v/df2onehot)](https://pypi.org/project/df2onehot/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/erdogant/df2onehot/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/df2onehot/month)](https://pepy.tech/project/df2onehot/month)
[![Downloads](https://pepy.tech/badge/df2onehot)](https://pepy.tech/project/df2onehot)
[![Coffee](https://img.shields.io/badge/-coffee-grey.svg)](https://erdogant.github.io/donate/?currency=USD&amount=5)

* df2onehot is Python package

### Contents
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Contribute](#-contribute)
- [Maintainers](#-maintainers)
- [License](#-copyright)

### Installation
* Install df2onehot from PyPI (recommended). df2onehot is compatible with Python 3.6+ and runs on Linux, MacOS X and Windows. 
* It is distributed under the MIT license.

#### Quick Start
```
pip install df2onehot
```

* Alternatively, install df2onehot from the GitHub source:
```bash
git clone https://github.com/erdogant/df2onehot.git
cd df2onehot
python -U setup.py install
```  

#### Import df2onehot package
```python
import df2onehot
```

#### Example:

```python
df = df2onehot.import_example()
# Convert
out = df2onehot.df2onehot(df)
```

```python
# Force features (int or float) to be numeric if unique non-zero values are above percentage.
out = df2onehot.df2onehot(df, perc_min_num=0.8)
```

```python
# Remove categorical features for which less then 2 values exists.
out = df2onehot.df2onehot(df, y_min=2)
```

```python
# Combine two rules above.
out = df2onehot.df2onehot(df, y_min=2, perc_min_num=0.8)
```


#### Maintainers
* Erdogan Taskesen, github: [erdogant](https://github.com/erdogant)
* Contributions are welcome.
* If you wish to buy me a <a href="https://erdogant.github.io/donate/?currency=USD&amount=5">Coffee</a> for this work, it is very appreciated :)
