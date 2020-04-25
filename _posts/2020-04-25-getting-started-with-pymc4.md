---
title: 'Getting started with PyMC4'
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

*You can find the notebook for this article [here](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/master/bayesian_linear_regression_pymc4.ipynb). It is part of the [bayesian-machine-learning](https://github.com/krasserm/bayesian-machine-learning) repo on Github.*

I recently started to implement examples from previous articles with [PyMC3](https://docs.pymc.io/) and [PyMC4](https://github.com/pymc-devs/pymc4) (see [here](https://github.com/krasserm/bayesian-machine-learning/blob/master/README.md) for an overview). 
PyMC4 is still in an early state of development with only an alpha release available at the time of writing this article.
The following is an introduction to PyMC4 using examples from [Bayesian regression with linear basis function models](/2019/02/23/bayesian-linear-regression/). A corresponding PyMC3 implementation is available [here](https://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/master/bayesian_linear_regression_pymc3.ipynb).


```python
import logging
import pymc4 as pm
import numpy as np
import arviz as az

import tensorflow as tf
import tensorflow_probability as tfp

print(pm.__version__)
print(tf.__version__)
print(tfp.__version__)

# Mute Tensorflow warnings ...
logging.getLogger('tensorflow').setLevel(logging.ERROR)
```

    4.0a2
    2.2.0-dev20200414
    0.10.0-dev20200414



## Introduction to PyMC4

PyMC4 uses [Tensorflow Probability](https://www.tensorflow.org/probability) (TFP) as backend and PyMC4 random variables are wrappers around TFP distributions. Models must be defined as [generator](https://docs.python.org/3/glossary.html#term-generator) functions, using a `yield` keyword for each random variable. PyMC4 uses [coroutines](https://www.python.org/dev/peps/pep-0342/) to interact with the generator to get access to random variables. Depending on the context, it may sample values from random variables, compute log probabilities of observed values, ... and so on. Details are covered in the [PyMC4 design guide](https://github.com/pymc-devs/pymc4/blob/master/notebooks/pymc4_design_guide.ipynb). Model generator functions must be decorated with ` @pm.model` as shown in the following trivial example:


```python
@pm.model
def model(y):
    x = yield pm.Normal('x', loc=0, scale=10)
    y = yield pm.Normal('y', loc=x, scale=1, observed=y)
```

`pm.sample()` samples from the posterior using NUTS. Samplers other than NUTS or variational inference methods are not implemented yet.


```python
y = np.random.randn(30) + 3

trace = pm.sample(model(y), num_chains=3)
trace
```




    Inference data with groups:
    	> posterior
    	> sample_stats
    	> observed_data



The returned `trace` object is an ArviZ [`InferenceData`](https://arviz-devs.github.io/arviz/notebooks/XarrayforArviZ.html) object. It contains posterior samples, observed data and sampler statistics. The posterior distribution over `x` can be displayed with:


```python
az.plot_posterior(trace, var_names=['model/x']);
```


![png](/img/2020-04-25/output_7_0.png)


Models can also be nested i.e. used like other PyMC4 random variables.


```python
@pm.model
def MyNormal(name, loc=0, scale=10):
    x = yield pm.Normal(name, loc=loc, scale=scale)
    return x

@pm.model
def model(y):
    x = yield MyNormal('x')
    y = yield pm.Normal('y', loc=x, scale=1, observed=y)
    
trace = pm.sample(model(y), num_chains=3)
az.plot_posterior(trace, var_names=['model/MyNormal/x']);    
```


![png](/img/2020-04-25/output_9_0.png)


## Linear basis function models

I introduced regression with linear basis function models in a [previous article](/2019/02/23/bayesian-linear-regression/). To recap, a linear regression model is a linear function of the parameters but not necessarily of the input. Input $x$ can be expanded with a set of non-linear basis functions $\phi_j(x)$, where $(\phi_1(x), \dots, \phi_M(x))^T = \boldsymbol\phi(x)$, for modeling a non-linear relationship between input $x$ and a function value $y$.

$$
y(x, \mathbf{w}) = w_0 + \sum_{j=1}^{M}{w_j \phi_j(x)} = w_0 + \mathbf{w}_{1:}^T \boldsymbol\phi(x) \tag{1}
$$

For simplicity I'm using a scalar input $x$ here. Target variable $t$ is given by the deterministic function $y(x, \mathbf{w})$ and Gaussian noise $\epsilon$.

$$
t = y(x, \mathbf{w}) + \epsilon \tag{2}
$$

Here, we can choose between polynomial and Gaussian basis functions for expanding input $x$. 


```python
from functools import partial
from scipy.stats import norm

def polynomial_basis(x, power):
    return x ** power

def gaussian_basis(x, mu, sigma):
    return norm(loc=mu, scale=sigma).pdf(x).astype(np.float32)

def _expand(x, bf, bf_args):
    return np.stack([bf(x, bf_arg) for bf_arg in bf_args], axis=1)

def expand_polynomial(x, degree=3):
    return _expand(x, bf=polynomial_basis, bf_args=range(1, degree + 1))

def expand_gaussian(x, mus=np.linspace(0, 1, 9), sigma=0.3):
    return _expand(x, bf=partial(gaussian_basis, sigma=sigma), bf_args=mus)

# Choose between polynomial and Gaussian expansion
# (by switching the comment on the following two lines)
expand = expand_polynomial
#expand = expand_gaussian
```

For example, to expand two input values `[0.5, 1.5]` into a polynomial design matrix of degree `3` we can use


```python
expand_polynomial(np.array([0.5, 1.5]), degree=3)
```




    array([[0.5  , 0.25 , 0.125],
           [1.5  , 2.25 , 3.375]])



The power of `0` is omitted here and covered by a $w_0$ in the model.

## Example dataset

The example dataset consists of `N` noisy samples from a sinusoidal function `f`.


```python
import matplotlib.pyplot as plt
%matplotlib inline

from bayesian_linear_regression_util import (
    plot_data, 
    plot_truth
)

def f(x, noise=0):
    """Sinusoidal function with optional Gaussian noise."""
    return 0.5 + np.sin(2 * np.pi * x) + np.random.normal(scale=noise, size=x.shape)

# Number of samples
N = 10

# Constant noise 
noise = 0.3

# Noisy samples 
x = np.linspace(0, 1, N, dtype=np.float32)
t = f(x, noise=noise)

# Noise-free ground truth 
x_test = np.linspace(0, 1, 100).astype(np.float32)
y_true = f(x_test)

plot_data(x, t)
plot_truth(x_test, y_true)
```


![png](/img/2020-04-25/output_16_0.png)


## Implementation with PyMC4

### Model definition

The model definition directly follows from Eq. $(1)$ and Eq. $(2)$ with normal priors over parameters. The size of parameter vector `w_r` ($\mathbf{w}_{1:}$ in Eq. $(1)$) is determined by the number of basis functions and set via the `batch_stack` parameter. With the above default settings, it is 3 for polynomial expansion and 9 for Gaussian expansion.


```python
import tensorflow as tf

@pm.model
def model(Phi, t, sigma=noise):
    """Linear model generator.
    
    Args:
    - Phi: design matrix (N,M)
    - t: noisy target values (N,)
    - sigma: known noise of t
    """

    w_0 = yield pm.Normal(name='w_0', loc=0, scale=10)
    w_r = yield pm.Normal(name='w_r', loc=0, scale=10, batch_stack=Phi.shape[1])
    
    mu = w_0 + tf.tensordot(w_r, Phi.T, axes=1)
    
    yield pm.Normal(name='t_obs', loc=mu, scale=sigma, observed=t)
```

### Inference

Tensorflow will automatically run inference on a GPU if available. With the current version of PyMC4, inference on a GPU is quite slow compared to a multi-core CPU (need to investigate that in more detail). To enforce inference on a CPU set environment variable `CUDA_VISIBLE_DEVICES` to an empty value. There is no progress bar visible yet during sampling but the following shouldn't take longer than a 1 minute.


```python
trace = pm.sample(model(expand(x), t), num_chains=3, burn_in=100, num_samples=1000)
```


```python
az.plot_trace(trace);
```


![png](/img/2020-04-25/output_23_0.png)



```python
az.plot_posterior(trace, var_names="model/w_0");
az.plot_posterior(trace, var_names="model/w_r");
```


![png](/img/2020-04-25/output_24_0.png)



![png](/img/2020-04-25/output_24_1.png)


### Prediction

To obtain posterior predictive samples for a test set `x_test` we simply call the model generator function again with the expanded test set. This is a nice improvement over PyMC3 which required to setup a shared Theano variable for setting test set values. Target values are ignored during predictive sampling, only the shape of the target array `t` matters.


```python
draws_posterior = pm.sample_posterior_predictive(model(expand(x_test), t=np.zeros_like(x_test)), trace, inplace=False)
draws_posterior.posterior_predictive
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<title>Show/Hide data repr</title>
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<title>Show/Hide attributes</title>
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

.xr-wrap {
  min-width: 300px;
  max-width: 700px;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt, dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><div class='xr-wrap'><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-c38941c4-9742-4114-a3e5-3c1cd16a59c6' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-c38941c4-9742-4114-a3e5-3c1cd16a59c6' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 3</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>model/t_obs_dim_0</span>: 100</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-1a205c05-77e1-4350-97a1-bd32522ecf16' class='xr-section-summary-in' type='checkbox'  checked><label for='section-1a205c05-77e1-4350-97a1-bd32522ecf16' class='xr-section-summary' >Coordinates: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2</div><input id='attrs-6fbcf313-3e62-4786-9671-e5e464ff8406' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-6fbcf313-3e62-4786-9671-e5e464ff8406' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-c0df3e38-ae3b-44e0-8e28-97b27bc16883' class='xr-var-data-in' type='checkbox'><label for='data-c0df3e38-ae3b-44e0-8e28-97b27bc16883' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([0, 1, 2])</pre></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-5a36de0a-53dd-46ea-9dff-114085398a38' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5a36de0a-53dd-46ea-9dff-114085398a38' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f1f93bc5-8924-486a-ae1e-f0253e8f9000' class='xr-var-data-in' type='checkbox'><label for='data-f1f93bc5-8924-486a-ae1e-f0253e8f9000' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([  0,   1,   2, ..., 997, 998, 999])</pre></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>model/t_obs_dim_0</span></div><div class='xr-var-dims'>(model/t_obs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 ... 94 95 96 97 98 99</div><input id='attrs-aeb6d2f8-ad75-4af9-b646-da48527bff00' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-aeb6d2f8-ad75-4af9-b646-da48527bff00' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3eed21f9-db8a-42d0-bee2-e8a05a17322c' class='xr-var-data-in' type='checkbox'><label for='data-3eed21f9-db8a-42d0-bee2-e8a05a17322c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
       54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
       72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
       90, 91, 92, 93, 94, 95, 96, 97, 98, 99])</pre></li></ul></div></li><li class='xr-section-item'><input id='section-ef6888d7-f8b3-4631-99bf-11a3076d35c3' class='xr-section-summary-in' type='checkbox'  checked><label for='section-ef6888d7-f8b3-4631-99bf-11a3076d35c3' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>model/t_obs</span></div><div class='xr-var-dims'>(chain, draw, model/t_obs_dim_0)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>0.62803966 ... -0.10609433</div><input id='attrs-5f35cb15-08b1-454e-aaeb-66b92898badb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-5f35cb15-08b1-454e-aaeb-66b92898badb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-1ba80739-e72e-405c-9109-1fdf91a36892' class='xr-var-data-in' type='checkbox'><label for='data-1ba80739-e72e-405c-9109-1fdf91a36892' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><pre class='xr-var-data'>array([[[ 6.2803966e-01,  3.0982676e-01,  1.3288246e+00, ...,
          2.0092756e-02, -4.6279129e-01, -3.5547027e-01],
        [ 1.2540956e+00,  1.3001926e+00,  4.7648013e-01, ...,
         -1.4047767e-01, -6.6063479e-02,  2.2666046e-01],
        [ 8.7482959e-01,  7.1901262e-01,  1.2609010e+00, ...,
          3.6891103e-01,  2.3930666e-01,  1.9714403e-01],
        ...,
        [ 6.5140450e-01,  9.2145377e-01,  2.7004269e-01, ...,
          6.3866097e-04,  3.6582848e-01,  4.0039763e-01],
        [ 4.4500881e-01,  2.6833433e-01,  5.4804039e-01, ...,
          7.7021873e-01,  2.5889888e-02,  6.1815977e-03],
        [ 7.7372921e-01,  7.9454470e-01,  6.1503142e-01, ...,
          1.0394448e-01, -4.7731856e-01, -6.0296464e-01]],

       [[ 1.0802209e-02,  5.3853476e-01,  4.2005211e-01, ...,
          3.4785268e-01,  5.5825341e-01,  3.6537340e-01],
        [ 1.0661882e+00,  6.1011136e-01,  1.2609197e+00, ...,
          2.7852780e-01,  6.0179305e-01,  8.8738966e-01],
        [ 7.4540353e-01,  1.2344036e+00,  1.2811742e+00, ...,
          7.6069474e-01,  5.6832170e-01,  1.1162102e+00],
        ...,
        [-9.9507570e-03,  3.3239186e-01,  4.7235852e-01, ...,
         -3.1367943e-01, -1.1621615e-01,  8.4965013e-02],
        [ 6.0881937e-01,  5.4845160e-01,  3.3895850e-01, ...,
         -1.8985049e-01,  5.1551007e-02, -2.9580078e-01],
        [ 4.2067486e-01,  1.0590549e+00,  8.4452099e-01, ...,
         -4.9186319e-01, -1.4563501e-01, -1.7367038e-01]],

       [[-2.7087212e-01,  2.2036786e-01,  2.1426165e-01, ...,
          1.7241767e-01,  3.1225359e-01,  6.8863893e-01],
        [ 7.2146583e-01,  6.2246352e-01,  6.6259170e-01, ...,
          3.3384523e-01, -2.5926927e-01, -3.3233041e-01],
        [ 4.1656739e-01,  7.5270784e-01,  5.5890125e-01, ...,
          1.1958110e-01, -1.2425715e-01, -2.5198370e-01],
        ...,
        [ 5.0131804e-01, -7.0139110e-02,  1.1371263e+00, ...,
          1.8864080e-01, -1.3917631e-01,  4.4616908e-01],
        [ 1.9719602e-01,  6.5913051e-01,  8.3163023e-01, ...,
          5.0224549e-01,  4.1368300e-01,  5.7770413e-01],
        [ 5.0372481e-01,  3.2341504e-01,  6.1320949e-01, ...,
         -6.8751842e-02, -7.4040598e-01, -1.0609433e-01]]], dtype=float32)</pre></li></ul></div></li><li class='xr-section-item'><input id='section-26cc2edd-e860-436f-a4b2-2527df12879c' class='xr-section-summary-in' type='checkbox'  checked><label for='section-26cc2edd-e860-436f-a4b2-2527df12879c' class='xr-section-summary' >Attributes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2020-04-22T05:50:53.606431</dd><dt><span>arviz_version :</span></dt><dd>0.7.0</dd></dl></div></li></ul></div></div>



The predictive mean and standard deviation is obtained by averaging over chains (axis `0`) and predictive samples (axis `1`) for each of the 100 data points in `x_test` (axis `2`).


```python
predictive_samples = draws_posterior.posterior_predictive.data_vars['model/t_obs'].values

m = np.mean(predictive_samples, axis=(0, 1))
s = np.std(predictive_samples, axis=(0, 1))
```

These statistics can be used to plot model predictions and their uncertainties (together with the ground truth and the noisy training dataset).


```python
plt.fill_between(x_test, m + s, m - s, alpha = 0.5, label='Predictive std. dev.')
plt.plot(x_test, m, label='Predictive mean');

plot_data(x, t)
plot_truth(x_test, y_true, label=None)

plt.legend();
```


![png](/img/2020-04-25/output_30_0.png)


Try running the example again with Gaussian expansion i.e. setting `expand = expand_gaussian` and see how it compares to polynomial expansion. Also try running with a different number of basis functions by overriding the default arguments of `expand_polynomial` and `expand_gaussian`. You can find more PyMC4 examples in the [notebooks](https://github.com/pymc-devs/pymc4/tree/master/notebooks) diretory of the PyMC4 project.
