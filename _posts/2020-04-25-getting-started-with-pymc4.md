---
title: Getting started with PyMC4
subtitle: Bayesian neural networks
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

*You can find the notebook for this article [here](https://github.com/krasserm/bayesian-machine-learning/blob/master/bayesian_neural_networks_pymc4.ipynb). It is part of the [bayesian-machine-learning](https://github.com/krasserm/bayesian-machine-learning) repo on Github.*

This article demonstrates how to implement a simple Bayesian neural network for regression with an early [PyMC4 development snapshot](https://github.com/pymc-devs/pymc4/tree/1c5e23825271fc2ff0c701b9224573212f56a534) (from Jul 29, 2020). It can be installed with 

```bash
pip install git+https://github.com/pymc-devs/pymc4@1c5e23825271fc2ff0c701b9224573212f56a534
```

I'll update this article from time to time to cover new features or to fix breaking API changes. The latest update (Aug. 19, 2020) includes the recently added support for variational inference (VI). The following sections assume that you have a basic familiarity with [PyMC3](https://docs.pymc.io/). If this is not the case I recommend reading [Getting started with PyMC3](https://docs.pymc.io/notebooks/getting_started.html) first.


```python
import logging
import pymc4 as pm
import numpy as np
import arviz as az

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

%matplotlib inline

print(pm.__version__)
print(tf.__version__)
print(tfp.__version__)

# Mute Tensorflow warnings ...
logging.getLogger('tensorflow').setLevel(logging.ERROR)
```

    4.0a2
    2.4.0-dev20200818
    0.12.0-dev20200818


## Introduction to PyMC4

PyMC4 uses [Tensorflow Probability](https://www.tensorflow.org/probability) (TFP) as backend and PyMC4 random variables are wrappers around TFP distributions. Models must be defined as [generator](https://docs.python.org/3/glossary.html#term-generator) functions, using a `yield` keyword for each random variable. PyMC4 uses [coroutines](https://www.python.org/dev/peps/pep-0342/) to interact with the generator to get access to these variables. Depending on the context, PyMC4 may sample values from random variables, compute log probabilities of observed values, ... and so on. Details are covered in the [PyMC4 design guide](https://github.com/pymc-devs/pymc4/blob/master/notebooks/pymc4_design_guide.ipynb). Model generator functions must be decorated with ` @pm.model` as shown in the following trivial example:


```python
@pm.model
def model(x):
    # prior for the mean of a normal distribution
    loc = yield pm.Normal('loc', loc=0, scale=10)
    
    # likelihood of observed data
    obs = yield pm.Normal('obs', loc=loc, scale=1, observed=x)
```

This models normally distributed data centered at a location `loc` to be inferred. Inference can be started with `pm.sample()` which uses the [No-U-Turn Sampler](https://jmlr.org/papers/volume15/hoffman14a/hoffman14a.pdf) (NUTS). Samplers other than NUTS are currently not implemented in PyMC4.


```python
# 30 data points normally distributed around 3
x = np.random.randn(30) + 3

# Inference
trace = pm.sample(model(x))
trace
```

```
arviz.InferenceData
- posterior
- sample_stats
- observed_data
```

The returned `trace` object is an ArviZ [`InferenceData`](https://arviz-devs.github.io/arviz/notebooks/XarrayforArviZ.html) object. It contains posterior samples, observed data and sampler statistics. The posterior distribution over `loc` can be displayed with:


```python
az.plot_posterior(trace, var_names=['model/loc']);
```


![png](/img/2020-04-25/output_7_0.png)


A recent addition to PyMC4 is variational inference and supported methods currently are `advi` and `fullrank_advi`. After fitting the model, posterior samples can be obtained from the resulting `approximation` object (representing a mean-field approximation in this case).


```python
fit = pm.fit(model(x), num_steps=10000, method='advi')
trace = fit.approximation.sample(1000)
```


```python
az.plot_posterior(trace, var_names=['model/loc']);
```


![png](/img/2020-04-25/output_10_0.png)


The history of the variational lower bound  (= negative loss) during training can be displayed with


```python
plt.plot(-fit.losses)
plt.ylabel('Variational lower bound')
plt.xlabel('Step');
```


![png](/img/2020-04-25/output_12_0.png)


which confirms a good convergence after about 10,000 steps. Models can also be composed through nesting and used like other PyMC4 random variables.


```python
@pm.model
def prior(name, loc=0, scale=10):
    loc = yield pm.Normal(name, loc=loc, scale=scale)
    return loc

@pm.model
def model(x):
    loc = yield prior('loc')
    obs = yield pm.Normal('obs', loc=loc, scale=1, observed=x)
    
trace = pm.sample(model(x))
az.plot_posterior(trace, var_names=['model/prior/loc']);    
```


![png](/img/2020-04-25/output_14_0.png)


A more elaborate example is shown below where a neural network is composed of several layers. 

## Example dataset

The dataset used in the following example contains `N` noisy samples from a sinusoidal function `f` in two distinct regions (`x1` and `x2`).


```python
def f(x, noise):
    """Generates noisy samples from a sinusoidal function at x."""
    return np.sin(2 * np.pi * x) + np.random.randn(*x.shape) * noise

N = 40
noise = 0.1

x1 = np.linspace(-0.6, -0.15, N // 2, dtype=np.float32)
x2 = np.linspace(0.15, 0.6, N // 2, dtype=np.float32)

x = np.concatenate([x1, x2]).reshape(-1, 1)
y = f(x, noise=noise)

x_test = np.linspace(-1.5, 1.5, 200, dtype=np.float32).reshape(-1, 1)
f_test = f(x_test, noise=0.0)

plt.scatter(x, y, marker='o', c='k', label='Samples')
plt.plot(x_test, f_test, 'k--', label='f')
plt.legend();
```


![png](/img/2020-04-25/output_18_0.png)


## Bayesian neural network

### Model definition

To model the non-linear relationship between `x` and `y` in the dataset we use a ReLU neural network with two hidden layers, 5 neurons each. The weights of the neural network are random variables instead of deterministic variables. This is what makes a neural network a Bayesian neural network. Here, we assume that the weights are independent random variables. 

The neural network defines a prior over the mean `loc` of the data likelihood `obs` which is represented by a normal distribution. For simplicity, the aleatoric uncertainty (`noise`) in the data is assumed to be known. Thanks to PyMC4's model composition support, priors can be defined layer-wise using the `layer` generator function and composed to a neural network as shown in function `model`. During inference, a posterior distribution over the neural network weights is obtained. 


```python
@pm.model
def layer(name, x, n_in, n_out, prior_scale, activation=tf.identity):
    w = yield pm.Normal(name=f'{name}_w', loc=0, scale=prior_scale, batch_stack=(n_in, n_out))
    b = yield pm.Normal(name=f'{name}_b', loc=0, scale=prior_scale, batch_stack=(1, n_out))
    
    return activation(tf.tensordot(x, w, axes=[1, 0]) + b)

@pm.model
def model(x, y, prior_scale=1.0):    
    o1 = yield layer('l1', x, n_in=1, n_out=5, prior_scale=prior_scale, activation=tf.nn.relu)
    o2 = yield layer('l2', o1, n_in=5, n_out=5, prior_scale=prior_scale, activation=tf.nn.relu)
    o3 = yield layer('l3', o2, n_in=5, n_out=1, prior_scale=prior_scale)
    
    yield pm.Normal(name='obs', loc=o3, scale=noise, observed=y)
```

The `batch_stack` parameter of random variable constructors is used to define the shape of the random variable.

### Inference

Tensorflow will automatically run inference on a GPU if available. With the current version of PyMC4, MCMC inference using NUTS on a GPU is quite slow compared to a multi-core CPU (need to investigate that in more detail). To enforce inference on a CPU set environment variable `CUDA_VISIBLE_DEVICES` to an empty value. There is no progress bar visible yet during sampling but the following shouldn't take longer than a few minutes on a modern multi-core CPU.


```python
# MCMC inference with NUTS
trace = pm.sample(model(x, y, prior_scale=3), burn_in=100, num_samples=1000)
```

Variational inference is significantly faster but the results are less convincing than the MCMC results. I need to investigate that further to see if I'm doing something wrong or if this is an issue with the current PyMC4 development snapshot. We'll therefore use the MCMC results in the following section. If you want to see the VI results, run the following cell instead of the previous one.


```python
# Variational inference with full rank ADVI
fit = pm.fit(model(x, y, prior_scale=0.5), num_steps=150000, method='fullrank_advi')

# Draw samples from the resulting mean-field approximation
trace = fit.approximation.sample(1000)
```

The full `trace` can be visualized with `az.plot_trace(trace)`. Here, we only display the posterior over the last layer weights (without bias).


```python
az.plot_posterior(trace, var_names="model/layer/l3_w");
```


![png](/img/2020-04-25/output_26_0.png)


### Prediction

To obtain posterior predictive samples for a test set `x_test` we simply call the `model` generator function again with the test set as argument. This is a nice improvement over PyMC3 which required to setup a shared Theano variable for setting test set values. Target values are ignored during predictive sampling, only the shape of the target array `y` matters, hence we set it to an array of zeros with the same shape as `x_test`.


```python
draws_posterior = pm.sample_posterior_predictive(model(x=x_test, y=np.zeros_like(x_test)), trace, inplace=False)
draws_posterior.posterior_predictive
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
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

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
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
</style><pre class='xr-text-repr-fallback'>&lt;xarray.Dataset&gt;
Dimensions:          (chain: 10, draw: 1000, model/obs_dim_0: 200, model/obs_dim_1: 1)
Coordinates:
  * chain            (chain) int64 0 1 2 3 4 5 6 7 8 9
  * draw             (draw) int64 0 1 2 3 4 5 6 ... 993 994 995 996 997 998 999
  * model/obs_dim_0  (model/obs_dim_0) int64 0 1 2 3 4 5 ... 195 196 197 198 199
  * model/obs_dim_1  (model/obs_dim_1) int64 0
Data variables:
    model/obs        (chain, draw, model/obs_dim_0, model/obs_dim_1) float32 ...
Attributes:
    created_at:     2020-08-19T12:12:02.008383
    arviz_version:  0.9.0</pre><div class='xr-wrap' hidden><div class='xr-header'><div class='xr-obj-type'>xarray.Dataset</div></div><ul class='xr-sections'><li class='xr-section-item'><input id='section-7c46b00d-646f-4410-a108-48f232b7be69' class='xr-section-summary-in' type='checkbox' disabled ><label for='section-7c46b00d-646f-4410-a108-48f232b7be69' class='xr-section-summary'  title='Expand/collapse section'>Dimensions:</label><div class='xr-section-inline-details'><ul class='xr-dim-list'><li><span class='xr-has-index'>chain</span>: 10</li><li><span class='xr-has-index'>draw</span>: 1000</li><li><span class='xr-has-index'>model/obs_dim_0</span>: 200</li><li><span class='xr-has-index'>model/obs_dim_1</span>: 1</li></ul></div><div class='xr-section-details'></div></li><li class='xr-section-item'><input id='section-37f56e16-8169-4329-bd39-033864999c34' class='xr-section-summary-in' type='checkbox'  checked><label for='section-37f56e16-8169-4329-bd39-033864999c34' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>chain</span></div><div class='xr-var-dims'>(chain)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 6 7 8 9</div><input id='attrs-0eeccbe3-e46d-4840-a0f6-49500acb6a1f' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-0eeccbe3-e46d-4840-a0f6-49500acb6a1f' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-01b13910-659b-4a75-a610-37543426aaa9' class='xr-var-data-in' type='checkbox'><label for='data-01b13910-659b-4a75-a610-37543426aaa9' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>draw</span></div><div class='xr-var-dims'>(draw)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 995 996 997 998 999</div><input id='attrs-8da08f97-5716-4582-baf1-6ebd580ec713' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-8da08f97-5716-4582-baf1-6ebd580ec713' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5bf128e0-710d-49f1-9c30-550311cd180d' class='xr-var-data-in' type='checkbox'><label for='data-5bf128e0-710d-49f1-9c30-550311cd180d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2, ..., 997, 998, 999])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>model/obs_dim_0</span></div><div class='xr-var-dims'>(model/obs_dim_0)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0 1 2 3 4 5 ... 195 196 197 198 199</div><input id='attrs-7aa54c95-4fbc-46de-bcfa-0d9c9e543201' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-7aa54c95-4fbc-46de-bcfa-0d9c9e543201' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-3aee3527-352b-43e7-a92e-d2d5e4c9244d' class='xr-var-data-in' type='checkbox'><label for='data-3aee3527-352b-43e7-a92e-d2d5e4c9244d' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
        28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
        42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
        56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
        84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
        98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
       112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
       126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
       140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
       154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
       168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
       196, 197, 198, 199])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>model/obs_dim_1</span></div><div class='xr-var-dims'>(model/obs_dim_1)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-d2809ce2-13fe-4ce1-8b73-d1e6d65795d8' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-d2809ce2-13fe-4ce1-8b73-d1e6d65795d8' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-89ba0ed4-75da-441b-86f4-443a8fea6f33' class='xr-var-data-in' type='checkbox'><label for='data-89ba0ed4-75da-441b-86f4-443a8fea6f33' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([0])</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-252da4c5-4190-4df9-9eb6-0e8f7930fdf9' class='xr-section-summary-in' type='checkbox'  checked><label for='section-252da4c5-4190-4df9-9eb6-0e8f7930fdf9' class='xr-section-summary' >Data variables: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>model/obs</span></div><div class='xr-var-dims'>(chain, draw, model/obs_dim_0, model/obs_dim_1)</div><div class='xr-var-dtype'>float32</div><div class='xr-var-preview xr-preview'>26.57001 26.04135 ... -14.317561</div><input id='attrs-bc8ca1ce-a73b-45e5-8938-8eff592dd95b' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-bc8ca1ce-a73b-45e5-8938-8eff592dd95b' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-aa3156b7-f141-4a52-bce8-aab0fab7d89b' class='xr-var-data-in' type='checkbox'><label for='data-aa3156b7-f141-4a52-bce8-aab0fab7d89b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([[[[ 2.65700092e+01],
         [ 2.60413494e+01],
         [ 2.57281590e+01],
         ...,
         [-4.75241995e+00],
         [-4.78487253e+00],
         [-4.75552654e+00]],

        [[ 3.16064491e+01],
         [ 3.10987301e+01],
         [ 3.04416885e+01],
         ...,
         [-5.26556778e+00],
         [-5.20179415e+00],
         [-5.47270155e+00]],

        [[ 4.52943230e+01],
         [ 4.45980377e+01],
         [ 4.39933929e+01],
         ...,
...
         ...,
         [-5.59512615e+00],
         [-5.61213923e+00],
         [-5.73862267e+00]],

        [[ 2.81977844e+00],
         [ 2.81224990e+00],
         [ 3.02726460e+00],
         ...,
         [-5.47826385e+00],
         [-5.42433500e+00],
         [-5.45992947e+00]],

        [[ 3.89422917e+00],
         [ 3.80268312e+00],
         [ 3.86203289e+00],
         ...,
         [-1.39646444e+01],
         [-1.41936607e+01],
         [-1.43175611e+01]]]], dtype=float32)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-06fb4778-5e93-4a49-b46d-359b4e3e7aa4' class='xr-section-summary-in' type='checkbox'  checked><label for='section-06fb4778-5e93-4a49-b46d-359b4e3e7aa4' class='xr-section-summary' >Attributes: <span>(2)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>created_at :</span></dt><dd>2020-08-19T12:12:02.008383</dd><dt><span>arviz_version :</span></dt><dd>0.9.0</dd></dl></div></li></ul></div></div>



The predictive mean and standard deviation can be obtained by averaging over chains (axis `0`) and predictive samples (axis `1`) for each of the 200 data points in `x_test` (axis `2`).


```python
predictive_samples = draws_posterior.posterior_predictive.data_vars['model/obs'].values

m = np.mean(predictive_samples, axis=(0, 1)).flatten()
s = np.std(predictive_samples, axis=(0, 1)).flatten()
```

These statistics can be used to plot model predictions and their variances (together with function `f` and the noisy training data). One can clearly see a higher predictive variance (= higher uncertainty) in regions outside the training data.


```python
plt.plot(x_test, m, label='Expected value');
plt.fill_between(x_test.flatten(), m + 2 * s, m - 2 * s, alpha = 0.3, label='Uncertainty')

plt.scatter(x, y, marker='o', c='k')
plt.plot(x_test, f_test, 'k--')

plt.ylim(-1.5, 2.5)
plt.legend();
```


![png](/img/2020-04-25/output_32_0.png)


If you think something can be improved in this article (and I'm sure it can) or if I missed other important aspects of PyMC4 please let me know.
