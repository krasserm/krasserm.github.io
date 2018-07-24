---
title: From expectation maximization to stochastic variational inference
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

## Introduction

Given a probabilistic model $p(\mathbf{x};\mathbf\theta)$ and some observations $\mathbf{x}$, we often want to estimate optimal parameter values $\mathbf{\hat{\theta}}$ that maximize the data likelihood. This can be done via [maximum likelihood](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) (ML) estimation or [maximum a posteriori](https://de.wikipedia.org/wiki/Maximum_a_posteriori) (MAP) estimation if point estimates of $\mathbf\theta$ are sufficient:

$$
\mathbf{\hat{\theta}} = \underset{\mathbf\theta}{\mathrm{argmax}}\ p(\mathbf{x};\mathbf\theta)\tag{1}
$$

In many cases, direct computation and optimization of the likelihood function $p(\mathbf{x};\mathbf\theta)$ is complex or impossible. One option to ease computation is the introduction of [latent variables](https://en.wikipedia.org/wiki/Latent_variable) $\mathbf{t}$ so that we have a complete data likelihood $p(\mathbf{x},\mathbf{t};\mathbf\theta)$ which can be decomposed into a conditional likelihood $p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)$ and a prior $p(\mathbf{t})$.

$$
p(\mathbf{x},\mathbf{t};\mathbf\theta) = p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)p(\mathbf{t})\tag{2}
$$

Latent variables are not observed directly but assumed to cause observations $\mathbf{x}$. Their choice is problem-dependent. To obtain the the marginal likelihood $p(\mathbf{x};\mathbf\theta)$, we have to integrate i.e. marginalize out the latent variables.

$$
p(\mathbf{x};\mathbf\theta) = 
\int p(\mathbf{x},\mathbf{t};\mathbf\theta)d\mathbf{t} = 
\int p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)p(\mathbf{t})d\mathbf{t}
\tag{3}
$$

Usually, we choose a latent variable model such that parameter estimation for the conditional likelihood $p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)$ is easier than for the marginal likelihood $p(\mathbf{x};\mathbf\theta)$. For example, the conditional likelihood of a Gaussian [mixture model](https://en.wikipedia.org/wiki/Mixture_model) (GMM) is a single Gaussian for which parameter estimation is easier than for the marginal likelihood which is a mixture of Gaussians. The latent variable $\mathbf{t}$ in a GMM determines the assignment to mixture components and follows a categorical distribution. If we can solve the integral in Eq. 3 we can also compute the posterior distribution of the latent variables by using [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem):

$$
p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta) = 
\frac{p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)p(\mathbf{t})}
    {p(\mathbf{x};\mathbf\theta)}
\tag{4}
$$

With the posterior, inference for the latent variables becomes possible. Note that in this article the term *estimation* is used to refer to (point) estimation of parameters via ML or MAP and *inference* to refer to Bayesian inference of random variables by computing the posterior. 

A major challenge in Bayesian inference is that the integral in Eq. 3 is often impossible or very difficult to compute in closed form. Therefore, many techniques exist to approximate the posterior in Eq. 4. They can be classified into numerical approximations ([Monte Carlo techniques](https://en.wikipedia.org/wiki/Monte_Carlo_method)) and deterministic approximations. This article is about deterministic approximations only, and their stochastic variants.

## Expectation maximization (EM)

Basis for many Bayesian inference methods is the [expectation-maximization](https://en.wikipedia.org/wiki/Expectation-maximization_algorithm) (EM) algorithm. It is an iterative algorithm for estimating the parameters of latent variable models, often with closed-form updates at each step. We start with a rather general view of the EM algorithm that also serves as a basis for discussing variational inference methods later. It is straightforward to show<sup>[2]</sup> that the marginal log likelihood can be written as 

$$
\log p(\mathbf{x};\mathbf\theta) = 
  \mathcal{L}(q, \mathbf\theta) + 
  \mathrm{KL}(q \mid\mid p)
\tag{5}
$$

with 

$$
\mathcal{L}(q, \mathbf\theta) = \int q(\mathbf{t}) \log 
  \frac{p(\mathbf{x},\mathbf{t};\mathbf\theta)}
       {q(\mathbf{t})} d\mathbf{t}
\tag{6}
$$

and 

$$
\mathrm{KL}(q \mid\mid p) = - \int q(\mathbf{t}) \log 
  \frac{p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)}
       {q(\mathbf{t})} d\mathbf{t}
\tag{7}
$$

where $q(\mathbf{t})$ is any probability density function. $\mathrm{KL}(q \mid\mid p)$ is the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between $q(\mathbf{t})$ and $p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)$ that measures how much $q$ diverges from $p$. The Kullback-Leibler divergence is zero for identical distributions and greater than zero otherwise. Thus, $\mathcal{L}(q, \mathbf\theta)$ is a lower bound of the log likelihood. It is equal to the log likelihood if $q(\mathbf{t}) = p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)$. In the E-step of the EM algorithm, $q(\mathbf{t})$ is therefore set to $p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)$ using the parameter values of the previous iteration $l-1$. 

$$
q^{l}(\mathbf{t}) = p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta^{l-1})\tag{8}
$$

Note that this requires that $p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)$ is known, like in the GMM case where the posterior is a categorical distribution, as mentioned above. In the M-step, $\mathcal{L}(q, \mathbf\theta)$ is optimized w.r.t. $\mathbf\theta$ using $q(\mathbf{t})$ from the E-step:

$$
\mathbf\theta^{l} = \underset{\mathbf\theta}{\mathrm{argmax}}\ \mathcal{L}(q^{l}, \mathbf\theta)\tag{9}
$$

In general, this is much simpler than optimizing $p(\mathbf{x};\mathbf\theta)$ directly. E and M steps are repeated until convergence. However, the requirement that the posterior $p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)$ must be known is rather restrictive and there are many cases where the posterior is intractable. In these cases, further approximations must be made.

## Variational EM

If the posterior is unknown, we have to assume specific forms of $q(\mathbf{t})$ and maximize the lower bound $\mathcal{L}(q, \mathbf\theta)$ w.r.t. these functions. The area of mathematics related to these optimization problems is called [calculus of variations](https://en.wikipedia.org/wiki/Calculus_of_variations)<sup>[3]</sup>, hence to name *variational EM*, or *variationial inference* in general. A widely used approximation for the unknown posterior is the [mean-field approximation](https://en.wikipedia.org/wiki/Mean_field_theory)<sup>[2][3]</sup> which factorizes $q(\mathbf{t})$ into $M$ partitions:

$$
q(\mathbf{t}) = \prod_{i=1}^{M} q_i(\mathbf{t}_i)\tag{10}
$$

For example, if $\mathbf{t}$ is 10-dimensional, we can factorize $q(\mathbf{t})$ into a product of 10 $$q_i(\mathbf{t}_i)$$, one for each dimension, assuming independence between dimensions. In the mean-field approximation, the E-step of the EM algorithm is modified to find optimal $$q_i$$ iteratively. Without showing detailed update formulas, an update for $q_i$ can be computed from parameters $\mathbf\theta$ of the previous iteration $l-1$ and all $$q_{j \ne i}$$: 

$$
q_i^l(\mathbf{t}_i) = f(p(\mathbf{x},\mathbf{t};\mathbf\theta^{l-1}), q_{j \ne i})\tag{11}
$$

This is repeated for all $q_i$ until convergence. The E-step of the variational EM algorithm is therefore

$$
q^l(\mathbf{t}) = \prod_{i=1}^{M} q_i^l(\mathbf{t}_i)\tag{12}
$$

and the M-step uses the posterior approximation $q^l(\mathbf{t})$ from the E-step to estimate parameters $\mathbf\theta^l$:

$$
\mathbf\theta^{l} = \underset{\mathbf\theta}{\mathrm{argmax}}\ \mathcal{L}(q^{l}, \mathbf\theta)\tag{13}
$$

The mean field approach allows inference for many interesting latent variable models but it requires analytical solutions w.r.t. the approximate posterior which is not always possible. Especially when used in context of deep learning where the approximate posterior $q(\mathbf{t})$ and the conditional likelihood $p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)$ are neural networks with at least one non-linear hidden layer, the mean field approach is not applicable any more<sup>[4]</sup>. Further approximations are required.

## Stochastic variational inference

Let's assume we have a latent variable model with one latent variable $$\mathbf{t}^{(i)}$$ for each observation $$\mathbf{x}^{(i)}$$. Observations $$\mathbf{x}^{(i)}$$ come from an i.i.d. dataset. To make the following more concrete let's say that $$\mathbf{x}^{(i)}$$ are images and $$\mathbf{t}^{(i)}$$ are $D$-dimensional latent vectors that cause the generation of $$\mathbf{x}^{(i)}$$ under the generative model $$p(\mathbf{x},\mathbf{t};\mathbf\theta) = p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)p(\mathbf{t})$$.

Our goal is to find optimal parameter values for the marginal likelihood $p(\mathbf{x};\mathbf\theta)$ by maximizing its variational lower bound. Here, we neither know the true posterior $p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)$ nor can we apply the mean field approximation<sup>[4]</sup>, so we have to make further approximations. We start by assuming that $q(\mathbf{t})$ is a factorized Gaussian i.e. a Gaussian with a diagonal covariance matrix and that we have a separate distribution $$q^{(i)}$$ for each latent variable $$\mathbf{t}^{(i)}$$: 

$$
q^{(i)}(\mathbf{t}^{(i)}) = 
  \mathcal{N}(\mathbf{t}^{(i)} \lvert \mathbf{m}^{(i)},\mathrm{diag}(\mathbf{s}^{2(i)}))
\tag{14}
$$

The problem here is that we have to estimate too many parameters. For example, if the latent space is 50-dimensional we have to estimate about 100 parameters per training object! This is not what we want. Another option is that all $q^{(i)}$ share their parameters $\mathbf{m}$ and $\mathbf{s}^2$ i.e. all $q^{(i)}$ are identical. This would keep the number of parameters constant but would be too restrictive though. If we want to support different $q^{(i)}$ for different $\mathbf{t}^{(i)}$ but with a limited number of parameters we should consider using parameters for $q$ that are functions of $\mathbf{x}^{(i)}$. These functions are themselves parametric functions that share a set of parameters $\mathbf\phi$:

$$
q^{(i)}(\mathbf{t}^{(i)}) = \mathcal{N}(\mathbf{t}^{(i)} \lvert 
  m(\mathbf{x}^{(i)},\mathbf\phi), \mathrm{diag}(s^2(\mathbf{x}^{(i)},\mathbf\phi)))
\tag{15}
$$

So we finally have a variational distribution $q(\mathbf{t} \lvert \mathbf{x};\mathbf\phi)$ with a fixed number of parameters $\mathbf\phi$ as approximation for the true but unknown posterior $p(\mathbf{t} \lvert \mathbf{x};\mathbf\theta)$. To implement the (complex) functions $m$ and $s$ that map from an input image to the mean and the variance of that distribution we can use a [convolutional neural network](https://de.wikipedia.org/wiki/Convolutional_Neural_Network) (CNN) that is parameterized by $\mathbf\phi$. 

Similarly, for implementing $p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)$ we can use another neural network, parameterized by $\mathbf\theta$, that maps a latent vector $\mathbf{t}$ to the sufficient statistics of that probability distribution. Since $\mathbf{t}$ is often a lower-dimensional embedding or code of image $\mathbf{x}$, $q(\mathbf{t} \lvert \mathbf{x};\mathbf\phi)$ is referred to as *probabilistic encoder* and $p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)$ as *probabilistic decoder*. 

![](/img/2018-04-03/auto-encoder-1.png)

### Variational auto-encoder

Both, encoder and decoder, can be combined to a *variational auto-encoder*<sup>[4]</sup> that is trained with the variational lower bound $\mathcal{L}$ as optimization objective using standard stochastic gradient ascent methods. For our model, the variational lower bound for a single training object $\mathbf{x}^{(i)}$ can also be formulated as:

$$
\mathcal{L}(\mathbf\theta, \mathbf\phi, \mathbf{x}^{(i)}) =
  \mathbb{E}_{q(\mathbf{t} \lvert \mathbf{x}^{(i)};\mathbf\phi)} \left[\log p(\mathbf{x}^{(i)} \lvert \mathbf{t};\mathbf\theta)\right]
  - \mathrm{KL}(q(\mathbf{t} \lvert \mathbf{x}^{(i)};\mathbf\phi) \mid\mid p(\mathbf{t}))
\tag{16}
$$

The first term is the expected negative *reconstruction error* of an image $\mathbf{x}^{(i)}$. This term is maximized when the reconstructed image is as close as possible to the original image. It is computed by first feeding an input image $\mathbf{x}^{(i)}$ through the encoder to compute the mean and the variance of the variational distribution $q(\mathbf{t} \lvert \mathbf{x};\mathbf\phi)$. To compute an approximate value of the expected negative reconstruction error, we sample from the variational distribution. Since this is a Gaussian distribution, sampling is very efficient. To compute $p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)$ we feed the samples through the decoder. A single sample per training object is usually sufficient<sup>[4]</sup> if the mini-batch size during training is large enough e.g. > 100.

![](/img/2018-04-03/auto-encoder-2.png)

The second term in Eq. 16, the negative KL divergence, is maximized when the approximate posterior $q(\mathbf{t} \lvert \mathbf{x};\mathbf\phi)$ is equal to the prior $p(\mathbf{t})$. The prior is usually chosen to be the standard normal distribution $\mathcal{N}(\mathbf{0},\mathbf{I})$. This term therefore acts as a regularization term to avoid that the variance of $q(\mathbf{t} \lvert \mathbf{x};\mathbf\phi)$ becomes zero, otherwise, $q(\mathbf{t} \lvert \mathbf{x};\mathbf\phi)$ would degenerate to a delta function and the variational auto-encoder to a usual auto-encoder. Regularizing $q(\mathbf{t} \lvert \mathbf{x};\mathbf\phi)$ to have non-zero variance makes the decoder more robust against small changes in $\mathbf{t}$ and the latent space a continuous space of codes that can be decoded to realistic images. 

### Gradient of the variational lower bound

To be able to use the variational lower bound as optimization objective or loss function in tools like [Tensorflow](https://www.tensorflow.org/), we have to make sure that it is differentiable. This is easy to achieve for the regularization term which can be integrated analytically in the Gaussian case

$$
- \mathrm{KL}(q(\mathbf{t} \lvert \mathbf{x}^{(i)};\mathbf\phi) \mid\mid p(\mathbf{t})) =
  \frac{1}{2} \sum_{j=1}^{D}(1 + \log((s_j)^2) - (m_j)^2 - (s_j)^2)
\tag{17}
$$

where $m_j$ and $s_j$ denote the $j$-th elements of the vectors computed with functions $m$ and $s$ (see Eq. 15). $D$ is the dimensionality of these vectors. The computation of the expected negative reconstruction error, on the other hand, involves sampling from $q(\mathbf{t} \lvert \mathbf{x};\mathbf\phi)$ which is not differentiable. However, a simple reparameterization trick allows to express the random variable $\mathbf{t}$ as deterministic variable $\mathbf{t} = g(\mathbf{m}, \mathbf{s}, \mathbf\epsilon) = \mathbf{m} + \mathbf{s} \mathbf\epsilon$ plus a random (noise) variable $\mathbf\epsilon \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ that doesn't depend on any parameters to be optimized. With this trick we can easily compute the gradient of function $g$ and can ignore $\mathbf\epsilon$ i.e. the sampling procedure during back-propagation.

![](/img/2018-04-03/auto-encoder-3.png)

We haven't defined the functional form of the probabilistic decoder $p(\mathbf{x} \lvert \mathbf{t};\mathbf\theta)$ yet. If we train the variational auto-encoder with grey-scale [MNIST images](https://en.wikipedia.org/wiki/MNIST_database), for example, it makes sense to use a multivariate Bernoulli distribution. In this case, the output of the decoder network is the single parameter of this distribution. It defines for each pixel the probability of being white. These probabilities are then simply mapped to values from 0-255 to generate grey-scale images. In the output layer of the decoder network there is one node per pixel with a sigmoid activation function. Hence, we compute the binary cross-entropy between the input image and the decoder output to estimate the expected reconstruction error. 

You can find a complete implementation example of a variational auto-encoder [in this notebook](http://nbviewer.jupyter.org/github/krasserm/bayesian-machine-learning/blob/master/variational_autoencoder.ipynb). It is part of the [bayesian-machine-learning](https://github.com/krasserm/bayesian-machine-learning) repository.

Stochastic variational inference algorithms implemented as variational auto-encoders scale to very large datasets as they can be trained based on mini-batches. Furthermore, they can also be used for data other than image data. For example, Gómez-Bombarelli et. al.<sup>[5]</sup> use a sequential representation of chemical compounds together with an  RNN-based auto-encoder to infer a continuous latent space of chemical compounds that can be used e.g. for generating new chemical compounds with properties that are desirable for drug discovery. I'll cover that in another blog post.

## References

\[1\] Dimitris G. Tzikass, Aristidis et. al. [The Variational Approximation for Bayesian Inference](http://www.cs.uoi.gr/~arly/papers/SPM08.pdf).  
\[2\] Kevin P. Murphy. [Machine Learning, A Probabilistic Perspective](https://mitpress.mit.edu/books/machine-learning-0), Chapters 11 and 21.  
\[3\] Christopher M. Bishop. [Pattern Recognition and Machine Learning](http://www.springer.com/de/book/9780387310732), Chapters 9 and 10.  
\[4\] Diederik P Kingma, Max Welling [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114).  
\[5\] Gómez-Bombarelli et. al. [Automatic chemical design using a data-driven continuous representation of molecules](https://arxiv.org/abs/1610.02415).  

