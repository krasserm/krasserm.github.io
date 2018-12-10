---
title: Weight normalization implementation options for Keras and Tensorflow
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

At the time of writing this blog post no official implementation of [weight normalization](https://arxiv.org/abs/1602.07868) 
for Tensorflow and Keras is available. I therefore want to present a few options for implementing weight normalization in 
Tensorflow and Keras projects. The presented options for Keras only work in combination with the Tensorflow backend or
with the Keras version that is bundled with Tensorflow.

Tim Salimans, the author of the weight normalization paper [published code](https://github.com/openai/weightnorm/tree/master) 
for Keras 1.x and an older version of Tensorflow (about two years ago). If you're still using Keras 1.x or an older version 
of Tensorflow, I recommend using one of these implementations. The Keras-based implementation extends the Keras SGD and Adam 
optimizer with weight normalization functionality in a generic way so that different Keras layers (convolutional, dense, ...) 
can be trained with weight normalization. A procedure for data-based initialization is also provided. A Keras 2.x port of 
that code is available [here](https://github.com/krasserm/weightnorm/tree/master/keras_2). I tested 
the Adam-based weight normalization implementation in [another project](https://github.com/krasserm/wdsr) (an implementation
of [WDSR](https://arxiv.org/abs/1808.08718) for single image super-resolution) and get almost identical results as the 
PyTorch-based [reference implementation](https://github.com/JiahuiYu/wdsr_ntire2018) (PyTorch already contains an 
[official implementation](https://pytorch.org/docs/stable/nn.html#weight-norm) of weight normalization). 

The authors of the WDSR reference implementation also published a 
[Tensorflow port](https://github.com/ychfan/tf_estimator_barebone/blob/master/models/wdsr.py) that contains a 2D convolutional 
layer with support for weight normalization ([`Conv2DWeightNorm`](https://github.com/ychfan/tf_estimator_barebone/blob/master/common/layers.py)). 
This layer can also be used as Keras layer when using the Keras version bundled with Tensorflow 1.11 or higher and can be 
used in combination with any optimizer. Extending other layer types to support weight normalization should be easy using 
this template (but less elegant compared to a generic wrapper as described further below). The 
[`Conv2DWeightNorm`](https://github.com/ychfan/tf_estimator_barebone/blob/master/common/layers.py) code doesn't provide 
a procedure for data-based initialization though. For the special case of [WDSR](https://arxiv.org/abs/1808.08718), this 
is not needed as data-based initialization has a similar effect as batch-normalization which is known to harm single 
image super-resolution training but this cannot be generalized to other projects, of course. I also tested the 
`Conv2DWeightNorm` layer in the Keras version of WDSR (see section [Weight normalization](https://github.com/krasserm/wdsr#weight-normalization)) 
and convergence is similar to that of Adam-based weight normalization. The latter shows slightly better initial 
convergence but training with `Conv2DWeightNorm` in combination with a default Adam optimizer quickly catches up. The 
difference is probably due to different weight initializations. 

There's also a long-time open [pull request](https://github.com/tensorflow/tensorflow/pull/21276) 
for adding weight normalization to Tensorflow, also supporting the bundled Keras version, but review is still pending. 
It is a generic wrapper layer that works for several types of Tensorflow and Keras layers. Data-based initialization is 
also supported but only in eager mode. 

Several other implementations are evailable but most of them are slight variations of what I presented here. I hope 
this post saves you some time finding a weight normalization implementation for your project. If I missed something or 
an official implementation was published in the meantime, please let me know and I'll update this blog post.
