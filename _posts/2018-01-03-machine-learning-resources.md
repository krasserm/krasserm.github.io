---
title: Resources for getting started with ML and DL
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

The following is a collection of resources that I found useful when I started to learn machine learning and deep learning.
In this post I'm using the term *machine learning* to refer to classical machine learning and the term *deep learning* 
to refer to machine learning with deep neural networks. There are numerous other good resources out there which are not 
mentioned here. This doesn't mean I consider them as inferior, it's just that I haven't used them and therefore can't 
comment on.
  
## First steps

If you are completely new to machine learning I recommend starting with the outstanding Stanford 
[Machine Learning course](https://www.coursera.org/learn/machine-learning) by Andrew Ng. It is easy to follow and covers 
topics that every machine learning engineer really should know. The course uses Octave (an open source alternative to MATLAB) 
for programming. Algorithms are implemented from scratch in order to get a better understanding how they work. The course 
is also a good preparation for the [Deep Learning specialization](https://www.coursera.org/specializations/deep-learning) 
at Coursera.

After having taken the course I felt the need to learn Python and re-implement the exercises with [scikit-learn](http://scikit-learn.org). 
Scikit-learn is a Python machine learning library that provides optimized and easy-to-use implementations for all algorithms 
presented in the course. I published the results as [machine-learning-notebooks](https://github.com/krasserm/machine-learning-notebooks) 
project on GitHub.

If you are new to Python, the [Python tutorial](https://docs.python.org/3/tutorial/) is a great resource to start with. 
I also recommend to work at least through the [NumPy tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html), 
[SciPy tutorial](https://docs.scipy.org/doc/scipy/reference/tutorial/index.html), 
[Pandas tutorial](http://pandas.pydata.org/pandas-docs/stable/10min.html) and 
[Pyplot tutorial](http://matplotlib.org/users/pyplot_tutorial.html) before starting with the 
[scikit-learn tutorials](http://scikit-learn.org/stable/tutorial/index.html). After having worked through these tutorials 
you should be prepared for implementing the algorithms presented in the course with scikit-learn.

## Further courses

- [Deep Learning specialization](https://www.coursera.org/specializations/deep-learning). This specialization consists of 
five courses, tought by Andrew Ng, covering deep neural network basics, regularization and optimization and models for 
computer vision and sequences (text, speech, ...). If you enjoyed the quality and accessibility of Andrew's 
[Machine Learning course](https://www.coursera.org/learn/machine-learning) you will probably like this course too. It 
provides you with the skills needed to follow more advanced literature in that field, including research papers.  

  The initial programming exercises for the basics are in plain Python/numpy to get a better understanding how forward 
and backward propagation work. Models for computer vision are implemented with [Tensorflow](https://www.tensorflow.org/) 
and [Keras](https://keras.io/). Many examples cover recent research literature from 2014 or newer (ResNet, GoogLeNet, 
FaceNet, ... and many more). The last course on sequence models wasn't available yet at the time of writing this post. 
 
A good understanding of statistical inference basics is important to get more out of the machine learning, deep learning 
and statistics literature listed further below. If you need a refresher on statistical inference basics then the following 
courses might be helpful:

- [Inferential statistics](https://www.coursera.org/learn/inferential-statistics-intro). This course covers the basics of 
inference for numerical and categorical data, hypothesis testing and statistical tests such as ANOVA and Chi-squared. 
It follows the [frequentist approach](https://en.wikipedia.org/wiki/Frequentist_inference) to statistical inference and 
is part of the [Statistics with R specialization](https://www.coursera.org/specializations/statistics). The course content 
(except R basics) is also covered by the freely available book [OpenIntro Statistics](https://www.openintro.org/stat/textbook.php). 

- [Bayesian statistics](https://www.coursera.org/learn/bayesian-statistics). Many advanced machine learning and deep 
learning techniques are based on [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference). The course 
teaches the basics (Bayes' rule, conjugate models, Bayesian inference on discrete and continuous data, ...) and compares
them to the frequentist approach. Other basics such as Markov Chain Monte Carlo (MCMC) and hierarchical models are not 
covered though. A good companion to this course is the book 
[Doing Bayesian data analysis](https://www.amazon.com/Doing-Bayesian-Data-Analysis-Second/dp/0124058884) (see below). 
Before taking this course, familiarity with the frequentist approach is helpful. 

## Books

- [Machine learning - a probabilistic perspective](https://mitpress.mit.edu/books/machine-learning-0). A comprehensive 
book on classical machine learning techniques. Its focus is rather theoretical and the descriptions are math-heavy. All 
concepts are explained in an excellent way and therefore rather easy to follow even for machine learning beginners, given 
basic familiarity with multivariate calculus, probability and linear algebra. The book covers both the frequentist and 
Bayesian approach to inferring parameters of statistical models. Code examples are in MATLAB but there is also a 
[Python port](https://github.com/probml/pyprobml) available.  

- [Deep learning](http://www.deeplearningbook.org/). A comprehensive book on deep learning techniques. Part 1 covers 
machine learning basics. Part 2 covers deep neural network basics, convolutional neural networks (CNNs) and recurrent 
neural networks (RNNs). The content is comparable to that of the 
[Deep Learning specialization](https://www.coursera.org/specializations/deep-learning) but is presented in a more academic 
way. Part 3 covers more advanced topics such as auto-encoders, representation learning and deep generative models. This 
is not a book for a practitioners but one of the best deep learning overview books I've seen.    

- [Hands-on machine learning with scikit-learn and Tensorflow](http://shop.oreilly.com/product/0636920052289.do). If 
you've already taken a first machine learning and deep learning course, this book is for you. It is packed with useful 
code examples and guidelines for real-world machine learning projects. Part 1 focuses on the implementation of classical
machine learning models with scikit-learn. Part 2 focuses on deep learning with Tensorflow. In addition to CNNs and RNNs 
this part also has chapters on auto-encoders and reinforcement learning. Both, theory and code examples are presented 
in a clear and concise way.  
   
- [Deep learning with Python](https://www.manning.com/books/deep-learning-with-python). Another excellent deep learning 
book for practitioners with code examples using Keras. Keras is a deep learning framework with a higher-level API than 
Tensorflow that aims to enable rapid prototyping. In addition to a detailed coverage of CNNs and RNNs this book also has 
chapters on advanced deep learning best practices and generative deep learning. It is a good complement to part 2 of the 
previous books (from a tools perspective). If you are not sure which one is better to start with, I recommend this one as 
first steps are easier with Keras than with Tensorflow in my opinion.  
 
- [Introduction to statistical learning](http://www.springer.com/book/9781461471370). If 
[Machine learning - a probabilistic perspective](https://mitpress.mit.edu/books/machine-learning-0) is too math-heavy for 
you, this book is a good alternative. It covers statistical machine learning basics with a minimum of maths and approaches 
it from a frequentist inference perspective. It is also an excellent introduction to R. If you want to go deeper after 
having read this book, both in terms of math and number of approaches, I recommend 
[The elements of statistical learning](http://www.springer.com/book/9780387848570). Both books are also freely available 
as PDF ([ISL](http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf), 
[ESL](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12.pdf)).    

- [Doing Bayesian data analysis](https://www.amazon.com/Doing-Bayesian-Data-Analysis-Second/dp/0124058884). An excellent 
introduction to Bayesian statistics that prepares you well for reading more advanced literature in that field. It covers 
the Bayesian analogues to traditional statistical tests (t, ANOVA, Chi-squared, ...) and to multiple linear and logistic 
regression among many others. It requires only a basic knowledge of calculus. For me, the book was a helpful companion to 
the books [Machine learning - a probabilistic perspective](https://mitpress.mit.edu/books/machine-learning-0) and 
[Deep learning](http://www.deeplearningbook.org/).  Code examples are written in R using packages 
[JAGS](http://mcmc-jags.sourceforge.net/) and [Stan](http://mc-stan.org/users/interfaces/rstan) for MCMC sampling. There's 
also a [Python port](https://github.com/JWarmenhoven/DBDA-python) available using [PyMC3](http://docs.pymc.io/).  

- [Data Science from Scratch](http://shop.oreilly.com/product/0636920033400.do). This book is about data science in its 
most distilled form. Don't expect too much depth here but a great overview of data science topics such as probability and 
statistics, data preparation and machine learning basics. The book focuses on understanding fundamental data science tools 
by implementing them in plain Python from scratch. Well-known statistical and machine learning libraries are not used here 
but each chapter contains references to libraries you should actually use for your own projects and links for further reading.

There is also a large number of useful blogs and survey papers about machine learning which I'll leave for a separate 
post. I nevertheless hope you find this a useful guide for getting started with machine learning. 
