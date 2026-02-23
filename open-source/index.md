# Open Source Projects

## Gradion AI Projects

| Project | Description |
|---------|-------------|
| [freeact](https://github.com/gradion-ai/freeact) | Lightweight, general-purpose agent that acts via code actions rather than JSON tool calls. Progressively discovers and loads tools and skills as needed, preserving context for complex tasks. Creates new tools from successful code actions, evolving its own tool library over time. |
| [ipybox](https://github.com/gradion-ai/ipybox) | Python code execution sandbox with first-class support for programmatic MCP tool calling. Generates typed Python APIs from MCP server schemas and executes code in a stateful IPython kernel. Features programmatic tool call approval workflows and lightweight sandboxing via Anthropic's sandbox-runtime. |


## Personal Projects

| Project | Description |
|---------|-------------|
| [bayesian-machine-learning](https://github.com/krasserm/bayesian-machine-learning) | A collection of notebooks about Bayesian methods for machine learning, like [Bayesian regression](/2019/02/23/bayesian-linear-regression/), [Gaussian processes](/2018/03/19/gaussian-processes/), [Bayesian optimization](/2018/03/21/bayesian-optimization/), [variational inference in Bayesian neural networks](/2019/03/14/bayesian-neural-networks/), ..., etc. Each notebook covers a single topic and combines an introduction, mathematical basics and a simple implementation. |
| [perceiver-io](https://github.com/krasserm/perceiver-io) | The perceiver-io library is a modular implementation of [Perceiver](https://arxiv.org/abs/2103.03206), [Perceiver IO](https://arxiv.org/abs/2107.14795), and [Perceiver AR](https://arxiv.org/abs/2202.07765) in PyTorch, with a PyTorch Lightning integration for distributed training and a Hugging Face integration for inference. The project provides both ported [official models](https://github.com/krasserm/perceiver-io/blob/main/docs/pretrained-models.md#official-models) and [custom models](https://github.com/krasserm/perceiver-io/blob/main/docs/pretrained-models.md#training-checkpoints) used in [training examples](https://github.com/krasserm/perceiver-io/blob/main/docs/training-examples.md). |
| [super-resolution](https://github.com/krasserm/super-resolution) | This project provides a Tensorflow 2.x based implementation of three popular single image super-resolution models: [EDSR](https://arxiv.org/abs/1707.02921), [WDSR](https://arxiv.org/abs/1808.08718) and [SRGAN](https://arxiv.org/abs/1609.04802). Pre-trained weights, training and inference examples as well as a data loader for the DIV2K dataset are included. |
| [fairseq-image-captioning](https://github.com/krasserm/fairseq-image-captioning) | Implements an *Image Captioning Transformer* with the [fairseq](https://github.com/facebookresearch/fairseq) sequence modelling toolkit by combining ideas from [Self-critical Sequence Training for Image Captioning](https://arxiv.org/abs/1612.00563) and [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](https://arxiv.org/abs/1707.07998) with the [Transformer](https://arxiv.org/abs/1706.03762) architecture. |
| [face-recognition](https://github.com/krasserm/face-recognition) | Demonstrates how to build a face recognition system with [Keras](https://keras.io/), [Dlib](http://dlib.net/) and [OpenCV](https://opencv.org/). The process involves preprocessing images for face alignment, generating 128-dimensional face embeddings with a convolutional neural network (CNN), training classifiers on labeled embeddings and predicting identities of new inputs. |
| [streamz](https://github.com/krasserm/streamz) | Streamz is a combinator library designed to integrate [Functional Streams for Scala](https://fs2.io) (FS2), [Akka Streams](https://doc.akka.io/docs/akka/current/stream/index.html), and [Apache Camel](https://camel.apache.org/) endpoints, allowing seamless interoperability between these technologies. Camel endpoints can be integrated into FS2 applications with the [Camel DSL for FS2](https://github.com/krasserm/streamz/blob/master/streamz-camel-fs2/README.md) and into Akka Streams applications with the [Camel DSL for Akka Streams](https://github.com/krasserm/streamz/blob/master/streamz-camel-akka/README.md). |
| [akka-stream-eventsourcing](https://github.com/krasserm/akka-stream-eventsourcing) | This project brings to [Akka Streams](https://doc.akka.io/docs/akka/current/stream/index.html) what [Akka Persistence](https://doc.akka.io/docs/akka/current/persistence.html) brings to [Akka Actors](https://doc.akka.io/docs/akka/current/index-actors.html): persistence via event sourcing. In the same way as persistent actors in Akka Persistence, stateful *request processors* in akka-stream-eventsourcing form a consistency boundary around internal state but additionally provide type safety and end-to-end back-pressure for the whole event sourcing message flow. |


## Contributions

| Project | Contributions |
|---------|--------------|
| [sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime) | https://github.com/search?q=repo%3Aanthropic-experimental%2Fsandbox-runtime+author%3Akrasserm&type=pullrequests&s=created&o=desc |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | https://github.com/search?q=repo%3Aggml-org%2Fllama.cpp+author%3Akrasserm&type=pullrequests |
| [eventuate](https://github.com/rbmhtechnology/eventuate) | https://github.com/search?q=repo%3ARBMHTechnology%2Feventuate+author%3Akrasserm&type=commits |
| [akka](https://github.com/akka/akka) | https://github.com/search?q=repo%3Aakka%2Fakka-core+author%3Akrasserm&type=commits |
| [alpakka](https://github.com/akka/alpakka) | https://github.com/search?q=repo%3Aakka%2Falpakka+author%3Akrasserm&type=pullrequests |
| [akka-persistence-cassandra](https://github.com/akka/akka-persistence-cassandra) | https://github.com/search?q=repo%3Aakka%2Fakka-persistence-cassandra+author%3Akrasserm&type=commits |
| [eventsourced](https://github.com/eligosource/eventsourced) | https://github.com/search?q=repo%3Aeligosource%2Feventsourced+author%3Akrasserm&type=commits |
| [ipf](https://github.com/oehf/ipf) | https://github.com/search?q=repo%3Aoehf%2Fipf+author%3Akrasserm&type=commits |
| [camel](https://github.com/apache/camel) | https://github.com/search?q=repo%3Aapache%2Fcamel+krasser&type=commits |
