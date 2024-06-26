---
title: Impact of prompt masking on LLM agent planning performance
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

I recently experimented with [fine-tuning an LLM agent on synthetic trajectories](/2024/05/31/planner-fine-tuning/). 
During fine-tuning, a planner module learns to create a task description for the next step and select an appropriate tool to execute that step. 
The selected tool is responsible for translating the informal task description into tool-specific executable actions and returning a (summarized) observation to the planner. 

[Fine-tuning](https://github.com/krasserm/bot-with-plan/tree/master/train#gba-planner-7b-v01) was done with a loss over the full sequence i.e. prompt and completion tokens. [Common practice](https://sebastianraschka.com/blog/2024/llm-research-insights-instruction-copy.html#11-instruction-masking-during-instruction-finetuning) however is to compute the loss over completion tokens only and masking the prompt. This difference is illustrated in the Figures 1 and 2:

<a href="/img/2024-06-26/masking-1.png" target="_blank"><img src="/img/2024-06-26/masking-1.png" alt="masking-1"></a>

*Figure 1*. Training example with loss over prompt and completion tokens.

<a href="/img/2024-06-26/masking-2.png" target="_blank"><img src="/img/2024-06-26/masking-2.png" alt="masking-2"></a>

*Figure 2*. Training example with loss over completion tokens only.

Motivation for computing the loss over the full sequence was to learn from previous task-observation pairs available in a prompt, a pattern that might be useful for predicting the next step in an agent trajectory.
This is related to [Instruction Tuning With Loss Over Instructions](https://arxiv.org/abs/2405.14394), a paper that reports improved downstream tasks performance when applying a loss function to the instruction (= prompt) and completion part of training data.  

To investigate if prompt masking has a significant impact on LLM agent planning performance, I [fine-tuned two planner models](https://github.com/krasserm/bot-with-plan/tree/master/train#gba-planner-7b-v02), one with prompt masking and the other without, and [evaluated their performance](https://github.com/krasserm/bot-with-plan/tree/master/simulation#prompt-masking) in a [simulation environment](https://github.com/krasserm/bot-with-plan#environments), with a GPT-4 based planner as reference.
The reported metrics are `pass_rate`, `bad_task_rate` and `completion_rate` (details [here](https://github.com/krasserm/bot-with-plan/tree/master#evaluation)):

| series                 | pass_rate   | bad_task_rate | completion_rate |
|:-----------------------|:-----------:|:-------------:|:---------------:|
| fine-tuned w/ masking  | 0.85 ± 0.01 | 0.14 ± 0.01   | 0.98 ± 0.01     |
| fine-tuned w/o masking | 0.88 ± 0.01 | 0.12 ± 0.01   | 0.99 ± 0.01     |
| gpt-4                  | 0.90 ± 0.01 | 0.11 ± 0.01   | 0.98 ± 0.01     |

Observations made in the simulation environment have high variance, incl. missing or partial observations with a configurable probability. Therefore, metric statistics (mean and standard error) are calculated from 12 evaluations runs. 

Fine-tuning with prompt masking seems to result in decreased planner performance but with only weak support from a statistical significance test: a t-test on the metrics of series `fine-tuned w/ masking` and `fine-tuned w/o masking` gives a p-value of 0.10 for `pass_rate` and 0.22 for `bad_task_rate`. 
Using a p-value of 0.05 as significance threshold, a preliminary conclusion is that prompt masking doesn't have significant impact on planner performance, at least in the used simulation environment.
