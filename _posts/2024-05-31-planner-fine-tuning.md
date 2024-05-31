---
title: Planner fine-tuning on synthetic agent trajectories
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

[Notebook](https://github.com/krasserm/grammar-based-agents/blob/master/planner_finetuned.ipynb)  
[Repository](https://github.com/krasserm/grammar-based-agents/tree/master)

In a [previous article](/2024/03/06/modular-agent/) I experimented with separating planning from function calling in [ReAct](https://arxiv.org/abs/2210.03629)-style LLM agents. A central planner module is responsible for describing the task for the next step and selecting an appropriate tool. The selected tool is responsible for translating the informal task description into tool-specific executable actions. Reducing the planner's responsibility to task formulation and tool selection only, without having to deal with function calling details, enables usage of smaller, less capable LLMs for planning. It also eases the generation of datasets for planner fine-tuning.

This article first how to generate a synthetic dataset for planner fine-tuning and how to fine-tune a 7B LLM to reach the performance of a GPT-4 based planner. To generate a synthetic dataset, we run an agent with a GPT-4 based planner in a simulation environment and record its trajectories. The interface to the environment is a set of simulated tools, instead or real ones. For example, a simulated `search_internet` tool, backed by GPT-4, generates search results from GPT-4's internal memory instead of actually searching the internet. 

For fine-tuning it is less important if observations are made in a simulation or a real environment. A planner has to learn to use these observations as-is, regardless whether they are hallucinated or not, and appropriately plan the next steps. The last part of this article runs the fine-tuned planner in a real environment, with a corresponding set of real tools as interface. The planner learns the available tools from the generated dataset so that they don't need to be specified in the prompt which can significantly reduce inference latencies. 

The predefined set of [simulated tools](https://github.com/krasserm/grammar-based-agents/tree/master/simulation/tools) and their corresponding [real tools](https://github.com/krasserm/grammar-based-agents/tree/master/gba/tools) used in this article are just examples and can be adjusted to whatever is needed for other applications. [Generating trajectories](https://github.com/krasserm/grammar-based-agents/tree/master/simulation/README.md#generate-trajectories) in a simulation environment and [planner fine-tuning](https://github.com/krasserm/grammar-based-agents/tree/master/train/README.md#planner-fine-tuning) with a different set of tools is straightforward with the framework provided by the [grammar-based-agents](https://github.com/krasserm/grammar-based-agents) project. The currently used tools are:

| Tool name          | Tool description                                                                          |
|--------------------|-------------------------------------------------------------------------------------------|
| `ask_user`         | Useful for asking user about information missing in the request.                          |
| `calculate_number` | Useful for numerical tasks that result in a single number.                                |
| `create_event`     | Useful for adding a single entry to my calendar at given date and time.                   |
| `search_wikipedia` | Useful for searching factual information in Wikipedia.                                    |
| `search_internet`  | Useful for up-to-date information on the internet.                                        |
| `send_email`       | Useful for sending an email to a single recipient.                                        |
| `use_bash`         | Useful for executing commands in a Linux bash.                                            |
| `final_answer`     | Useful for providing the final answer to a request. Must always be used in the last step. |

The `final_answer` tool is a special tool used by the agent for providing a final answer to the user. Simulated tools `search_internet` and `search_wikipedia` report with a probability of 0.1 that they couldn't find an answer to the query or provide an incomplete answer. This is helpful to make the planner more robust to error conditions during fine-tuning. The [corresponding real tools](https://github.com/krasserm/grammar-based-agents/tree/master/gba/tools/search) are fully-functional RAG-based search engines.

## Dataset generation

### Requests

For running an [agent simulation](https://github.com/krasserm/grammar-based-agents/tree/master/simulation), we first need to [generate](https://github.com/krasserm/grammar-based-agents/tree/master/simulation#generate-requests) a set of requests i.e. questions and instructions that can be answered in one or more steps using a combination of available tools. Requests are generated with GPT-4 for a variety of topics. Overall, 2780 requests are generated. Examples include:

- *Get the average Rotten Tomatoes scores for DreamWorks' last 5 movies.*
- *Email me articles about the renovation of the Taj Mahal happening this year.*
- *Compare the cost-per-click (CPC) and engagement rate of Snapchat ads to those of TikTok ads for the age group 18-24.*
- *Give me the average time of men's 200m freestyle winners for the past three Olympics and how it compares to the all-time record.*
- ...

### Trajectories

Trajectories for the 2780 training requests are [generated](https://github.com/krasserm/grammar-based-agents/tree/master/simulation#generate-trajectories) by an agent with a GPT-4 based planner and the predefined set of GPT-4 based simulated tools. A request like *Get the average Rotten Tomatoes scores for DreamWorks' last 5 movies*, for example, may result in [this trajectory](/docs/2024-05-31/trajectory.json), with a revised plan at each step. Note how `thoughts` at each step may plan several steps ahead whereas `task` describes the very next step only. Forcing the planner to summarize previous steps and reason about missing information or actions further improves planning accuracy.

### Quality filtering

Low quality trajectories must be excluded from the final training dataset. For [rating](https://github.com/krasserm/grammar-based-agents/tree/master/simulation#evaluate-trajectories) the quality of trajectories we ask GPT-4 to reflect on the steps and the final answer in the trajectory and assign ratings from 1 (poor) to 5 (excellent). Only trajectories with ratings 4 and 5 are accepted for the final training set.

## Training dataset

For each step in a trajectory a training example can be [generated](https://github.com/krasserm/grammar-based-agents/tree/master/simulation#package-dataset). It contains the initial request, task-observations pairs from previous steps and the expected planner output for the next step. An example prompt is

````
User request: 

```
Get the average Rotten Tomatoes scores for DreamWorks' last 5 movies. 
```

Context information: 

```
Task: Find the last 5 movies released by DreamWorks. 
Result: The last five movies released by DreamWorks are 'The Bad Guys' (2022), 'Boss Baby: Family Business' (2021), 'Trolls World Tour' (2020), 'Abominable' (2019), and 'How to Train Your Dragon: The Hidden World' (2019). 

Task: Search the internet for the Rotten Tomatoes score of 'The Bad Guys' (2022). 
Result: The Rotten Tomatoes score of 'The Bad Guys' (2022) is 88%. 
```

Plan the next step.
````

The expected output is:

```json
{
  "context_information_summary": "The last five movies released by DreamWorks are 'The Bad Guys' (2022), 'Boss Baby: Family Business' (2021), 'Trolls World Tour' (2020), 'Abominable' (2019), and 'How to Train Your Dragon: The Hidden World' (2019). The Rotten Tomatoes score for 'The Bad Guys' (2022) is 88%.", 
  "thoughts": "Since the Rotten Tomatoes score for 'The Bad Guys' was found through an internet search, the same approach should be used to find the score for 'Boss Baby: Family Business' (2021).", 
  "task": "Search the internet for the Rotten Tomatoes score of 'Boss Baby: Family Business' (2021).", 
  "selected_tool": "search_internet"
}
```

After [filtering](https://github.com/krasserm/grammar-based-agents/tree/master/simulation#package-dataset), we end up with a training set of 8579 examples. An important design decision was to omit the (static) set of available tools in the prompt and let the planner learn the available tools implicitly during fine-tuning. The advantage is a much shorter prompt and therefore lower inference latencies.

## Planner fine-tuning

The base model for [planner fine-tuning](https://github.com/krasserm/grammar-based-agents/tree/master/train#planner-fine-tuning) is Mistral-7B-v0.1. It is trained for 3 epochs on the generated dataset with QLoRA using [autotrain-advanced](https://github.com/huggingface/autotrain-advanced), running locally. The loss is currently computed on the prompt and completion because the prompt contains a significant amount of planning-specific data which are task-observation pairs from the agent's current trajectory. After merging the adapters back into the base model, they are [converted](https://github.com/krasserm/grammar-based-agents/tree/master/train#gguf-conversion-and-quantization) to GGUF, quantized to 8-bit and 4-bit and served on a llama.cpp server. 

## Planner evaluation

The fine-tuned planners are [evaluated](https://github.com/krasserm/grammar-based-agents/tree/master/simulation#planner-evaluation) in the simulation environment, together with the GPT-4 based planner and the zero-shot planner from the [previous article](/2024/03/06/modular-agent/). Evaluation is done on a separate test set of 50 requests.

| series          | pass_rate   | bad_task_rate | completion_rate |
|:----------------|:-----------:|:-------------:|:---------------:|
| zero-shot 8bit  | 0.72 ± 0.05 | 0.30 ± 0.04   | 0.88 ± 0.02     |
| fine-tuned 4bit | 0.89 ± 0.04 | 0.14 ± 0.01   | 0.96 ± 0.02     |
| fine-tuned 8bit | 0.88 ± 0.04 | 0.09 ± 0.01   | 0.95 ± 0.03     |
| gpt-4           | 0.91 ± 0.05 | 0.07 ± 0.01   | 0.97 ± 0.02     |

Basis for evaluation is the same rating procedure that has been used for filtering the training dataset. Evaluation metrics are pass rate, bad task rate and completion rate. 

- *pass rate* is defined as the fraction of requests that have been answered with a rating of 4 or higher.
- *bad task rate* is the fraction of steps with a task description rating of 3 or lower. 
- *completion rate* is the number of requests that the agent could complete with a final answer in 10 steps or less.

4 evaluation runs are executed for each planner to account for the non-deterministic behavior of simulated tools `search_internet` and `search_wikipedia`. These tools may decide with a probability of 0.1 to provide no answer or an incomplete answer. The mean and the standard deviation over 4 runs is reported in the table above.

The pass rate of the fine-tuned planners (8-bit and 4-bit) is within a single standard deviation of that of the GPT-4 based planner but significantly higher than that of the zero-shot planner. This doesn't say much about their efficiency though. A better metric for the efficiency is bad task rate. A higher number means longer and therefore less efficient trajectories. The 8-bit fine-tuned planner is close to the GPT-4 based planner and significantly better than the 4-bit quantized planner on this metric. Again, there is a large gap to the zero-shot planner. The completion rates of the fine-tuned planners and the GPT-4 based planner are similar with a large gap to the zero-shot planner.

## Real environment

An agent configured with the fine-tuned planner can also solve complex user requests in a real environment.


```python
from sentence_transformers import SentenceTransformer, CrossEncoder
from gba.client import MistralInstruct, Llama3Instruct, LlamaCppClient

# Proxy for 8-bit finetuned Mistral-7B-v0.1 planner
mistral_finetuned = MistralInstruct(
    llm=LlamaCppClient(url="http://localhost:8082/completion", temperature=-1),
)

# Proxy for 8-bit quantized Llama-3-8B-Instruct
llama3_instruct = Llama3Instruct(
    llm=LlamaCppClient(url="http://localhost:8084/completion", temperature=-1),
)

embedding_model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    device="cuda:0",
)

rerank_model = CrossEncoder(
    "mixedbread-ai/mxbai-rerank-large-v1",
    device="cuda:0",
)

searxng_endopoint = "http://localhost:8080"
```


```python
from gba.agent import Agent
from gba.client import ChatClient
from gba.planner import FineTunedPlanner
from gba.tools import *
from gba.tools.functions import create_event, send_email


search_wikipedia_tool = SearchWikipediaTool(
    llm=llama3_instruct,
    embedding_model=embedding_model,
    rerank_model=rerank_model,                    
    top_k_nodes=10,
    top_k_related_documents=1,
    top_k_related_nodes=3,
)

search_internet_tool = SearchInternetTool(
    llm=llama3_instruct,
    rerank_model=rerank_model,
    searxng_endpoint=searxng_endopoint,
    fetch_webpage_timeout=5.0,
    top_k_documents=3,
    top_k_nodes_per_document=5,
    top_k_snippets=None,
)

tools = [
    AskTool(),
    CalculateTool(model=llama3_instruct),
    FunctionCallTool(model=llama3_instruct, fn=create_event),
    FunctionCallTool(model=llama3_instruct, fn=send_email),
    search_internet_tool,
    search_wikipedia_tool,    
    RespondTool(model=llama3_instruct),
]

client = ChatClient(model=mistral_finetuned)
planner = FineTunedPlanner(client=client)
agent = Agent(planner=planner, tools=tools)
```

    Loading int8 search index...
    Loading document mapping...
    Loading binary search index...


Here's an example that uses the real tools `search_internet`, `create_request`, `ask_user` and `send_email`. The output shows the task, selected tool and the tool call result (observation) at each step. The `send_email` tool, defined as [application-specific function](https://github.com/krasserm/grammar-based-agents/blob/master/gba/tools/functions.py), additionally outputs the generated email body. The return value of the agent's `run` method is the final answer to the user.


```python
agent.run("create a reminder on the 3rd day of the next olympic winter games "
          "and invite my friend via email to join me watching on tv")
```

    Task: Search for the dates of the next Olympic Winter Games.
    Tool: search_internet
    Searching the internet for query 'next Olympic Winter Games dates'
    Observation: The next Olympic Winter Games, Milano Cortina 2026, will take place from February 6, 2026, to February 22, 2026.
    
    Task: Create a reminder for watching the Olympic Winter Games on February 8, 2026.
    Tool: create_event
    Observation: Event 'Olympic Winter Games' successfully added to calendar, date=2026-02-08, time=None
    
    Task: Ask the user for their friend's email address to send the invitation to watch the Olympic Winter Games together on TV.
    Tool: ask_user
    Observation: michael@example.com
    
    Task: Send an email to michael@example.com inviting him to watch the Olympic Winter Games together on TV on February 8, 2026.
    Tool: send_email
    Email body: Hi Michael, I'd love for you to join me in watching the Olympic Winter Games together on TV on February 8, 2026. The event starts at an unspecified time. Hope to see you then!
    Observation: Email sent to 'michael@example.com' with subject 'Join me to watch the Olympic Winter Games!'

    'A reminder has been created for the 3rd day of the next Olympic Winter Games, which is February 8, 2026, and an email invitation has been sent to michael@example.com to join me watching on TV.'


The next example uses the real tools `search_wikipedia` and `calculate_number` to answer a user question. The `calculate_number` tool additionally outputs the Python code generated to perform the calculation:


```python
agent.run("what is the name of the person who directed the movie Saving Private Ryan, "
          "the most successful movie directed by this person and the time difference "
          "in years between the release dates of the most successful movie and Saving "
          "Private Ryan?")
```

    Task: Search Wikipedia for the director of the movie Saving Private Ryan.
    Tool: search_wikipedia
    Searching wikipedia for query 'Director of Saving Private Ryan'
    Observation: The director of the movie Saving Private Ryan is Steven Spielberg.
    
    Task: Search Wikipedia for the most successful movie directed by Steven Spielberg.
    Tool: search_wikipedia
    Searching wikipedia for query 'most successful Steven Spielberg movie'
    Observation: Jurassic Park, during its release, grossed more than $914 million worldwide, becoming the most successful film released up to that time.
    
    Task: Search Wikipedia for the release date of the movie Saving Private Ryan.
    Tool: search_wikipedia
    Searching wikipedia for query 'Saving Private Ryan release date'
    Observation: The movie Saving Private Ryan was released on July 24, 1998.
    
    Task: Calculate the time difference in years between June 11, 1993 (Jurassic Park release date) and July 24, 1998 (Saving Private Ryan release date).
    Tool: calculate_number
    ```python
    from datetime import datetime
    jurassic_park_release_date = datetime(1993, 6, 11)
    saving_private_ryan_release_date = datetime(1998, 7, 24)
    time_difference = saving_private_ryan_release_date - jurassic_park_release_date
    result = time_difference.days / 365
    ```
    Observation: 5.120547945205479

    "The director of the movie Saving Private Ryan is Steven Spielberg, and his most successful movie is Jurassic Park, which was released in 1993, making it 5 years before Saving Private Ryan's release date."


It is interesting to see that the planner doesn't search for the release date of *Jurassic Park* but rather uses its internal knowledge to come up with the (correct) date of *June 11, 1993*. There is actually one training example that contains the Jurassic Park release date but in context of another request. It must be investigated if fine-tuning or base model pre-training causes the planner to leverage internal knowledge. In general, the fine-tuned planner has a strong tendency to lookup even trivial knowledge with search tools.

## Conclusion

Fine-tuning a 7B LLM on synthetic trajectories from an agent simulation results in a planner with a performance comparable to a GPT-4 based planner when evaluated on a test set of requests in a simulation environment. A qualitative evaluation in a real environment demonstrates versatile tool usage for solving complex user requests.

During fine-tuning, the planner learns available tools from the generated trajectories so that they don't need to be specified in the prompt which can significantly reduce inference latencies. The framework provided by the [grammar-based-agents](https://github.com/krasserm/grammar-based-agents) project can easily be adjusted to a different set of tools for specialization to other application domains.
