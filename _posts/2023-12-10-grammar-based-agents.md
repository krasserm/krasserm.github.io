---
title: Open LLM agents with schema-guided generation of function calls
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

[Notebook](https://github.com/krasserm/grammar-based-agents/blob/master/example_agent.ipynb)  
[Repository](https://github.com/krasserm/grammar-based-agents)

LLM agents use large language models (LLMs) to decide which tools to use for interacting with their environment during multi-step task solving. Tool usage is done via function calling. The LLM is prompted or fine-tuned to generate JSON output representing one or more function calls. To ensure that the JSON output of an LLM follows a tool-specific schema one can use constrained decoding methods to control next token generation.

Here, [grammar-based sampling](https://github.com/ggerganov/llama.cpp/pull/1773) available in llama.cpp is used for constrained decoding. A JSON schema of available tools is converted to a grammar used for generating instances of that schema (= tool calls) during decoding. This is used for implementing a function calling interface for a Llama-2-70b model, an LLM with (limited) tool usage capabilities.

The [implementation](https://github.com/krasserm/grammar-based-agents) uses LangChain interfaces and is compatible LangChain's [agent framework](https://python.langchain.com/docs/modules/agents/). In its current state, it is a simple prototype for demonstrating schema-guided generation in LangChain agents. It is general enough to be used with many other language models supported by llama.cpp, after some tweaks to the prompt templates.

## Agent

A Llama-2 agent with schema-guided generation of function calls can be created as follows. Further details are described in section [Components](#components).


```python
from langchain.agents import AgentExecutor  
from langchain.tools import StructuredTool  
from langchain_experimental.chat_models.llm_wrapper import Llama2Chat

from gba.agent import Agent
from gba.llm import LlamaCppClient  
from gba.math import Llama2Math
from gba.tool import ToolCalling

from example_tools import (
    create_event,  
    search_images,  
    search_internet,  
)

# Custom LangChain LLM implementation that interacts with a model 
# hosted on a llama.cpp server
llm = LlamaCppClient(url="http://localhost:8080/completion", temperature=-1)  
  
# Converts incoming messages into a Llama-2 compatible chat prompt 
# and implements the LangChain chat model interface
chat_model = Llama2Chat(llm=llm)  
  
# Layers a tool calling protocol on top of chat_model resulting in
# an interface similar to the ChatOpenAI function calling interface
tool_calling_model = ToolCalling(model=chat_model)

# Proxy for a math LLM hosted on a llmama.cpp server. Backend of the
# `calculate` tool.
math_llm = LlamaCppClient(url="http://localhost:8088/completion", temperature=-1)
math_model = Llama2Math(llm=math_llm)

def calculate(expression: str):
    """A tool for evaluating mathematical expressions. Do not use variables in expression."""
    return math_model.run(message=expression)

# List of tools created from Python functions and used by the agent  
tools = [  
    StructuredTool.from_function(search_internet),  
    StructuredTool.from_function(search_images),  
    StructuredTool.from_function(create_event),  
    StructuredTool.from_function(calculate),  
]
 
# Custom LangChain agent implementation (similar to OpenAIFunctionsAgent)
agent_obj = Agent.from_llm_and_tools(
    model=tool_calling_model, 
    tools=tools,
)

# LangChain's AgentExecutor for running the agent loop
agent = AgentExecutor.from_agent_and_tools(
    agent=agent_obj, 
    tools=tools, 
    verbose=True,
)
```

The `llm` is a proxy for a 4-bit quantized [Llama-2 70b chat model](https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF) hosted on a [llama.cpp server](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md). Tools in this example are mainly mockups except the `calulate` tool which uses another LLM to interpret and evaluate mathematical queries. Instructions for serving these LLMs are available [here](https://github.com/krasserm/grammar-based-agents/blob/master/README.md#getting-started).

### Tool usage

Here's an example of a request that is decomposed by the agent into multiple steps, using a tool at each step. To increase the probability that the model generates an appropriate tool call at each step i.e. selects the right tool and arguments, an unconstrained thinking phase precedes the constrained tool call generation phase. This enables attention to thinking results during the tool call generation phase (details in section [ToolCalling](#ToolCalling)).


```python
agent.run("Who is Leonardo DiCaprio's current girlfriend and what is her age raised to the 0.24 power?")
```

````
> Entering new AgentExecutor chain...

Thoughts: I need to call the tool 'search_internet' to find out who Leonardo DiCaprio's current girlfriend is and then calculate her age raised to the 0.24 power using the tool 'calculate'.
Invoking: `search_internet` with `{'query': "Leonardo DiCaprio's current girlfriend"}`
Leonardo di Caprio started dating Vittoria Ceretti in 2023. She was born in Italy and is 25 years old

Thoughts: I need to call another tool to obtain more information, specifically the "calculate" tool, to calculate Vittoria Ceretti's age raised to the power of 0.24.
Invoking: `calculate` with `{'expression': '25^0.24'}`
Executing Llama2Math Python code:
```
result = pow(25, 0.24)
```
2.16524

Thoughts: I have enough information to respond with a final answer to the user.
Vittoria Ceretti, 2.16524

> Finished chain.
````

In the last step, the agent responds to the user directly with the final answer. The answer `Vittoria Ceretti, 2.16524` is short but correct (at the time of writing).

### Direct response

The agent may also decide to respond to the user immediately, without prior tool usage.


```python
agent.run("Tell me a joke")
```

```
> Entering new AgentExecutor chain...

Thoughts: I can directly respond to the user with a joke, here's one: "Why don't scientists trust atoms? Because they make up everything!"
Why don't scientists trust atoms? Because they make up everything!

> Finished chain.
```

### External memory

For maintaining conversational state with the user, the agent can be configured with external `memory`.


```python
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder  

chat_history = MessagesPlaceholder(variable_name="chat_history")  
memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
)

# Agent that additionally maintains conversational state with the user
conversational_agent_obj = Agent.from_llm_and_tools(
    model=tool_calling_model, 
    tools=tools, 
    extra_prompt_messages=[chat_history],
)  
conversational_agent = AgentExecutor.from_agent_and_tools(
    agent=conversational_agent_obj, 
    tools=tools, 
    memory=memory, 
    verbose=True,
)
```

Let's start the conversation by requesting an image of a brown dog.


```python
conversational_agent.run("find an image of a brown dog")
```

```
> Entering new AgentExecutor chain...

Thoughts: I need to call the tool search_images(query: str) to find an image of a brown dog.
Invoking: `search_images` with `{'query': 'brown dog'}`
[brown_dog_1.jpg](https://example.com/brown_dog_1.jpg)

Thoughts: I have enough information to respond with a final answer, which is the URL of the image file "brown_dog_1.jpg".
Here is an image of a brown dog: <https://example.com/brown_dog_1.jpg>

> Finished chain.
```

The next request is in context of the previous one, and the agent updates the search query accordingly.

```python
conversational_agent.run("dog should be running too")
```

```
> Entering new AgentExecutor chain...

Thoughts: I need to call another tool to search for images of a running dog.
Invoking: `search_images` with `{'query': 'brown dog running'}`
[brown_dog_running_1.jpg](https://example.com/brown_dog_running_1.jpg)

Thoughts: I have enough information to respond with a final answer, which is the URL of the image of a brown dog running.
https://example.com/brown_dog_running_1.jpg

> Finished chain.
```

The agent was able to create the query `brown dog running` only because it had access to conversational state (`brown` is mentioned only in the first request).

## Components

### `LlamaCppClient`

`LlamaCppClient` is a proxy for a model hosted on a llama.cpp server. It implements LangChain's `LLM` interface and relies on the caller to provide a valid Llama-2 chat prompt.

```python
prompt = "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nFind an image of brown dog [/INST]"
llm(prompt)
```

```
"Sure, here's a cute image of a brown dog for you! 🐶💩\n\n[Image description: A photo of a brown dog with a wagging tail, sitting on a green grassy field. The dog has a playful expression and is looking directly at the camera.]\n\nI hope this image brings a smile to your face! Is there anything else I can assist you with? 😊"
```

If a tool schema is provided, it is converted by `LlamaCppClient` to a grammar used for grammar-based sampling on the server side so that the output is a valid instance of that schema.


```python
from gba.tool import tool_to_schema

tool_schema = tool_to_schema(StructuredTool.from_function(search_images))
tool_schema
```

```
{'type': 'object',
 'properties': {'tool': {'const': 'search_images'},
  'arguments': {'type': 'object',
   'properties': {'query': {'type': 'string'}}}}}
```

```python
llm(prompt, schema=tool_schema)
```

```
'{ "arguments": { "query": "brown dog" }, "tool": "search_images" }'
```

### `Llama2Chat`

[Llama2Chat](https://python.langchain.com/docs/integrations/chat/llama2_chat) wraps `llm` and implements a chat model interface that applies the Llama-2 chat prompt to incoming messages (see also [these](https://github.com/langchain-ai/langchain/pull/8295#issuecomment-1668988543) [examples](https://github.com/langchain-ai/langchain/pull/8295#issuecomment-1811914445) for other chat prompt formats).


```python
from langchain.schema.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="Find an image of brown dog"),
]
chat_model.predict_messages(messages=messages)
```

```
AIMessage(content="Sure, here's a cute image of a brown dog for you! 🐶💩\n\n[Image description: A photo of a brown dog with a wagging tail, sitting on a green grassy field. The dog has a playful expression and is looking directly at the camera.]\n\nI hope this image brings a smile to your face! Is there anything else I can assist you with? 😊")
```

```python
chat_model.predict_messages(messages=messages, schema=tool_to_schema(StructuredTool.from_function(search_images)))
```

```
AIMessage(content='{ "arguments": { "query": "brown dog" }, "tool": "search_images"}')
```

### `ToolCalling`

`ToolCalling` adds tool calling functionality to the wrapped `chat_model`, resulting in an interface very similar to the function calling interface of `ChatOpenAI`. It converts a list of provided tools to a [oneOf](https://json-schema.org/understanding-json-schema/reference/combining#oneof) JSON schema and submits it to the wrapped chat model. It also constructs a system prompt internally that informs the backend LLM about the presence of these tools and adds a special `respond_to_user` tool used by the LLM for providing a final response.

```python
chat = [HumanMessage(content="What is (2 * 5) raised to power of 0.8 divided by 2?")]
tool_calling_model.predict_messages(messages=chat, tools=tools)
```

```
Thoughts: I need to call the calculate tool to evaluate the expression and obtain the result.
AIMessage(content='', additional_kwargs={'tool_call': {'tool': 'calculate', 'arguments': {'expression': '(2 * 5) ** 0.8 / 2'}}})
```

The wrapped `chat_model` first receives a message with an instruction to think about the current state and what to do next. This message doesn't contain a schema so that the model output is unconstrained. Then the model receives another message with an instruction to act i.e. respond with a tool call. This message contains the JSON schema of the provided tools so that the tool call can be generated with grammar-based sampling.

## Outlook

This two-phase approach of unconstrained thinking and constrained tool call generation may also be the basis for using more specialized models e.g. one for the thinking phase and another one for the tool call generation phase. I plan to present this in another article.
