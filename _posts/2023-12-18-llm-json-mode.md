---
title: Schema-guided generation with open LLMs
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

**Update 2024-05-31**: This article has been updated to use Llama-3-8B-Instruct, instead of Llama-2-70B-Chat, and the latest revision of [grammar-based-agents](https://github.com/krasserm/grammar-based-agents).

Notebook ([original version](https://github.com/krasserm/grammar-based-agents/blob/wip-article-2/example_json.ipynb), [latest version](https://github.com/krasserm/grammar-based-agents/blob/master/json_mode.ipynb))  
Repository ([original version](https://github.com/krasserm/grammar-based-agents/tree/wip-article-2), [latest version](https://github.com/krasserm/grammar-based-agents/tree/master))

OpenAI recently introduced [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode) for its chat models. Anyscale provides a [similar service](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features) that additionally supports user-defined JSON schemas. Both do not disclose how this is done but it's relatively easy to implement it with grammar-based sampling in llama.cpp.

For this implementation I'll use an updated version of the components introduced in [Schema-guided generation in LangChain agents](/2023/12/10/grammar-based-agents/). These are a LangChain [LLM proxy](https://github.com/krasserm/grammar-based-agents/blob/master/gba/client/llamacpp.py) communicating with a model running on a llama.cpp server, enforcing a user-defined schema if provided, and an [LLM wrapper](https://github.com/krasserm/grammar-based-agents/blob/master/gba/client/chat.py) that applies a chat prompt template to incoming messages.

Two models are used here, a Llama-3-8B-Instruct model and a Mistral-7B-instruct model. Instructions for running them on a llama.cpp server are available [here](https://github.com/krasserm/grammar-based-agents/blob/master/README.md#getting-started). Application examples are taken from [this article](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features).

## Schema-guided generation


```python
import json
from typing import List

import jsonref
from pydantic import BaseModel, Field

from gba.client import ChatClient, Llama3Instruct, LlamaCppClient

# LLM proxy for an 8-bit quantized Llama-3-8B instruct model hosted on a llama.cpp server
llama3_llm = LlamaCppClient(url="http://localhost:8084/completion", temperature=-1)

# LLM wrapper that applies a Llama-3 chat prompt (+ exposes chat model interface)
llama3_chat = Llama3Instruct(llm=llama3_llm)

# Chat client used by application
llama3_client = ChatClient(llama3_chat)
```


```python
system_message = {"role": "system", "content": "You are a helpful assistant."}
```

### Basic example


```python
class GameResult(BaseModel):
    winner_team: str
    winner_score: int
    loser_team: str
    loser_score: int

user_message = {"role": "user", "content": "Who won the world series in 2020?"}

response = llama3_client.complete(
    messages=[system_message, user_message],
    schema=GameResult.model_json_schema(),
)
response["content"]
```




    '{"winner_team": "Los Angeles Dodgers", "winner_score": 4, "loser_team": "Tampa Bay Rays", "loser_score": 2}'



### Handling arrays


```python
class SortResult(BaseModel):
    """The format of the answer."""
    sorted_numbers: List[int] = Field(description="List of the sorted numbers")

user_message = {"role": "user", "content": "Sort the following numbers: 2, 8, 6, 7"}

response = llama3_client.complete(
    messages=[system_message, user_message],
    schema=SortResult.model_json_schema(),
)
response["content"]
```




    '{ "sorted_numbers": [2, 6, 7, 8] }'



### Handling nested structures

For handling nested structures, [jsonref](https://github.com/gazpachoking/jsonref) is used to to resolve references in schemas.


```python
class Person(BaseModel):
    """The object representing a person with name and age"""
    name: str = Field(description="Name of the person")
    age: int = Field(description="The age of the person")

class Result(BaseModel):
    """The format of the answer."""
    sorted_list: List[Person] = Field(description="List of the sorted objects")

user_message = {"role": "user", "content": "Alice is 10 years old, Bob is 7 and Carol is 2. Sort them by age in ascending order."}

response = llama3_client.complete(
    messages=[system_message, user_message],
    schema=jsonref.replace_refs(Result.model_json_schema()),
)
response["content"]
```




    '{ "sorted_list": [ {"name": "Carol", "age": 2}, {"name": "Bob", "age": 7}, {"name": "Alice", "age": 10} ] }'



## System prompt extensions

There is one issue though. Field descriptions in schemas are ignored because they are not included in the grammar. For example, if we add format hints to field descriptions like `... in uppercase`, they have no effect.


```python
class Person(BaseModel):
    name: str = Field(description="Name of the person in uppercase")
    age: int = Field(description="The age of the person")
    
class Result(BaseModel):
    sorted_list: List[Person] = Field(description="List of the sorted objects")

system_message = {"role": "system", "content": "You are a helpful assistant."}
user_message = {"role": "user", "content": "Alice is 10 years old, Bob is 7 and Carol is 2. Sort them by age in ascending order."}

response = llama3_client.complete(
    messages=[system_message, user_message],
    schema=jsonref.replace_refs(Result.model_json_schema()),
)
response["content"]
```




    '{ "sorted_list": [ {"name": "Carol", "age": 2}, {"name": "Bob", "age": 7}, {"name": "Alice", "age": 10} ] }'



This can be mitigated by adding field descriptions to the system prompt. The `object_from_schema` function generates a JSON object from the provided schema with field descriptions as values.


```python
from gba.utils import object_from_schema

schema = jsonref.replace_refs(Result.model_json_schema())
schema_instance = object_from_schema(schema)

system_prompt = f"""You are a helpful assistant. 

Generate JSON output in the following format:

{json.dumps(schema_instance, indent=2)}"""

system_message = {"role": "system", "content": system_prompt}

print(system_prompt)
```

    You are a helpful assistant. 
    
    Generate JSON output in the following format:
    
    {
      "sorted_list": [
        {
          "name": "Name of the person in uppercase",
          "age": "The age of the person"
        }
      ]
    }


Then the output is as expected.


```python
response = llama3_client.complete(
    messages=[system_message, user_message],
    schema=jsonref.replace_refs(Result.model_json_schema()),
)
response["content"]
```




    '{ "sorted_list": [ { "name": "CAROL", "age": 2 }, { "name": "BOB", "age": 7 }, { "name": "ALICE", "age": 10 } ] }'



## Support other models

Using other open models is straightforward as shown here for a Mistral-7b-instruct model. You just need to replace `Llama3Instruct` with `MistralInstruct` for applying a Mistral-specific chat prompt template. Examples of other chat prompt templates are [here](https://github.com/langchain-ai/langchain/pull/8295#issuecomment-1668988543) and [here](https://github.com/langchain-ai/langchain/pull/8295#issuecomment-1811914445).


```python
from gba.client import MistralInstruct

# LLM proxy for a Mistral-7b-instruct model hosted on a llama.cpp server
mistral_llm = LlamaCppClient(url="http://localhost:8081/completion", temperature=-1)

# LLM wrapper that applies the Mistral chat prompt (+ exposes chat model interface)
mistral_instruct = MistralInstruct(llm=mistral_llm)

# Chat client used by application
mistral_client = ChatClient(mistral_instruct)

response = mistral_client.complete(
    messages=[{"role": "user", "content": "Sort the following numbers: 2, 8, 6, 7"}],
    schema=SortResult.model_json_schema(),
)
response["content"]
```




    '{ "sorted_numbers": [2, 6, 7, 8] }'
