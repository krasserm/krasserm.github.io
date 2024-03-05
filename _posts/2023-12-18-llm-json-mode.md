---
title: Reliable JSON mode for open LLMs
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

[Notebook](https://github.com/krasserm/grammar-based-agents/blob/master/example_json.ipynb)  
[Repository](https://github.com/krasserm/grammar-based-agents)

OpenAI recently introduced [JSON mode](https://platform.openai.com/docs/guides/text-generation/json-mode) for its chat models. Anyscale provides a [similar service](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features) that additionally supports user-defined JSON schemas. Both do not disclose how this is done but it's relatively easy to implement it with grammar-based sampling in llama.cpp, and system prompt extensions.

For this implementation I'll reuse the components introduced in [Open LLM agents with schema-based generation of function calls](https://krasserm.github.io/2023/12/10/grammar-based-agents/). These are a LangChain [LLM proxy](https://krasserm.github.io/2023/12/10/grammar-based-agents/#llamacppclient) communicating with a model running on a llama.cpp server, enforcing a user-defined schema if provided, and an [LLM wrapper](https://krasserm.github.io/2023/12/10/grammar-based-agents/#llama2chat) that applies a chat prompt template to incoming messages.

Two models are used here, a Llama-2-70b-chat model and a Mistral-7b-instruct model. Instructions for running them on a llama.cpp server are available [here](https://github.com/krasserm/grammar-based-agents/blob/master/README.md#getting-started). Application examples are taken from [this article](https://www.anyscale.com/blog/anyscale-endpoints-json-mode-and-function-calling-features).

## Schema-based generation


```python
import json
from typing import List

import jsonref
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain_experimental.chat_models.llm_wrapper import Llama2Chat
from pydantic import BaseModel, Field

from gba.llm import LlamaCppClient

# LLM proxy for a Llama-2-70b model hosted on a llama.cpp server
llama2_llm = LlamaCppClient(url="http://localhost:8080/completion", temperature=-1)

# LLM wrapper that applies a Llama-2 chat prompt (+ exposes chat model interface)
llama2_chat = Llama2Chat(llm=llama2_llm)
```


```python
system_message = SystemMessage(content="You are a helpful assistant.")
```

### Basic example


```python
class GameResult(BaseModel):
    winner_team: str
    loser_team: str
    winner_score: int
    loser_score: int

human_message = HumanMessage(content="Who won the world series in 2020?")

response = llama2_chat.predict_messages(
    messages=[system_message, human_message],
    schema=GameResult.model_json_schema(),
)
response.content
```




    '{ "loser_score": 2, "loser_team": "Tampa Bay Rays", "winner_score": 4, "winner_team": "Los Angeles Dodgers" }'



Grammar-based sampling in llama.cpp orders JSON keys alphabetically, by default. This can be customized with `prop_order`.


```python
response = llama2_chat.predict_messages(
    messages=[system_message, human_message],
    schema=GameResult.model_json_schema(),
    prop_order=["loser_team", "loser_score", "winner_team", "winner_score"]
)
response.content
```




    '{ "loser_team": "Tampa Bay Rays", "loser_score": 2, "winner_team": "Los Angeles Dodgers", "winner_score": 4 }'



### Handling arrays


```python
class SortResult(BaseModel):
    """The format of the answer."""
    sorted_numbers: List[int] = Field(description="List of the sorted numbers")

human_message = HumanMessage(content="Sort the following numbers: 2, 8, 6, 7")

response = llama2_chat.predict_messages(
    messages=[system_message, human_message],
    schema=SortResult.model_json_schema(),
)
response.content
```




    '{"sorted_numbers": [2, 6, 7, 8]}'



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

human_message = HumanMessage(content="Alice is 10 years old, Bob is 7 and Carol is 2. Sort them by age in ascending order.")

response = llama2_chat.predict_messages(
    messages=[system_message, human_message],
    schema=jsonref.replace_refs(Result.model_json_schema()),
)
response.content
```




    '{ "sorted_list": [ { "age": 2, "name": "Carol" }, { "age": 7, "name": "Bob" }, { "age": 10, "name": "Alice" } ] }'



## System prompt extensions

There is one issue though. Field descriptions in schemas are ignored because they are not included in the grammar. For example, if we add format hints to field descriptions like `... in uppercase`, they have no effect.


```python
class Person(BaseModel):
    name: str = Field(description="Name of the person in uppercase")
    age: int = Field(description="The age of the person")

class Result(BaseModel):
    sorted_list: List[Person] = Field(description="List of the sorted objects")

system_message = SystemMessage(content="You are a helpful assistant.")
human_message = HumanMessage(content="Alice is 10 years old, Bob is 7 and Carol is 2. Sort them by age in ascending order.")

response = llama2_chat.predict_messages(
    messages=[system_message, human_message],
    schema=jsonref.replace_refs(Result.model_json_schema()),
)
response.content
```




    '{ "sorted_list": [ { "age": 2, "name": "Carol" }, { "age": 7, "name": "Bob" }, { "age": 10, "name": "Alice" } ] }'



This can be mitigated by adding field descriptions to the system prompt. The `object_from_schema` function generates a JSON object from the provided schema with field descriptions as values.


```python
from gba.utils import object_from_schema

schema = jsonref.replace_refs(Result.model_json_schema())
schema_instance, keys = object_from_schema(schema, return_keys=True)

system_prompt = f"""You are a helpful assistant. 

Generate JSON output in the following format:

{json.dumps(schema_instance, indent=2)}"""

system_message = SystemMessage(content=system_prompt)
print(system_message.content)
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
response = llama2_chat.predict_messages(
    messages=[system_message, human_message],
    schema=jsonref.replace_refs(Result.model_json_schema()),
)
response.content
```




    '{ "sorted_list": [ {"age": 2, "name": "CAROL"}, {"age": 7, "name": "BOB"}, {"age": 10, "name": "ALICE"} ] }'



For generating output with JSON keys in the same order as in our schema definition, we have to set `prop_order` to the list of `keys` returned from an `object_from_schema` call.


```python
response = llama2_chat.predict_messages(
    messages=[system_message, human_message],
    schema=jsonref.replace_refs(Result.model_json_schema()),
    prop_order=keys,
)
response.content
```




    '{ "sorted_list": [ {"name": "CAROL", "age": 2}, {"name": "BOB", "age": 7}, {"name": "ALICE", "age": 10} ] }'



## Support other models

Using other open models is straightforward as shown here for a Mistral-7b-instruct model. You just need to replace `Llama2Chat` with `MistralInstruct` for applying a Mistral-specific chat prompt template. Examples of other chat prompt templates are [here](https://github.com/langchain-ai/langchain/pull/8295#issuecomment-1668988543) and [here](https://github.com/langchain-ai/langchain/pull/8295#issuecomment-1811914445).


```python
from gba.chat import MistralInstruct

# LLM proxy for a Mistral-7b-instruct model hosted on a llama.cpp server
mistral_llm = LlamaCppClient(url="http://localhost:8081/completion", temperature=-1)

# LLM wrapper that applies the Mistral chat prompt (+ exposes chat model interface)
mistral_instruct = MistralInstruct(llm=mistral_llm)

response = mistral_instruct.predict_messages(
    messages=[HumanMessage(content="Sort the following numbers: 2, 8, 6, 7")],
    schema=SortResult.model_json_schema(),
)
response.content
```




    '{ "sorted_numbers": [2, 6, 7, 8] }'


