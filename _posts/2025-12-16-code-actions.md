---
title: 'Code Actions as Tools'
subtitle: 'Evolving Tool Libraries for Agents'
layout: post
comments: True
author: "Martin Krasser"
header-img: "img/distributed.png"
---

This article was originally posted [here](https://gradion-ai.github.io/agents-nanny/2025/12/16/code-actions-as-tools-evolving-tool-libraries-for-agents/).

---

Programmatic tool calling is gaining traction in agent development. Instead of emitting one JSON tool call at a time, agents generate executable "code actions" that call tools in a sandboxed environment. This pattern is inspired by Apple's [CodeAct](https://machinelearning.apple.com/research/codeact) and appears in many agentic systems. More [recent](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling) [implementations](https://blog.cloudflare.com/code-mode/) increasingly focus on programmatic calling of MCP tools.

These solutions typically generate Python or TypeScript APIs for MCP tools, let an agent write code actions that call these APIs, execute the code in a sandboxed environment, and feed results back to the agent. This improves performance compared to JSON-based approaches, but it often misses an important point: a generated code action can itself become a tool, available for reuse in later code actions.

This article fills that gap by presenting an approach for building reusable tools from code actions. When following certain design principles, code actions can be saved in a way that supports efficient discovery, inspection, and reuse in later code actions. This is demonstrated by example using a [Claude Code plugin](https://gradion-ai.github.io/ipybox/ccplugin/) that bundles a code action skill, and the [ipybox](https://gradion-ai.github.io/ipybox/) MCP server for local, sandboxed code execution.

## Code Actions as Reusable Tools

Most programmatic tool calling implementations treat code actions as ephemeral: generated, executed, then discarded. But a working code action represents a tested solution. An agent iterates on it based on execution feedback until it produces the desired outcome. If it is then saved in a discoverable format with a callable API, that code action becomes a tool that future code actions can import and compose with other tools.

Code actions can also be modified after they have been stored, whether to fix bugs discovered during execution or to add new functionality. The agent thus serves two roles: a domain-specific agent performing the task at hand, and a toolsmith evolving its own capabilities.

The key difference from JSON tool calling is mutability. In a JSON-based approach, tools are static: defined at development time and immutable at runtime. With code actions as tools, an agent's tool library can evolve at runtime: tools can be added, modified, or composed based on what the agent learns while working.

## Example: Composing GitHub MCP Server Tools

The approach is demonstrated using tools from the GitHub MCP server. The task is to retrieve the latest commits from a user's most-starred repositories. This requires composing two tools: `search_repositories` to find repositories, and `list_commits` to fetch commits for each repository.

> **Setup:** For environment setup, plugin installation, and a step-by-step walkthrough of the GitHub example in Claude Code, refer to the [plugin documentation](https://gradion-ai.github.io/ipybox/ccplugin/). The subsections below provide an overview of the core ideas.

### Generating Tool APIs

To generate a Python API for the GitHub MCP server tools, register the server with ipybox by providing its connection details:

```
# -------------
# User prompt
# -------------

Register this MCP server at ipybox under name github
{
  "url": "https://api.githubcopilot.com/mcp/",
  "headers": {
    "Authorization": "Bearer ${GITHUB_API_KEY}"
  }
}
```

ipybox then generates one module per tool, such as [search_repositories](https://github.com/gradion-ai/ipybox/blob/main/docs/generated/mcptools/github/search_repositories_orig.py) or [list_commits](https://github.com/gradion-ai/ipybox/blob/main/docs/generated/mcptools/github/list_commits.py). Each generated module exposes a `run()` function to invoke the MCP tool, and a typed input parameter model:

```python
# -------------------------
# search_repositories API
# -------------------------

class Params(BaseModel):
    query: str
    sort: Sort | None = None
    order: Order | None = None
    perPage: int | None = None

def run(params: Params) -> str:
    # Invokes the MCP tool and returns the result
    ...
```

### Augmenting Tools with Output Types

Many MCP servers, including GitHub's, do not provide output schemas. GitHub MCP tools return JSON strings with undocumented structure. Without knowing that structure in advance, an agent cannot reliably write code that processes outputs inside a code action. The agent would have to call the tool first, bring the raw output back into its context to inspect it, and only then write processing code. That undermines the goal of handling intermediate results inside the execution environment.

To address this, we ask the agent to generate an output parser for `search_repositories`:

```
# -------------
# User prompt
# -------------

Use the codeact skill to generate an output parser for search_repositories
```

The agent calls the tool's `run()` function with sample inputs, inspects representative responses, and then augments the tool API with a structured output type plus a `run_parsed()` function:

```python
# --------------------------------------
# search_repositories API augmentation
# --------------------------------------

class Repository(BaseModel):
    name: str
    stargazers_count: int
    # ... other fields extracted from actual responses

class ParseResult(BaseModel):
    repositories: list[Repository]

def run_parsed(params: Params) -> ParseResult:
    # Import parser logic from a separate, generated module
    from mcpparse.github.search_repositories import parse
    return parse(run(params))
```

Importantly, output parsers are stored separately from the original tool API. This keeps the tool interface clean and avoids mixing [parsing implementation details](https://github.com/gradion-ai/ipybox/blob/main/docs/generated/mcpparse/github/search_repositories.py) into the interface definition.

Once generated, the [augmented tool API](https://github.com/gradion-ai/ipybox/blob/main/docs/generated/mcptools/github/search_repositories.py) is immediately usable in code actions. For tools that do provide output schemas, ipybox generates structured output types during tool API generation, making this augmentation step unnecessary.

### Composing Tools in a Code Action

With typed outputs available, the agent can now compose tools within a single code action. For a task like

```
# -------------
# User prompt
# -------------

Use the codeact skill to get the latest 5 commits of the 3 github repos
of torvalds with the most stars. For each repo, output name, stars and
the first line of commit messages, and the link to the commit
```

the agent generates a code action that combines the `search_repositories` and `list_commits` tools.

```python
# -------------
# Code action
# -------------

from mcptools.github import search_repositories, list_commits

# Search returns typed Repository objects
repos = search_repositories.run_parsed(
    search_repositories.Params(query="user:torvalds", sort="stars", perPage=3)
)

# Iterate and call list_commits for each repo, all within one code action
for repo in repos.repositories:
    commits = list_commits.run(
        list_commits.Params(owner="torvalds", repo=repo.name, perPage=5)
    )
    # Printed results are returned to the agent
    for commit in json.loads(commits):
        first_line = commit["commit"]["message"].split("\n")[0]
        print(f"{repo.name}: {first_line}")
```

This single code action makes four tool calls (one `search_repositories`, then three `list_commits`). With JSON-based tool calling, the same workflow requires four separate inference passes, with intermediate results accumulating in the context window.

For brevity, the above example parses `list_commits` output inline. In practice, you should generate an output parser for it as well.

### Saving Code Actions for Reuse

Once a code action works, it can be saved as a parameterized tool for later reuse:

```
# -------------
# User prompt
# -------------

Save this as code action under github category with name commits_of_top_repos. 
Make username, top_n_repos and last_n_commits parameters.
```

A key design choice here is to separate interface from implementation. The interface is defined in a [commits_of_top_repos.api](https://github.com/gradion-ai/ipybox/blob/main/docs/generated/gentools/github/commits_of_top_repos/api.py) module, the implementation in [commits_of_top_repos.impl](https://github.com/gradion-ai/ipybox/blob/main/docs/generated/gentools/github/commits_of_top_repos/impl.py). The plugin's code action skill instructs the agent to follow this structure.

The API module defines a parameterized `run()` function, output types, and a tool description via a docstring:

```python
# -------------------------
# commits_of_top_repos API
# -------------------------

class CommitInfo(BaseModel):
    sha: str
    message: str
    url: str

class RepoCommits(BaseModel):
    name: str
    stars: int
    commits: list[CommitInfo]

def run(username: str, top_n_repos: int = 3, last_n_commits: int = 5) -> list[RepoCommits]:
    """Get latest commits from a user's top repositories by stars."""
    ...
```

During tool discovery, only the API needs inspection. The implementation stays hidden. This saves tokens during discovery and reduces distraction by keeping non-essential implementation details out of the agent's context.

New code actions can then import and reuse the saved tool directly:

```python
# -------------
# Code action
# -------------

from gentools.github.commits_of_top_repos import run

# Reuse the saved code action as a tool
results = run(username="torvalds", top_n_repos=3, last_n_commits=5)
```

## Tool Discovery

As the number of MCP tool APIs and saved code actions grows, loading them all into the agent's context window upfront becomes impractical. Even a moderately sized tool collection can consume a large fraction of the available context, leaving less room for the actual task.

Progressive tool discovery addresses this by deferring tool loading until it is needed. Tools are organized in a package hierarchy on the filesystem that an agent can explore. When a task arrives, the agent searches the filesystem for relevant tools and inspects the APIs of promising candidates. This approach trades some discovery overhead for substantial context savings.

The plugin's discovery mechanism is intentionally simple. More sophisticated options are possible, such as hybrid vector and keyword search over tool descriptions, but filesystem-based discovery is a practical starting point. It can scale to hundreds of tools without introducing additional infrastructure dependencies.

## Conclusion

Programmatic tool calling brings performance gains over JSON-based approaches, but it reaches its full potential when code actions become first-class tools. In a JSON-based approach, tools are static: defined at development time and immutable at runtime. With code actions as tools, an agent's tool library can evolve at runtime: tools can be added, modified, or composed based on what the agent learns while working. For efficient tool discovery and use, it is important to save code actions with typed interfaces separated from implementations. Agents can then inspect signatures and docstrings without loading implementation details.
