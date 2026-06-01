# Building an autonomous ML researcher with Claude Code dynamic workflows

This article was originally posted [here](https://gradion-ai.github.io/agents-nanny/2026/05/31/building-an-autonomous-ml-researcher-with-claude-code-dynamic-workflows/).

---

As an experiment, I re-implemented the autonomous ML research-and-engineering workflow encoded in Hugging Face's [`ml-intern`](https://github.com/huggingface/ml-intern) as a Claude Code [dynamic workflow](https://code.claude.com/docs/en/workflows) that delegates execution to the [Hugging Face skills](https://github.com/huggingface/skills) (`hf-skills`) instead of `ml-intern`'s custom tools<sup>1</sup>. I did it in three steps: extract a technology-neutral specification of the workflow, compile that specification into a single generic workflow script, then run the script against a concrete task. The result is one workflow that accepts any ML research task as an argument, rather than having Claude Code write a new workflow script for each task.

<!-- more -->

## From `ml-intern` to a technology-neutral spec

<table class="post-grid"><tr><td>
<p>I first <a href="https://gradion-ai.github.io/agents-nanny/ml-research-workflow/spec-prompt.txt">prompted</a> Claude Code to write a <a href="https://gist.github.com/krasserm/42e7fffcdb47f25d0f54b83488fadfcd#file-ml-research-workflow-spec-md">technology-neutral specification</a> of the workflow <code>ml-intern</code> encodes in its system prompts and agentic loop. It names abstract capabilities (search papers, inspect a dataset, submit a job, track metrics) and, in an appendix, maps each onto a concrete <code>hf-skill</code>. It is a behavioral contract, not an implementation.</p>
<p>The spec is a research-first, validate-before-spend, monitor-and-iterate loop with hard rules (persistence, no silent scope changes, sized timeouts, an explicit OOM recovery order) and a control contract (bounded iteration, repetition guards, approval gates). Its central principle: assume internal ML knowledge is stale, so ground every config and import in freshly retrieved literature and working example code rather than writing ML code from memory.</p>
</td><td>
<a href="https://gradion-ai.github.io/agents-nanny/posts/ml-research-workflow/workflow-spec.html"><img alt="The workflow as specified" src="https://gradion-ai.github.io/agents-nanny/ml-research-workflow/workflow-spec-diagram-sm.png" /></a>
<p><sub>The workflow as specified (<a href="https://gradion-ai.github.io/agents-nanny/posts/ml-research-workflow/workflow-spec.html">interactive version</a>).</sub></p>
</td></tr></table>

## From spec to a generic workflow script

<table class="post-grid"><tr><td>
<p>I then <a href="https://gradion-ai.github.io/agents-nanny/ml-research-workflow/script-prompt.txt">prompted</a> Claude Code to compile the spec into a <a href="https://gist.github.com/krasserm/42e7fffcdb47f25d0f54b83488fadfcd#file-ml-research-js">generic workflow script</a> invocable with the <code>Workflow</code> tool. Beyond transcribing the spec, the prompt asked for three additions: critics where the discipline needs enforcing, HF Jobs failure analysis and retry as a first-class concern, and a free local CPU smoke test (one train step, one eval step) before the first paid GPU job.</p>
<p>The result is a 13-phase pipeline where each phase is a subagent that calls the relevant <code>hf-skill</code>; the script contributes only orchestration and gates. Resources are verified before use, generated code is reviewed before it runs, paid jobs auto-submit only under a cost cap, and every fix passes a scope-change guard that refuses to swap the method, model, dataset, or sequence length to escape an error.</p>
</td><td>
<a href="https://gradion-ai.github.io/agents-nanny/posts/ml-research-workflow/workflow-script.html"><img alt="The 13-phase pipeline in ml-research.js" src="https://gradion-ai.github.io/agents-nanny/ml-research-workflow/workflow-script-diagram-sm.png" /></a>
<p><sub>The 13-phase pipeline (<a href="https://gradion-ai.github.io/agents-nanny/posts/ml-research-workflow/workflow-script.html">interactive version</a>).</sub></p>
</td></tr></table>

## How the spec maps to the script

<table class="post-grid"><tr><td>
<p>The mapping is close to one-to-one, with a few deliberate differences. The spec's single research phase splits into research plus an adversarial verify. Its implementation-and-preflight phase fans out into implement, code review, a free CPU smoke, and a tiny billable GPU preflight, the script's answer to the spec's acknowledged sandbox gap (no <code>hf-skill</code> provides a disposable GPU sandbox).</p>
<p>The script also adds explicit gates for job readiness and final conformance, and omits the spec's optional improvement loop at my request, so it stops at the first verified result instead of continuing to chase a better one through hyperparameter sweeps and other refinements (different data, approach, ...). The mapping view labels each correspondence as kept, adapted, added, or omitted.</p>
</td><td>
<a href="https://gradion-ai.github.io/agents-nanny/posts/ml-research-workflow/spec-to-script-map.html"><img alt="Spec to script phase mapping" src="https://gradion-ai.github.io/agents-nanny/ml-research-workflow/spec-to-script-map-diagram-sm.png" /></a>
<p><sub>Spec to script mapping (<a href="https://gradion-ai.github.io/agents-nanny/posts/ml-research-workflow/spec-to-script-map.html">interactive version</a>).</sub></p>
</td></tr></table>

## A first end-to-end run

I invoked the generic script with one line, the task passed as an argument:

```
/ml-research Fine-tune Qwen/Qwen2-0.5B on trl-lib/Capybara with LoRA on HF Jobs (cost cap is 10 USD)
```

The run completed on the first attempt and spent 2.14 USD of the 10 USD cap, producing a public [LoRA adapter](https://huggingface.co/krasserm/qwen2-0.5b-capybara-sft) and a [Trackio dashboard](https://huggingface.co/spaces/krasserm/huggingface-static-d6070a). The recipe was synthesized from papers the workflow verified actually exist (LoRA r=8 on `q_proj,v_proj` at lr 1e-4, effective batch 64, cosine schedule, 2 epochs, max_length 2048), grounded in the [LoRA](https://huggingface.co/papers/2106.09685) and [Secret Recipe](https://huggingface.co/papers/2412.13337) papers. The full job ran on a single L40S for about 59 minutes with clean monotonic eval loss convergence.

The clearest evidence for the failure-analysis loop appeared during GPU preflight: the workflow hit two non-fatal monitoring bugs (a pyarrow/Trackio serialization crash on the LoRA config, and a double `trackio.finish()`), diagnosed each from the logs, and fixed them before the full job, which is exactly what the cheap preflight is for: catching these bugs before they waste a paid run.

## Conclusion

An autonomous research process, like the one `ml-intern` implements, can be captured as a technology-neutral spec and compiled into a Claude Code dynamic workflow with two simple prompts. The workflow contributes only discipline (phase ordering, verification gates, persistence and anti-scope-change rules, failure analysis) and stays generic over the task, so a new task is an argument rather than new code. Having Claude Code write task-specific workflow scripts is also possible but was not attempted here. A preliminary conclusion, from a single end-to-end run, is that Claude Code dynamic workflows (currently in research preview) are already a practical path to spec-driven research automation.

---

<p class="footnote"><sup>1</sup> <code>ml-intern</code> is a self-contained application that ships its own agent loop, tools, and UI. Claude Code already provides that scaffolding and <code>hf-skills</code> wraps the Hugging Face tools (search, datasets, jobs, tracking), so the only part neither provides is the research process definition, which is what the spec captures.</p>