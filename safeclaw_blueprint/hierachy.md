

Here is the transcript of the video, supplemented with Mermaid diagrams to represent the concepts and structures drawn on the glass board by the speaker.

### Single AI Agents and their Challenges

AI agents have a problem. They take in a request, and then they work autonomously to achieve a goal. But in long-horizon tasks, it can be tricky to keep the agent focused across all of the different steps that the agent needs to execute. Because you see, in a single agent architecture, you might well hit a few predictable failure modes. Like, for example, **context dilution**. As the task grows, the signal of the original goal gets a bit lost in the noise of the intermediate steps. Or we might have problems with **tool saturation**. The more tools you give the agent access to, the harder the tool selection becomes, and the more chances there are to call the wrong tool, or maybe just to pass invalid arguments. And there's also the **lost in the middle** phenomenon. Even if the right instruction is in the prompt, LLMs can underweight content buried in the middle of a long context window. 

```mermaid
graph LR
    A[Single AI AGENT] -->|Autonomous Work| G[GOAL]
    A -.-> B{Failure Modes}
    B --> C[Context Dilution]
    B --> D[Tool Saturation]
    B --> E[Lost in the Middle]
    
    classDef agent fill:#f9ebf9,stroke:#9b59b6,stroke-width:2px
    class A agent
```

### Introducing Hierarchical AI Agents

So instead of one agent trying to be the planner, and the implementer, and the quality assurance all at once, there is a trend for agentic workflows to adopt hierarchical AI agents. 

Now most hierarchical structures have two or three types of agent. And at the top here, this is considered to be the **high-level agent**. This guy is the top dog. So there's just probably one agent up here, and it can process strategic plans, it can perform task decomposition, it can manage processes. It's all the big picture stuff. 

And then below that, we have the middle layer. These are additional agents, and these are called **mid-level agents**, and they report into the high-level agents. So they receive directives from the high-level agent and then they process them. So they do things like, well they have their own tasks, like implement plans and further decompose tasks and coordinate teams of low-level agents. 

Yeah, there's another layer here, **low-level agents**. So let's draw some low-level agents in here. And I think you can guess what they are going to do. They report into the mid-level agent. And these guys are the doers, and they're specialized for narrow tasks. Perhaps they're trained on certain data, or they're given access to particular tools. So these low-level agents, they receive their directives from above, from the tier above here, and then they report back results as they go. Mid and high-level agents use the results from the low-level agents to inform the next steps of the process. 

And I suspect that this hierarchy looks rather familiar to you, because it probably mirrors how things are organized at your work. So we might have, at the top here, **executives**. And these are kind of analogous to the high-level agent, because they're setting the strategy. And then the strategy is decomposed into multiple projects that's managed by product teams or perhaps it's managed by a team of **PMs, project managers**. They're the mid-level agents. And then the actual work is performed by a series of **specialists** at the low-level agent level. They're reporting up through the hierarchy. You know, I've never really considered myself analogous to a low-level agent until right about now.

```mermaid
graph TD
    classDef highLevel fill:#eaf2f8,stroke:#2980b9,stroke-width:2px,color:#333;
    classDef midLevel fill:#e9f7ef,stroke:#27ae60,stroke-width:2px,color:#333;
    classDef lowLevel fill:#fdf2e9,stroke:#e67e22,stroke-width:2px,color:#333;

    High["<b>AGENT</b><br>High-Level Agent / Executive<br><i>(Strategy & Task Decomposition)</i>"]:::highLevel
    
    Mid1["<b>A</b><br>Mid-Level Agent / PM<br><i>(Implement Plans & Coordinate)</i>"]:::midLevel
    Mid2["<b>A</b><br>Mid-Level Agent / PM<br><i>(Implement Plans & Coordinate)</i>"]:::midLevel
    
    Low1["<b>A</b><br>Low-Level Agent<br><i>(Specialist / Doer)</i>"]:::lowLevel
    Low2["<b>A</b><br>Low-Level Agent<br><i>(Specialist / Doer)</i>"]:::lowLevel
    Low3["<b>A</b><br>Low-Level Agent<br><i>(Specialist / Doer)</i>"]:::lowLevel
    Low4["<b>A</b><br>Low-Level Agent<br><i>(Specialist / Doer)</i>"]:::lowLevel
    Low5["<b>A</b><br>Low-Level Agent<br><i>(Specialist / Doer)</i>"]:::lowLevel

    High -->|Directives| Mid1
    High -->|Directives| Mid2
    
    Mid1 -->|Directives| Low1
    Mid1 -->|Directives| Low2
    
    Mid2 -->|Directives| Low3
    Mid2 -->|Directives| Low4
    Mid2 -->|Directives| Low5
    
    Low1 -.->|Results| Mid1
    Low2 -.->|Results| Mid1
    Low3 -.->|Results| Mid2
    Low4 -.->|Results| Mid2
    Low5 -.->|Results| Mid2
    
    Mid1 -.->|Results| High
    Mid2 -.->|Results| High
```

### The Benefits of Hierarchical Agents

Now, this structure works because it's well, not exactly new thinking here. It applies a classic software engineering principle to the world of AI: **separation of concerns**. Now in a monolithic agent, the model is constantly context switching between high-level reasoning, like 'what shall I do next?', and then low-level execution, like 'which tool should I use to execute this task?'. But by kind of separating the work across a hierarchy, it overcomes a lot of limitations. 

And that includes limitations like solving context dilution. So instead of the entire conversation history being dumped into every prompt, a hierarchical system uses **contextual packets**. Which is to say, the high-level agent maintains the global state. But when it delegates a task to a lower-level agent, it only sends a kind of pruned, relevant slice of that context. So if this agent's job is let's say just to format a JSON file, well it probably doesn't need to know the initial 4,000-word strategy document. And that keeps the signal-to-noise ratio high, and it prevents the model from getting lost in the middle. 

And this hierarchy helps with tool saturation as well. So in IT, we generally follow the principle of least privilege, and this agentic hierarchy lets us do the same thing for AI through **tool specialization**. Low-level agents, these guys at the bottom, they only have access to specific tools. So if this was the security agent, it gets access to the vulnerability scanner tool, but it doesn't get access to the CI/CD pipeline tool. That's only for the devops agent, for this guy. So by limiting each agent to just a handful of tools, the agent isn't having to guess which tool to use. It's selecting from a narrow, purpose-built toolbox. 

Now another thing about the single agent architecture is you usually need to use the most expensive, fancy, high-reasoning model you can get your hands on, because some tasks are gonna need it. But for simpler tasks, incurring the inference costs of a big old frontier model can be a bit of a waste of compute. So in a hierarchy, you have a bit of **model flexibility**. You can fit different models to different tasks. So maybe the heavyweight frontier model, that's used as the high-level agent at the top here for all the complex planning and the task decomposition that this guy needs to do. But for some of the lower-level agents running more modular, self-contained tasks, they can run a much lighter weight model. 

And there's a bunch of other advantages as well. This is all very **modular**, so each agent can be tested and updated and swapped out without really touching the rest of the system. It allows for **parallelism**, which is to say we can have multiple agents working on different parts of the problem all at the same time. And it also provides **recursive feedback**, a recursive feedback loop. So when a low-level agent finishes a task, it reports back to the mid-level or the high-level agent, and that can kind of act as a bit of a quality gate where the supervisor can monitor output and trigger a retry or just kind of pivot if the result appears to be an error.

### The Limitations of Hierarchical Agents

So this all sounds pretty good, but hierarchical AI agents do have some real limitations to be aware of as well. 

And the first one of those is that **task decomposition is hard**. I've performed the project manager role enough myself to know that. So the entire system here, it hinges on the high-level agent's ability to break what is quite likely a pretty complex goal into the right subtasks and then to route them through to the right specialists. And if it decomposes poorly, so maybe it misses a step or maybe it sequences things in the wrong order, well then everything downstream is gonna inherit that mistake. It's garbage in, garbage out, and that garbage is flowing through three layers of agents. So the high-level agent needs to be good at planning, and current LLMs are inconsistent at planning. They can sometimes miss dependencies or they can underestimate complexity, or the thing I see the most is that they over-decompose simple tasks into unnecessary steps. 

So that's one thing to be concerned about. There's also **orchestration overhead**. So this is stuff like designing the state management and then defining the hand-off logic between the agents, building retry loops when things go wrong. Unlike a single agent, this requires architecting an entire system. And if the logic that governs how agents talk to each other is just a bit brittle, then the whole system can fall into a rather unfortunate recursive loop. That's where agents just kind of keep passing errors back and forth between each other until they hit their token limit. 

Now there's also the potential for the good old **telephone game** effect. Like how at work a manager issues an instruction, that gets filtered through a couple of colleagues, and then the person who's actually doing the work gets told something subtly but meaningfully different from what the manager was asking for in the first place. Well, the same thing can happen with agents. If task decomposition is slightly off or if the wrong bit of context gets pruned when sending the data packet all the way down the line here, well the specialized agent can end up perfectly executing the wrong task. 

So yes, hierarchical AI agents can keep your agent from getting lost in the middle, but they can still get lost in the org chart. So the trick is to treat the hierarchy like any other system you'd put into production. You need to design the handoffs, you need to validate the work, and just like in real life, never assume that the top dog always wrote a perfect plan.

---

### Summary of System Characteristics 

```mermaid
mindmap
  root((Hierarchical<br>AI Agents))
    Benefits
      Separation of Concerns
      Contextual Packets
      Tool Specialization
      Model Flexibility
      Modular
      Parallelism
      Recursive Feedback Loop
    Limitations
      Task Decomposition is Hard
      Orchestration Overhead
      Telephone Game
```
