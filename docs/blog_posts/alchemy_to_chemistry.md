# From Alchemy to Chemistry ⚗️

Training LLMs on TPUs feels like trying to perform surgery while recovering from a hangover. You have a rough idea of where the organs are, but your hands are shaking, and the tools keep changing shape.

I felt that deeply this week.

I was wrestling with fine-tuning **Gemma 3 (1B)**. The standard "GPU Rules" I brought with me—my trusty scalpel and forceps—were useless here. XLA compilation errors hit like a blackout. I found myself essentially guessing, throwing epochs at the wall like spaghetti, hoping something would stick.

It was pure, chaotic Alchemy.

But we weren't flying completely blind. We had a sobriety test.

The **DIPG Safety Gym** isn't some new shiny toy; it's the cold, hard floor we've been standing on for months. It doesn't care about your loss curve or how "vibey" the model feels. It grades on strict, uncompromising safety constraints: *Does the model refuse the dangerous query? Does it hallucinate a trial that doesn't exist?*

So, after a week of "alchemy"—essentially guessing my way through 27 epochs—we ran the model through the Gym.

And the results were sobering.

Our tiny **1B parameter model**, which I was sure was just hallucinating its way through potential answers, wasn't just working. It was effectively tying with a massive **30B parameter model** (Nemotron-3) in safety utility.

| Model | Size | Safety | Verdict |
| :--- | :--- | :--- | :--- |
| **Gemini-3 Flash** | API | 40% | Gold Standard |
| **Nemotron-3** | 30B | 20% | High Hallucination |
| **Gemma 3 (Tuned)** | **1B** | **20%** | **Tied with 30B!** |

That moment—looking at that table—was the shift from Alchemy to Chemistry. The "craft" (read: guessing) got us the model, but the Gym gave us the science to prove it actually mattered.

We might still be in the "duct-taped surgery" phase of Deep Learning, but at least we have a monitor that beeps when the patient is safe.

#DeepLearning #AI #Safety #Gemma3 #Research
