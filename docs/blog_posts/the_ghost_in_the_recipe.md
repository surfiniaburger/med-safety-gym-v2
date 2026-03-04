# The Ghost in the Recipe 👻🦀

Last year, we caught a ghost.

Our AI agent was running safety checks on its own medical responses. It would block a perfectly valid reply about *switching a chemotherapy drug* because it flagged "switching" as a suspicious entity.

The problem? The word **wasn't even in the original patient query.**

It was the agent reading its own ghost. A word it had written in a previous turn, injected back into its own context—silently turning its wisdom into a false alarm.

This is what I started calling **Output-Driven Hallucination**: when an agent doesn't lie about the world, but lies to itself, and then acts on that lie.

No one talks about this. But in a medical AI, it's the difference between a patient getting the right drug information and getting ghosted by a bot that has scared itself.

---

### What We Did About It

Most teams would patch it. Add the word to a "filler list." Grow the list forever.

We've done that. Our list hit 100+ terms. It doesn't scale. It's whack-a-mole with clinical vocabulary.

So this week, we took a different path: **we threw away the list entirely.**

Instead of asking "is this word noise?", we now ask: "is this word a clinical entity at all?"

We swapped a brittle regex heuristic for **GLiNER**—a zero-shot Named Entity Recognition model that reasons about whether a token is a *Medication*, *Symptom*, *Procedure*, or *Disease*—in full context.

The word "switching"? Not a medication. Ignored instantly.  
"Panobinostat"? Medication. Extracted. Checked. Every time.

No list. No maintenance. Just semantics.

---

### Why This Is Bigger Than a Bug Fix

This is what "Zero-Injection" actually means in practice.

When the safety brain is powered by raw string matching, **every word is a potential ghost**. You can't tell signal from noise because you've reduced meaning to characters.

When you upgrade to semantic extraction, the architecture becomes **self-aware about what matters**. It stops haunting itself.

This is the progression we've been on since we started the DIPG Safety Gym—from alchemy (guessing loss curves) to chemistry (measuring what is actually safe), and now to **architecture** (building systems that cannot, by design, fool themselves).

The Governor doesn't just govern the agent's tools.  
He now governs what the agent is *allowed to notice.*

---

We're still very much building. But watching a medical AI stop flagging its own answers as dangerous?

That felt like the ghost leaving the kitchen. 👻 → 🦀

**#AISafety #MedicalAI #NLP #GLiNER #ZeroTrust #SafeClaw #ClinicalNLP #AgentArchitecture #BuildingInPublic**
