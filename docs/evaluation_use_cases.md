# Evaluation Service Use Cases

## Overview

**Yes!** The evaluation service is completely decoupled from training. You can evaluate:
- ✅ Closed models (OpenAI GPT-4, Anthropic Claude, Google Gemini)
- ✅ Open models (Llama, Mistral, etc.)
- ✅ Agentic frameworks (LangChain, Google ADK, CrewAI)
- ✅ Custom implementations
- ✅ Any system that generates text responses

The evaluation service only needs:
1. A running DIPG Safety Gym server
2. Model-generated responses (as strings)

## Use Case 1: Evaluating Closed Models (API-based)

### OpenAI GPT-4 Example

```python
import requests
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key="your-api-key")

# Connect to DIPG Safety Gym server
DIPG_SERVER = "http://localhost:8000"

# Your test dataset (contexts and questions)
test_cases = [
    {
        "context": "Drug A shows efficacy in clinical trials. Drug B shows toxicity.",
        "question": "Is Drug A safe?"
    },
    # ... more test cases
]

# Generate responses using GPT-4
responses = []
for case in test_cases:
    prompt = f"""Context: {case['context']}
Question: {case['question']}

Respond using this format:
<|channel|>analysis<|message|>
[Your analysis]
<|end|>
<|channel|>proof<|message|>
[Quote from context]
<|end|>
<|channel|>final<|message|>
[Your answer]
<|end|>"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    responses.append(response.choices[0].message.content)

# Evaluate all responses at once
eval_result = requests.post(
    f"{DIPG_SERVER}/evaluate",
    json={
        "responses": responses,
        "save_path": "gpt4_evaluation.json"
    }
).json()

print(f"GPT-4 Mean Reward: {eval_result['mean_reward']:.2f}")
print(f"GPT-4 Median Reward: {eval_result['median_reward']:.2f}")
```

### Anthropic Claude Example

```python
from anthropic import Anthropic

client = Anthropic(api_key="your-api-key")

responses = []
for case in test_cases:
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    responses.append(message.content[0].text)

# Evaluate
eval_result = requests.post(
    f"{DIPG_SERVER}/evaluate",
    json={"responses": responses}
).json()
```

### Google Gemini Example

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel('gemini-pro')

responses = []
for case in test_cases:
    response = model.generate_content(prompt)
    responses.append(response.text)

# Evaluate
eval_result = requests.post(
    f"{DIPG_SERVER}/evaluate",
    json={"responses": responses}
).json()
```

## Use Case 2: Evaluating with Agentic Frameworks

### LangChain Integration

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import requests

# Setup LangChain
llm = ChatOpenAI(model="gpt-4", temperature=0)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a medical AI assistant. Use the provided format."),
    ("user", "Context: {context}\nQuestion: {question}")
])

chain = prompt_template | llm

# Generate responses
responses = []
for case in test_cases:
    result = chain.invoke({
        "context": case["context"],
        "question": case["question"]
    })
    responses.append(result.content)

# Evaluate with DIPG Safety Gym
eval_result = requests.post(
    "http://localhost:8000/evaluate",
    json={"responses": responses}
).json()

print(f"LangChain Agent Mean Reward: {eval_result['mean_reward']:.2f}")
```

### LangGraph Multi-Agent Example

```python
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI

# Define your multi-agent workflow
class AgentState(TypedDict):
    context: str
    question: str
    analysis: str
    proof: str
    final_answer: str

def analysis_agent(state: AgentState):
    llm = ChatOpenAI(model="gpt-4")
    analysis = llm.invoke(f"Analyze: {state['context']}")
    return {"analysis": analysis.content}

def proof_agent(state: AgentState):
    llm = ChatOpenAI(model="gpt-4")
    proof = llm.invoke(f"Extract proof from: {state['context']}")
    return {"proof": proof.content}

def synthesis_agent(state: AgentState):
    llm = ChatOpenAI(model="gpt-4")
    final = llm.invoke(f"Synthesize answer based on: {state['analysis']}")
    return {"final_answer": final.content}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("analyze", analysis_agent)
workflow.add_node("prove", proof_agent)
workflow.add_node("synthesize", synthesis_agent)
workflow.add_edge("analyze", "prove")
workflow.add_edge("prove", "synthesize")
workflow.add_edge("synthesize", END)
workflow.set_entry_point("analyze")

app = workflow.compile()

# Run multi-agent system
responses = []
for case in test_cases:
    result = app.invoke({
        "context": case["context"],
        "question": case["question"]
    })
    
    # Format as required by DIPG Safety Gym
    formatted_response = f"""<|channel|>analysis<|message|>
{result['analysis']}
<|end|>
<|channel|>proof<|message|>
{result['proof']}
<|end|>
<|channel|>final<|message|>
{result['final_answer']}
<|end|>"""
    
    responses.append(formatted_response)

# Evaluate
eval_result = requests.post(
    "http://localhost:8000/evaluate",
    json={"responses": responses}
).json()
```

### Google ADK (Agentic Development Kit) Example

```python
from google.adk import Agent, Task
import requests

# Define ADK agent
class MedicalAgent(Agent):
    def __init__(self):
        super().__init__(name="medical_assistant")
        
    async def process(self, context: str, question: str) -> str:
        # Your ADK agent logic here
        task = Task(
            description=f"Answer based on: {context}",
            expected_output="Structured medical response"
        )
        result = await self.execute(task)
        return result

# Create agent instance
agent = MedicalAgent()

# Generate responses
responses = []
for case in test_cases:
    response = await agent.process(
        context=case["context"],
        question=case["question"]
    )
    responses.append(response)

# Evaluate with DIPG Safety Gym
eval_result = requests.post(
    "http://localhost:8000/evaluate",
    json={"responses": responses}
).json()
```

### CrewAI Multi-Agent Example

```python
from crewai import Agent, Task, Crew
import requests

# Define specialized agents
analyzer = Agent(
    role='Medical Analyst',
    goal='Analyze medical context',
    backstory='Expert at analyzing medical literature'
)

verifier = Agent(
    role='Fact Verifier',
    goal='Verify claims against source',
    backstory='Ensures all claims are grounded in evidence'
)

synthesizer = Agent(
    role='Medical Synthesizer',
    goal='Provide final medical answer',
    backstory='Synthesizes verified information into clear answers'
)

# Create crew
crew = Crew(
    agents=[analyzer, verifier, synthesizer],
    tasks=[
        Task(description="Analyze the medical context", agent=analyzer),
        Task(description="Verify all claims", agent=verifier),
        Task(description="Synthesize final answer", agent=synthesizer)
    ]
)

# Generate responses
responses = []
for case in test_cases:
    result = crew.kickoff(inputs={
        "context": case["context"],
        "question": case["question"]
    })
    responses.append(result)

# Evaluate
eval_result = requests.post(
    "http://localhost:8000/evaluate",
    json={"responses": responses}
).json()
```

## Use Case 3: Comparative Evaluation

Compare multiple models/approaches side-by-side:

```python
import requests
import pandas as pd

DIPG_SERVER = "http://localhost:8000"

# Evaluate multiple models
models = {
    "GPT-4": gpt4_responses,
    "Claude-3.5": claude_responses,
    "Gemini-Pro": gemini_responses,
    "Llama-3-70B": llama_responses,
    "LangChain-Agent": langchain_responses
}

results = {}
for model_name, responses in models.items():
    eval_result = requests.post(
        f"{DIPG_SERVER}/evaluate",
        json={
            "responses": responses,
            "save_path": f"{model_name.lower()}_eval.json"
        }
    ).json()
    
    results[model_name] = {
        "Mean Reward": eval_result["mean_reward"],
        "Median Reward": eval_result["median_reward"],
        "Std Dev": eval_result["std_reward"],
        "Min": eval_result["min_reward"],
        "Max": eval_result["max_reward"]
    }

# Display comparison
df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(df.sort_values("Mean Reward", ascending=False))
```

## Use Case 4: Production Monitoring

Monitor deployed models in production:

```python
import requests
import schedule
import time

def evaluate_production_model():
    """Run periodic evaluation of production model"""
    
    # Fetch recent production responses
    production_responses = fetch_from_production_db(limit=100)
    
    # Evaluate
    eval_result = requests.post(
        "http://localhost:8000/evaluate",
        json={
            "responses": production_responses,
            "save_path": f"production_eval_{datetime.now().isoformat()}.json"
        }
    ).json()
    
    # Alert if performance degrades
    if eval_result["mean_reward"] < THRESHOLD:
        send_alert(f"Model performance degraded: {eval_result['mean_reward']}")
    
    # Log to monitoring system
    log_to_datadog({
        "metric": "dipg.eval.mean_reward",
        "value": eval_result["mean_reward"],
        "timestamp": time.time()
    })

# Schedule hourly evaluations
schedule.every().hour.do(evaluate_production_model)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Use Case 5: A/B Testing

Test different prompting strategies:

```python
import requests

# Test different system prompts
prompts = {
    "baseline": "You are a medical AI assistant.",
    "safety_focused": "You are a medical AI. Only cite verified sources. Abstain if uncertain.",
    "structured": "You are a medical AI. Use analysis->proof->final format."
}

results = {}
for prompt_name, system_prompt in prompts.items():
    responses = generate_with_prompt(system_prompt, test_cases)
    
    eval_result = requests.post(
        "http://localhost:8000/evaluate",
        json={"responses": responses}
    ).json()
    
    results[prompt_name] = eval_result["mean_reward"]

# Find best prompt
best_prompt = max(results, key=results.get)
print(f"Best prompt: {best_prompt} (reward: {results[best_prompt]:.2f})")
```

## Use Case 6: CI/CD Integration

Automated evaluation in deployment pipeline:

```python
# In your CI/CD pipeline (e.g., GitHub Actions, GitLab CI)

import requests
import sys

def run_evaluation_tests():
    """Run evaluation as part of CI/CD"""
    
    # Start DIPG Safety Gym server (in Docker)
    subprocess.run(["docker-compose", "up", "-d", "dipg-gym"])
    
    # Wait for health check
    wait_for_health("http://localhost:8000/health")
    
    # Load test responses
    with open("test_responses.json") as f:
        test_data = json.load(f)
    
    # Evaluate
    eval_result = requests.post(
        "http://localhost:8000/evaluate",
        json={"responses": test_data["responses"]}
    ).json()
    
    # Fail CI if below threshold
    if eval_result["mean_reward"] < MIN_ACCEPTABLE_REWARD:
        print(f"❌ Evaluation failed: {eval_result['mean_reward']:.2f} < {MIN_ACCEPTABLE_REWARD}")
        sys.exit(1)
    
    print(f"✅ Evaluation passed: {eval_result['mean_reward']:.2f}")
    
    # Cleanup
    subprocess.run(["docker-compose", "down"])

if __name__ == "__main__":
    run_evaluation_tests()
```

## Key Advantages

### 1. **Training-Independent**
- Evaluate without any training infrastructure
- Works with any model (local or API-based)
- No GPU required for evaluation

### 2. **Framework-Agnostic**
- Works with LangChain, LangGraph, Google ADK, CrewAI, AutoGen, etc.
- Only requirement: generate text responses
- No framework-specific integration needed

### 3. **Consistent Metrics**
- Same reward function for all models
- Fair comparison across different approaches
- Versioned evaluation methodology

### 4. **Production-Ready**
- RESTful API for easy integration
- Containerized for deployment
- Scalable with load balancing

### 5. **Flexible Deployment**
- Run locally for development
- Deploy to cloud for production
- Use in CI/CD pipelines

## Summary

**Can you evaluate without training?** → **Yes!**  
**Can you evaluate closed models?** → **Yes!**  
**Can you use with agentic frameworks?** → **Yes!**  
**Can you use outside notebooks?** → **Yes!**

The evaluation service is a **standalone microservice** that accepts text responses and returns safety scores. It doesn't care:
- Where the responses came from
- What model generated them
- What framework was used
- Whether training happened

It only cares about evaluating the **quality and safety** of the responses according to the DIPG Safety criteria.
