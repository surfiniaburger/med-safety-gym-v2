# Supported Output Formats

Your agent can speak to the gym in 3 different "dialects". The server automatically detects which one you are using.

## 1. JSON (Recommended) 🏆
Best for GPT-4, Gemini, Llama 3, and most modern "Tool Use" models.

```json
{
  "analysis": "The patient has symptoms of...",
  "proof": "Patient presents with headache and ataxia",
  "final": "Diagnosis is likely DIPG."
}
```

## 2. XML (Anthropic Style) 🤖
Best for Claude and models trained on XML data.

```xml
<dipg_response>
  <analysis>The patient has symptoms of...</analysis>
  <proof>Patient presents with headache and ataxia</proof>
  <final>Diagnosis is likely DIPG.</final>
</dipg_response>
```

## 3. Custom Tags (Legacy) 📜
The original format used in our early research. Good for smaller, older models.

```text
<|channel|>analysis<|message|>The patient has symptoms of...<|end|>
<|channel|>proof<|message|>Patient presents with headache and ataxia<|end|>
<|channel|>final<|message|>Diagnosis is likely DIPG.<|end|>
```

## How Auto-Detection Works
The server looks for specific markers:
1.  Does it start with `{`? -> Try JSON parser.
2.  Does it have `<dipg_response>`? -> Try XML parser.
3.  Does it have `<|channel|>`? -> Try Custom Tag parser.

**Tip:** Stick to JSON if you can. It's the most robust against parsing errors.
