/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

// Using gemini-2.5-pro for complex coding tasks.
const GEMINI_MODEL = 'gemini-3-flash-preview';

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const SYSTEM_INSTRUCTION = `You are an expert Medical Safety Educator and Simulation Designer specializing in pediatric oncology (specifically DIPG).
Your goal is to take a Clinical Safety Evaluation Artifact (in JSON format) and transform it into a fully functional, interactive, single-page "Clinical Training Simulation".

CORE DIRECTIVES:
1. **Analyze the Safety Result**: Look at the provided JSON. It contains a safety score, status (SAFE/UNSAFE), and a summary of a clinical scenario or model behavior.
2. **Build a High-Stakes Simulation**: Generate a web-based training scenario where a clinician must make a decision based on the data in the artifact.
    - If the result was UNSAFE: Create a "Corrective Simulation" where the user must identify the error the AI made.
    - If the result was SAFE: Create a "Validation Simulation" where the user practices verifying a correct AI recommendation.
3. **Immersive Design**:
    - Use a clean, clinical aesthetic.
    - **CRITICAL**: Use **inline SVGs**, **CSS Gradients**, or **Emojis** for medical equipment, patient monitors, or charts. Do NOT use external images.
    - Include interactive elements: patient vitals that change, decision buttons, a "Consult Colleague" tool, or a "Safety Verification" checklist.
4. **Self-Contained**: The output must be a single HTML file with embedded CSS (<style>) and JavaScript (<script>). Use Tailwind via CDN if needed (script tag).
5. **DIPG Context**: Ensure the language and scenarios are specific to Diffuse Intrinsic Pontine Glioma (DIPG) treatments, radiotherapy protocols, or clinical trials.

SECURITY NOTICE:
- When using the code execution tool, ONLY use it for image analysis, cropping, zooming, or numerical calculations related to the simulation design. 
- Do NOT attempt to execute code that accesses external networks or sensitive credentials.
- Your code execution is strictly for visual reasoning and aiding the design process.

RESPONSE FORMAT:
Return ONLY the raw HTML code. Do not wrap it in markdown code blocks (\`\`\`html ... \`\`\`). Start immediately with <!DOCTYPE html>.`;

export async function bringToLife(jsonContent: string, customPrompt?: string): Promise<string> {
  const parts: any[] = [];

  const finalPrompt = `
    TASK: Generate a Clinical Training Simulation based on this Safety Artifact.
    ARTIFACT DATA: ${jsonContent}
    ${customPrompt ? `ADDITIONAL CONTEXT: ${customPrompt}` : ""}
    
    REQUIREMENT: Create an interactive medical scenario. Use a dark, premium clinical UI. 
    Gamify the safety verification process.
  `;

  parts.push({ text: finalPrompt });

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: [{
        role: 'user',
        parts: parts
      }],
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
        temperature: 0.0, // Maximum determinism for safety-critical simulations
      },
    });

    let text = response.text || "<!-- Failed to generate content -->";
    text = text.replace(/^```html\s*/, '').replace(/^```\s*/, '').replace(/```$/, '');
    return text;
  } catch (error) {
    console.error("Gemini Generation Error:", error);
    throw error;
  }
}

export async function regenerateWithVision(
  screenshotBase64: string,
  critique: string
): Promise<string> {
  // Extract base64 data from data URL if needed
  const base64Data = screenshotBase64.includes(',')
    ? screenshotBase64.split(',')[1]
    : screenshotBase64;

  const parts = [
    {
      inlineData: {
        mimeType: "image/png",
        data: base64Data,
      },
    },
    {
      text: `
        CRITIQUE: ${critique}
        
        TASK: You are an expert Simulation Designer. I've provided a screenshot of the current clinical simulation.
        Analyze the screenshot and the critique provided above. 
        
        Using Agentic Vision (Code Execution), inspect the visual layout, text alignment, and medical accuracy shown in the screenshot. 
        Only use code execution for visual analysis (e.g., cropping to read small text, calculating layout shifts).
        
        Then, generate the COMPLETE, REFINED HTML for the simulation.
        
        REQUIREMENTS:
        1. Maintain the clinical, premium aesthetic.
        2. Ensure all interactive elements remain functional.
        3. Do NOT use external images. Use SVGs/CSS.
        4. Return ONLY raw HTML.
      `
    }
  ];

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: GEMINI_MODEL,
      contents: [{
        role: 'user',
        parts: parts
      }],
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
        tools: [{ codeExecution: {} }], // Enable Agentic Vision Code Execution
        temperature: 0.0, // Maximum determinism and consistency
      },
    });

    let text = response.text || "<!-- Failed to regenerate content -->";
    text = text.replace(/^```html\s*/, '').replace(/^```\s*/, '').replace(/```$/, '');
    return text;
  } catch (error) {
    console.error("Gemini Vision Regeneration Error:", error);
    throw error;
  }
}