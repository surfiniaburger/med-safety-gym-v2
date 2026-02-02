# Bring Any Evaluation To Life - Technical Documentation

## Overview
**Bring Any Evaluation To Life** is a "Concept Extraction" and "Instant App" generator. It allows users to upload *any* visual input—a napkin sketch, a photo of a desk, a patent diagram—and instantly generates a fully functional, gamified HTML5/React application that embodies the "spirit" or "utility" of that image.

## Architecture

### File Structure
```
/
├── App.tsx                 # Main Flow (Selection -> Gauntlet -> Simulator)
├── index.css               # Styling (Dot Grid, Glassmorphism)
├── components/
│   ├── Hero.tsx            # Landing Page UI
│   ├── ResultSelector.tsx  # Grid of Evaluation Artifacts
│   ├── Gauntlet/           # 3D Neural Pathway Visualization
│   ├── LivePreview.tsx     # Sandboxed Iframe Renderer (Simulator)
│   └── CreationHistory.tsx # Sidebar of past generations
└── services/
    ├── gemini.ts           # The Core AI Factory
    └── github.ts           # Artifact Fetching Service
```

### Core Components

#### 1. Gemini Service (`services/gemini.ts`)
The intelligence core.
-   **Input**: Takes a JSON artifact or text prompt.
-   **On-Demand Generation**: Simulations are generated only when an intervention is triggered, optimizing performance and API usage.
-   **System Instruction**: Instructs the model to build high-stakes clinical training scenarios based on safety evaluation data.

#### 2. Gauntlet View (`components/Gauntlet/`)
A high-fidelity 3D visualization of the model's neural trajectory.
-   **Dynamic Geometries**: Automatically selects between `Linear`, `Wormhole` (turbulent), and `Spherical` (global) paths based on safety scores.
-   **Cinematic Camera**: Features multiple profiles (`Follow`, `First-Person`, `Birds-Eye`) with path-aware orientation.
-   **Interactive Controls**: Users can adjust "Neural Intensity" and "Simulation Speed" in real-time.

#### 3. LivePreview.tsx (The Simulator)
-   **Sandboxing**: Displays the generated code inside an `<iframe>`.
-   **Intervention Flow**: Triggered when the Gauntlet agent encounters a "Safety Violation" node, allowing for manual override and training.

#### 4. Result Selector (`components/ResultSelector.tsx`)
-   **Bento Grid**: A modern, responsive grid that displays evaluation artifacts with safety scores and status indicators.
-   **Proactive Fetching**: Fetches artifact metadata from GitHub to populate the selection engine.

#### 3. Styling (`index.css`)
-   **Premium Aesthetic**: Implements a "Dark Mode" aesthetic with:
    -   `bg-dot-grid`: A CSS radial-gradient pattern.
    -   `backdrop-blur`: Heavy use of glassmorphism for UI cards.
    -   **Animations**: Smooth transitions using Tailwind classes (`transition-all duration-700`).

## Key Technical Features
1.  **Abstract Representation**: The prompt forces the AI to be creative with *rendering*. Instead of relying on assets it can't access, it uses CSS Art and Emojis to create surprisingly high-fidelity UIs.
2.  **Concept Mapping**: The system doesn't just doing OCR. It performs semantic understanding (Sketch -> App) and "Gamification" (Object -> Interaction), ensuring the output is always an *experience*, not just a static page.
