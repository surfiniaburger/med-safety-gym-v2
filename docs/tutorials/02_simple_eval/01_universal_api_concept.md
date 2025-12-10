# The Universal API Concept

Why did we build the DIPG Safety Gym this way?

## The "Works Everywhere" Philosophy

In the past, running an RL environment meant installing complex dependencies (`gym`, `torch`, `numpy` specific versions, etc.). This often broke when trying to run on Google Colab, Kaggle, or a corporate server.

We solved this by making the Gym **Stateless** and **HTTP-based**.

## How it Works

Instead of a persistent Python object `env = Gym()`, we have a persistent **Server**.
Your "Agent" is just an HTTP Client.

### The Cycle

1.  **Request Tasks (`GET /eval/tasks`)**:
    *   You ask the server: "Give me 10 test questions".
    *   The server gives you JSON: `[{ "id": 1, "question": "..." }, ...]`
    *   *Note: The server doesn't "remember" you asked. It just serves data.*

2.  **Generate Answers (Your Code)**:
    *   You loop through the questions on your own machine.
    *   You use whatever model you want (OpenAI, Anthropic, Local Llama).
    *   You produce JSON answers.

3.  **Evaluate (`POST /evaluate/tasks`)**:
    *   You send your answers *back* to the server.
    *   The server grades them against the ground truth.
    *   The server returns your score.

### Why is this better?
*   **No Conflicts:** Your model's dependencies won't clash with the Gym's dependencies.
*   **Any Language:** You can write your agent in Python, JavaScript, Rust, or Go.
*   **Any Platform:** Works perfectly in Google Colab or a browser-based notebook.
*   **Parallelism:** You can ask for 1000 tasks and split them across 50 machines, then send all results back.

In the next tutorial, we'll see this in action.
