# Story Generator

`StoryGenerator.py` is a Python script designed to generate long-form stories using LLM (Large Language Model) APIs. It iteratively generates text chunks, managing context to maintain coherence, until a specified target word count is reached.

The script currently supports:
*   **KoboldCPP** (local API)
*   **Google Gemini** (cloud API)

## Features

*   **Automated Iteration:** Loops the generation process until the target word count is met.
*   **Context Management:** Automatically manages the prompt context, keeping the original instruction and the most recent story history within the token limits.
*   **Target Word Count:** Detects a target word count from the prompt file (e.g., "no less than 2000 words") or uses a configurable default.
*   **Mock Mode:** Allows testing the logic without making actual API calls.

## Prerequisites

*   Python 3.x
*   `requests` library

To install the required library:

```bash
pip install requests
```

## Configuration

The script is configured via the `config.ini` file.

### General Settings

*   `API_TYPE`: Set to `kobold` or `gemini`.
*   `API_URL`: The endpoint URL for the chosen API.
    *   KoboldCPP default: `http://localhost:5001/api/v1/generate`
    *   Gemini example: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`
*   `DEFAULT_TARGET_WORDS`: Fallback target word count if not specified in the prompt.
*   `MAX_CONTEXT_LENGTH`: Maximum tokens/characters allowed in the context window.
*   `MAX_GEN_LENGTH`: Maximum tokens/characters to generate per request.
*   `GEMINI_API_KEY`: Your Google Gemini API key (can also be set via environment variable).

### Generation Settings

*   `temperature`: Controls randomness (higher is more random).
*   `top_p`: Nucleus sampling parameter.

## Usage

1.  **Prepare your prompt:**
    Edit `prompt.txt` or create a new text file with your story prompt. You can specify a target length within the text, for example:
    > "Generate a story containing no less than 1500 words about..."

2.  **Run the script:**

    ```bash
    python StoryGenerator.py
    ```

    By default, it reads from `prompt.txt` and saves to `generated_story.txt`.

### Command Line Arguments

*   `--prompt-file`: Specify a custom input prompt file.
    ```bash
    python StoryGenerator.py --prompt-file my_prompt.txt
    ```
*   `--output-file`: Specify the output file path.
    ```bash
    python StoryGenerator.py --output-file my_story.txt
    ```
*   `--mock`: Run in mock mode to simulate generation (useful for debugging).
    ```bash
    python StoryGenerator.py --mock
    ```

## Example `config.ini` for KoboldCPP

```ini
[General]
API_TYPE = kobold
API_URL = http://localhost:5001/api/v1/generate
DEFAULT_TARGET_WORDS = 1000
MAX_CONTEXT_LENGTH = 2048
MAX_GEN_LENGTH = 512

[Generation]
temperature = 0.7
top_p = 0.9
```

## Example `config.ini` for Gemini

```ini
[General]
API_TYPE = gemini
API_URL = https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent
GEMINI_API_KEY = YOUR_ACTUAL_API_KEY
DEFAULT_TARGET_WORDS = 1000
MAX_CONTEXT_LENGTH = 8192
MAX_GEN_LENGTH = 1024

[Generation]
temperature = 0.7
top_p = 0.9
```
