import requests
import re
import sys
import argparse
import time
import configparser
import os
import json

# Load configuration
config = configparser.ConfigParser()
config_file = 'config.ini'

if not os.path.exists(config_file):
    print(f"Error: Configuration file {config_file} not found.")
    sys.exit(1)

config.read(config_file)

try:
    API_TYPE = config['General'].get('API_TYPE', 'kobold').lower()
    API_URL = config['General']['API_URL']
    DEFAULT_TARGET_WORDS = int(config['General']['DEFAULT_TARGET_WORDS'])
    MAX_CONTEXT_LENGTH = int(config['General']['MAX_CONTEXT_LENGTH'])
    MAX_GEN_LENGTH = int(config['General']['MAX_GEN_LENGTH'])
    GEMINI_API_KEY = config['General'].get('GEMINI_API_KEY', '')
    REFERENCE_STORY_PATH = config['General'].get('reference_story', '')
    VERBOSE = config['General'].get('verbose', 'no').lower() == 'yes'

    TEMPERATURE = float(config['Generation']['temperature'])
    TOP_P = float(config['Generation']['top_p'])
except (KeyError, ValueError) as e:
    print(f"Error reading configuration: {e}")
    sys.exit(1)

# Override with environment variable if present
if 'GEMINI_API_KEY' in os.environ:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']

# Initialize verbose log
if VERBOSE:
    try:
        with open("verbose.log", "w", encoding='utf-8') as f:
            f.write("--- Verbose Log Started ---\n\n")
    except IOError as e:
        print(f"Error initializing verbose log: {e}")

def log_verbose(title, data):
    if VERBOSE:
        try:
            with open("verbose.log", "a", encoding='utf-8') as f:
                f.write(f"--- {title} ---\n")
                try:
                    f.write(json.dumps(data, indent=2))
                except (TypeError, ValueError):
                    f.write(str(data))
                f.write("\n\n")
        except IOError as e:
            print(f"Error writing to verbose log: {e}")

def load_prompt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)

def get_word_count(text):
    return len(text.split())

def extract_target_length(prompt_text):
    # Look for "no less than X words"
    match = re.search(r"no less than (\d+) words", prompt_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return DEFAULT_TARGET_WORDS

def generate_chunk(prompt, mock=False):
    if mock:
        # Simulate generation
        time.sleep(0.5)
        return " The engineer looked at the strange device and wondered how he could use it to generate power for the village."

    if API_TYPE == 'kobold':
        return generate_chunk_kobold(prompt)
    elif API_TYPE == 'gemini':
        return generate_chunk_gemini(prompt)
    else:
        print(f"Error: Unknown API_TYPE '{API_TYPE}'")
        return None

def split_text_into_chunks(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def analyze_story(story_text, mock=False):
    """
    Analyzes the provided story text to extract storyline, plot, characters, and setting.
    Handles long texts by chunking.
    """

    # Calculate a safe chunk size.
    # MAX_CONTEXT_LENGTH is in tokens (approx), usually 2048.
    # We want to leave room for the prompt and the generated summary.
    # 2048 tokens * 3 chars/token = 6144 chars.
    # Let's target ~4000 chars per chunk to be safe and allow for overhead.
    CHUNK_SIZE = 4000

    if len(story_text) <= CHUNK_SIZE:
        # Short enough to analyze in one go
        analysis_prompt = (
            "Analyze the following story and provide a detailed plot outline, including key beats, "
            "character arcs, and setting details. Ensure the storyline is clear and easy to follow.\n\n"
            f"Story Text:\n{story_text}"
        )
        print("Analyzing reference story...")
        if mock:
             return "[Mock Analysis: The story is about characters in a setting doing things.]"

        if API_TYPE == 'kobold':
            return generate_chunk_kobold(analysis_prompt)
        elif API_TYPE == 'gemini':
            return generate_chunk_gemini(analysis_prompt)
        else:
             return "Analysis failed: Unknown API type."
    else:
        # Long story, needs chunking
        print("Reference story is long. Analyzing in chunks...")
        chunks = split_text_into_chunks(story_text, CHUNK_SIZE)
        summaries = []

        for i, chunk in enumerate(chunks):
            print(f"Analyzing chunk {i+1}/{len(chunks)}...")
            prompt = (
                "Create a detailed outline of the events in the following part of the story, focusing on plot progression, "
                "character actions, and key scenes:\n\n"
                f"{chunk}"
            )

            if mock:
                summary = f"[Mock Summary of chunk {i+1}]"
            elif API_TYPE == 'kobold':
                summary = generate_chunk_kobold(prompt)
            elif API_TYPE == 'gemini':
                summary = generate_chunk_gemini(prompt)
            else:
                summary = None

            if summary:
                summaries.append(summary)
            else:
                print(f"Warning: Failed to summarize chunk {i+1}")

        combined_summary = "\n".join(summaries)

        # Final analysis of the combined summaries
        print("Performing final analysis on combined summaries...")
        final_prompt = (
            "Analyze the following story outlines and provide a comprehensive and unified plot outline of the full story. "
            "Include key plot points, character development, and setting details in chronological order.\n\n"
            f"Story Parts:\n{combined_summary}"
        )

        if mock:
            return "[Mock Final Analysis: The full story involves characters A and B going through events X, Y, and Z.]"

        if API_TYPE == 'kobold':
            return generate_chunk_kobold(final_prompt)
        elif API_TYPE == 'gemini':
            return generate_chunk_gemini(final_prompt)
        else:
             return "Analysis failed: Unknown API type."

def generate_chunk_kobold(prompt):
    payload = {
        "prompt": prompt,
        "max_length": MAX_GEN_LENGTH,
        "max_context_length": MAX_CONTEXT_LENGTH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
    }

    if VERBOSE:
        log_verbose("Kobold Request Payload", payload)
    
    try:
        response = requests.post(API_URL, json=payload)

        if VERBOSE:
            try:
                log_verbose("Kobold Response", response.json())
            except:
                log_verbose("Kobold Response (Raw)", response.text)

        response.raise_for_status()
        result = response.json()

        if 'results' in result and len(result['results']) > 0:
            return result['results'][0]['text']
        else:
            print("Warning: Unexpected API response format from KoboldCPP.")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"Error calling Kobold API: {e}")
        return None

def generate_chunk_gemini(prompt):
    if not GEMINI_API_KEY or GEMINI_API_KEY == 'YOUR_API_KEY_HERE':
        print("Error: GEMINI_API_KEY is not set in config or environment.")
        return None

    # Append key to URL if not present
    url = API_URL
    if 'key=' not in url:
        if '?' in url:
            url += f"&key={GEMINI_API_KEY}"
        else:
            url += f"?key={GEMINI_API_KEY}"

    # Gemini Payload
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "topP": TOP_P,
            "maxOutputTokens": MAX_GEN_LENGTH
        }
    }

    if VERBOSE:
        log_verbose("Gemini Request Payload", payload)
    
    try:
        response = requests.post(url, json=payload)

        if VERBOSE:
            try:
                log_verbose("Gemini Response", response.json())
            except:
                log_verbose("Gemini Response (Raw)", response.text)

        response.raise_for_status()
        result = response.json()
        
        # Parse Gemini Response
        # {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    return parts[0]['text']
        
        print(f"Warning: Unexpected API response format from Gemini: {result}")
        return ""
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        if e.response is not None:
             print(f"Response: {e.response.text}")
        return None

def construct_prompt(original_instruction, story_so_far, reference_context=None):
    # Strategy:
    # 1. Main Instruction (Prompt)
    # 2. Reference Plot Outline (if exists)
    # 3. Guidance (if reference exists)
    # 4. Story Context (Story so far)
    
    # Base instruction block
    base_instruction = f"Instructions: {original_instruction}\n"
    
    if reference_context:
        base_instruction += f"\nReference Plot Outline:\n{reference_context}\n"
        base_instruction += "\nGuidance: Write the next part of the story, ensuring it follows the Reference Plot Outline above.\n"

    base_instruction += "\nStory so far:\n"

    # Calculate available space
    # We need to ensure that (Prompt + Generation) <= MAX_CONTEXT_LENGTH
    max_prompt_tokens = MAX_CONTEXT_LENGTH - MAX_GEN_LENGTH
    safe_char_limit = int(max_prompt_tokens * 3) # Approx 3 chars per token
    
    # Space occupied by static parts
    static_len = len(base_instruction)
    available_chars_for_story = safe_char_limit - static_len
    
    if available_chars_for_story < 50:
        # Static parts are too long.
        # Priority: Keep instructions and reference, sacrifice story context?
        # If static_len is effectively consuming the whole context, we must trim the reference.
        print("Warning: Context limit reached by instructions.")

        # Try to keep at least some story context (e.g. 100 chars)
        min_story_context = 100
        available_for_static = safe_char_limit - min_story_context

        if available_for_static < len(original_instruction) + 50:
             # Even original instruction is too long? Just fail gracefully to simple prompt.
             print("Critical: Instruction too long.")
             simple_instruction = f"Instructions: {original_instruction}\n\nStory so far:\n"

             if len(simple_instruction) > safe_char_limit:
                 # Instruction itself exceeds limit. Fallback to just story tail.
                 return story_so_far[-safe_char_limit:]

             available_simple = safe_char_limit - len(simple_instruction)
             return simple_instruction + story_so_far[-available_simple:]

        # Truncate reference to fit
        # We know static_len > available_for_static
        # Rebuild base_instruction with truncated reference
        excess = static_len - available_for_static
        truncated_reference_len = len(reference_context) - excess - 20 # -20 for safety

        if truncated_reference_len > 0:
             truncated_ref = reference_context[:truncated_reference_len] + "... [Truncated]"
             base_instruction = f"Instructions: {original_instruction}\n\nReference Plot Outline:\n{truncated_ref}\n\nGuidance: Write the next part of the story, ensuring it follows the Reference Plot Outline above.\n\nStory so far:\n"
             available_chars_for_story = min_story_context
        else:
             # Reference too long to keep even a bit? Drop it.
             base_instruction = f"Instructions: {original_instruction}\n\nStory so far:\n"
             available_chars_for_story = safe_char_limit - len(base_instruction)
    
    if len(story_so_far) > available_chars_for_story:
        context_story = story_so_far[-available_chars_for_story:]
    else:
        context_story = story_so_far

    full_prompt = base_instruction + context_story
    return full_prompt

def main():
    parser = argparse.ArgumentParser(description="Generate a story using KoboldCPP or Gemini.")
    parser.add_argument("--prompt-file", default="prompt.txt", help="Path to the prompt file.")
    parser.add_argument("--output-file", default="generated_story.txt", help="Path to save the generated story.")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no API calls).")
    args = parser.parse_args()

    print("--- Story Generator Started ---")
    print(f"API Provider: {API_TYPE}")
    
    original_prompt = load_prompt(args.prompt_file)
    target_words = extract_target_length(original_prompt)
    
    print(f"Goal: Generate a story of at least {target_words} words.")

    # Check for reference story
    reference_context = ""
    if REFERENCE_STORY_PATH and os.path.exists(REFERENCE_STORY_PATH):
        print(f"Reading reference story from: {REFERENCE_STORY_PATH}")
        try:
            with open(REFERENCE_STORY_PATH, 'r', encoding='utf-8') as f:
                ref_text = f.read()

            # Analyze reference story (now handles long texts)
            analysis = analyze_story(ref_text, mock=args.mock)

            if analysis:
                print("Reference story analyzed successfully.")
                reference_context = analysis
            else:
                print("Warning: Failed to analyze reference story.")

        except Exception as e:
            print(f"Error processing reference story: {e}")

    story = ""
    chunk_count = 0
    
    while get_word_count(story) < target_words:
        print(f"\nGenerating chunk {chunk_count + 1}...")
        
        # Construct prompt
        if not story:
            # First iteration
            current_input = f"Instructions: {original_prompt}\n"
            if reference_context:
                current_input += f"\nReference Plot Outline:\n{reference_context}\n"

            current_input += "\nStory:\n"
        else:
            current_input = construct_prompt(original_prompt, story, reference_context)
        
        # Generate
        generated_text = generate_chunk(current_input, mock=args.mock)
        
        if generated_text is None:
            print("Aborting due to API error.")
            break
            
        if not generated_text.strip():
            print("Received empty response. Stopping.")
            break
            
        print(f"Received {len(generated_text)} characters.")
        
        story += generated_text
        current_words = get_word_count(story)
        print(f"Current word count: {current_words}/{target_words}")
        
        chunk_count += 1
        
        # Safety break for mock mode to avoid infinite loops if logic is wrong
        if args.mock and chunk_count > 50: 
            print("Mock mode safety limit reached.")
            break

    # Final save
    try:
        with open(args.output_file, "w", encoding='utf-8') as f:
            f.write(story)
        print(f"\nStory saved to {args.output_file}")
        print(f"Final word count: {get_word_count(story)}")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
