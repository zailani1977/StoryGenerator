import requests
import re
import sys
import argparse
import time
import configparser
import os

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

    TEMPERATURE = float(config['Generation']['temperature'])
    TOP_P = float(config['Generation']['top_p'])
except (KeyError, ValueError) as e:
    print(f"Error reading configuration: {e}")
    sys.exit(1)

# Override with environment variable if present
if 'GEMINI_API_KEY' in os.environ:
    GEMINI_API_KEY = os.environ['GEMINI_API_KEY']

def load_prompt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        sys.exit(1)

def get_word_count(text):
    return len(text.split())

def extract_target_length(prompt_text, default_val=DEFAULT_TARGET_WORDS):
    # Look for "no less than X words"
    match = re.search(r"no less than (\d+) words", prompt_text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return default_val

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

def analyze_story(story_text):
    """
    Analyzes the provided story text to extract storyline, plot, characters, and setting.
    """
    analysis_prompt = (
        "Analyze the following story and provide a concise summary of the storyline, "
        "plot, key characters, and setting. Format it clearly.\n\n"
        f"Story Text:\n{story_text}"
    )

    # We use generate_chunk logic but with a specific prompt.
    # The 'mock' argument isn't available here directly, but we can assume false or pass it if needed.
    # For simplicity, we just call the API function directly based on type.

    print("Analyzing reference story...")

    if API_TYPE == 'kobold':
        return generate_chunk_kobold(analysis_prompt)
    elif API_TYPE == 'gemini':
        return generate_chunk_gemini(analysis_prompt)
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
    
    try:
        response = requests.post(API_URL, json=payload)
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
    
    try:
        response = requests.post(url, json=payload)
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

def construct_prompt(original_instruction, story_so_far):
    # Strategy: 
    # Keep the instruction at the top to maintain the premise.
    # Append the last N characters of the story to fit within context.
    # We need to leave room for the instruction and the new generation.
    
    instruction_part = f"Instructions: {original_instruction}\n\nStory so far:\n"
    
    # We need to ensure that (Prompt + Generation) <= MAX_CONTEXT_LENGTH
    # So Prompt <= MAX_CONTEXT_LENGTH - MAX_GEN_LENGTH
    max_prompt_tokens = MAX_CONTEXT_LENGTH - MAX_GEN_LENGTH
    safe_char_limit = int(max_prompt_tokens * 3) # Approx 3 chars per token
    
    available_chars_for_story = safe_char_limit - len(instruction_part)
    
    if available_chars_for_story < 100:
        # Instruction is too long, just send the tail of the story
        return story_so_far[-safe_char_limit:]
    
    if len(story_so_far) > available_chars_for_story:
        context_story = story_so_far[-available_chars_for_story:]
    else:
        context_story = story_so_far
        
    return instruction_part + context_story

def main():
    parser = argparse.ArgumentParser(description="Generate a story using KoboldCPP or Gemini.")
    parser.add_argument("--prompt-file", default="prompt.txt", help="Path to the prompt file.")
    parser.add_argument("--output-file", default="generated_story.txt", help="Path to save the generated story.")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode (no API calls).")
    args = parser.parse_args()

    print("--- Story Generator Started ---")
    print(f"API Provider: {API_TYPE}")
    
    original_prompt = load_prompt(args.prompt_file)
    
    # Check for reference story
    reference_context = ""
    reference_word_count = 0
    if REFERENCE_STORY_PATH and os.path.exists(REFERENCE_STORY_PATH):
        print(f"Reading reference story from: {REFERENCE_STORY_PATH}")
        try:
            with open(REFERENCE_STORY_PATH, 'r', encoding='utf-8') as f:
                ref_text = f.read()

            reference_word_count = get_word_count(ref_text)
            print(f"Reference story word count: {reference_word_count}")

            # Limit ref_text to avoid token overflow during analysis if it's huge
            # Simple truncation for now.
            # Assuming max context length applies to analysis prompt too.
            max_analysis_input = int(MAX_CONTEXT_LENGTH * 3)
            if len(ref_text) > max_analysis_input:
                print("Warning: Reference story is too long, truncating for analysis.")
                ref_text = ref_text[:max_analysis_input]

            if not args.mock:
                analysis = analyze_story(ref_text)
            else:
                analysis = "[Mock Analysis: The story is about X doing Y.]"

            if analysis:
                print("Reference story analyzed successfully.")
                reference_context = f"\n\nBased on the following storyline/plot from a reference story:\n{analysis}\n"
            else:
                print("Warning: Failed to analyze reference story.")

        except Exception as e:
            print(f"Error processing reference story: {e}")

    # Determine target words
    # If reference story exists, its length is used as the default target, replacing DEFAULT_TARGET_WORDS
    default_target = reference_word_count if reference_word_count > 0 else DEFAULT_TARGET_WORDS
    target_words = extract_target_length(original_prompt, default_target)

    print(f"Goal: Generate a story of at least {target_words} words.")

    story = ""
    chunk_count = 0
    
    while get_word_count(story) < target_words:
        print(f"\nGenerating chunk {chunk_count + 1}...")
        
        # Construct prompt
        if not story:
            # First iteration: Just the instruction, maybe slightly formatted
            current_input = original_prompt

            # Inject reference context
            if reference_context:
                current_input += reference_context

            # Optional: Add a starter to guide the model into story mode
            if not current_input.endswith("\n"):
                current_input += "\n"
            current_input += "\nStory:\n"
        else:
            # We might want to keep the reference context in subsequent prompts too?
            # construct_prompt currently takes original_instruction.
            # We can append reference_context to original_instruction so it stays in context.
            effective_instruction = original_prompt + reference_context if reference_context else original_prompt
            current_input = construct_prompt(effective_instruction, story)
        
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
