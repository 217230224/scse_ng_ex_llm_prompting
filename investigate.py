## Import the necessary modules
import json
import ollama

## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result


## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant. "
        "You must use ONLY the provided JSON data of found items to identify possible matches. "
        "Not all details of an item must match to be a possible match; consider partial matches "
        "(e.g., same color, similar item type, or similar location).\n\n"
        "Return ONLY a JSON object, with exactly this structure:\n"
        "{\n"
        '    "matches": ["ITEM_ID"],\n'
        '    "confidence": "LOW"\n'
        "}\n\n"
        "Rules:\n"
        '- "matches" contains all possible matching item IDs from the provided data.\n'
        '- "confidence" must be exactly one of: LOW, MEDIUM, HIGH.\n'
        '- If there is no match, return an empty list for "matches".\n'
        "- Do NOT include any explanation, markdown, or extra text. Return raw JSON only.\n"
    )

    user_prompt = (
        f'A user lost an item. Description: "{description}"\n\n'
        f"Here are the available unclaimed found items (JSON):\n"
        f"{json.dumps(available_items, indent=2)}\n\n"
        f"Identify all possible matches and return the JSON result."
    )

    return system_prompt, user_prompt


## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result.
def parse_response(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


## Logic to validate the result returned by Qwen.
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    if not isinstance(result["matches"], list):
        return False
    if not isinstance(result["confidence"], str):
        return False
    if result["confidence"] not in ("LOW", "MEDIUM", "HIGH"):
        return False

    valid_ids = {item["id"] for item in available_items}
    for item_id in result["matches"]:
        if item_id not in valid_ids:
            return False

    return True


## Logic to display the matches found by Qwen in a user-friendly format.
def display_matches(result, available_items):
    print("\nMATCH RESULT")
    print("-" * 50)
    print(f"Confidence: {result['confidence']}")

    if not result["matches"]:
        print("\nNo matches were found for your description.")
        return

    print("\nPossible matches:\n")
    item_map = {item["id"]: item for item in available_items}
    for item_id in result["matches"]:
        item = item_map.get(item_id)
        if not item:
            continue
        print(f"ID: {item['id']}")
        print(f"Item: {item['item']}")
        print(f"Color: {item['color']}")
        print(f"Location: {item['location']}")
        print(f"Date found: {item['date']}")
        print()


## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50)

    description = input("\nDescribe the item you lost: ")

    items = load_items("found_items.json")
    available_items = get_unclaimed_items(items)

    system_prompt, user_prompt = build_prompt(description, available_items)

    print("\nSearching for possible matches...\n")

    response_text = ask_qwen(system_prompt, user_prompt)

    try:
        result = parse_response(response_text)
    except json.JSONDecodeError:
        print("Error: The model did not return valid JSON.")
        result = {"matches": [], "confidence": "LOW"}

    if not validate_result(result, available_items):
        print("Warning: The model response failed validation. Using empty result.")
        result = {"matches": [], "confidence": "LOW"}

    display_matches(result, available_items)

    save_result(result, "output/match_result.json")
    print("Result saved to output/match_result.json")


if __name__ == "__main__":
    main()