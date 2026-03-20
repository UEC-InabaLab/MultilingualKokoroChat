import httpx
import json
from openai import OpenAI

from utils import str2json, ReadDialogue, CreatePrompt_Hyp, SaveHypothesis

def FetchAnswer_grok(
    api_key:str,
    prompt:str,
    instruction:str,
    assistant:str,
    modelName:str,
):
    client = OpenAI(
        api_key='', # Your API key here
        base_url="https://api.x.ai/v1",
        timeout=httpx.Timeout(3600.0), # Override default timeout with longer timeout for reasoning models
    )
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]
    if assistant:
        messages.insert(1, {"role": "assistant", "content": assistant})        

    try:
        completion = client.chat.completions.create(
            model=modelName,
            messages=messages
        )
        print("Raw Response:\n", completion)
        content = completion.choices[0].message.content
        print("Final Answer:\n", content)
        return content

    except Exception as e:
        print(f"Error: {e}")
        return None
    

def main(
    api_key:str,
    ID:str,
    save_path:str = 'Batch/English/Grok',
):
    Instructions = """
# Data Description
- This is Japanese text counseling data from role-playing sessions where counselors acted as both counselor and client.
- Each line is separated by a colon (':'). The left side indicates the role name, and the right side is the utterance.

# Translation Instructions
- As a professional translator, translate this data into English.
- For the translation, please use polite and natural expressions that are appropriate for a counseling context.

# Output Format
- Your output MUST be a single, valid JSON object.
- Do not include any explanations or text outside the JSON.
- The JSON object MUST be in unpretty-printed format.
- The JSON MUST strictly conform to the schema below.
- "role" Must be either "カウンセラー" or "クライアント".
- "Japanese" Must copy the source utterance verbatim.

# JSON Schema
{"dialogue": [{"role": "string","Japanese": "string","English": "string"}]}
"""

    Dialogue = ReadDialogue(f"../KokoroChat/kokorochat_dialogues/{ID}.json")
    Prompt = CreatePrompt_Hyp(Dialogue)

    Answer = FetchAnswer_grok(
        api_key=api_key, # Your API key here
        prompt=Prompt,
        instruction=Instructions.strip(),
        assistant=None,
        modelName="grok-4-0709"
    )
    
    if Answer:
        Answer = str2json(Answer)
    
        SaveHypothesis(Dialogue, Answer, 'English', save_path)
    else:
        print(f"No valid answer received for file {ID}. Skipping saving hypothesis.")
