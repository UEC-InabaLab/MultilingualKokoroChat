import os
import json
import time
from openai import OpenAI
from copy import deepcopy
from pydantic import BaseModel, Field
from typing import Literal, List

from utils import ReadDialogue, CreatePrompt_Hyp, Save_BatchOutput, SaveHypothesis

class Utterance_Zh(BaseModel):
    role: Literal['カウンセラー', 'クライアント']
    Japanese:str = Field(..., description="The original Japanese utterance.")
    Chinese:str = Field(..., description="The translated Chinese utterance.")

class Utterance_En(BaseModel):
    role: Literal['カウンセラー', 'クライアント']
    Japanese:str = Field(..., description="The original Japanese utterance.")
    English:str = Field(..., description="The translated English utterance.")
    
class Dialogue_Zh(BaseModel):
    dialogue: List[Utterance_Zh]

class Dialogue_En(BaseModel):
    dialogue: List[Utterance_En]

def to_strict_json_schema(schema: dict) -> dict:
    s = deepcopy(schema)
    def visit(node):
        if isinstance(node, dict):
            # すべてのobjectに additionalProperties: False を強制
            if node.get("type") == "object":
                node.setdefault("properties", {})
                node["additionalProperties"] = False
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)
    visit(s)
    return s


def Create_BatchInput(
    DialogueList:list,
    instruction:str,
    batch_path:str = 'Batch/English/input_gpt',
    model:str = "gpt-5-2025-08-07",
    schema:str = None,
)->tuple[list, str]:
    requests = []
    if schema:
        format = {
            "type": "json_schema",
            "json_schema": {
                "name": "Dialogue",
                "schema": schema,
                "strict": True
            }
        }

    IDs =[]
    for Dialogue in DialogueList:
        ID = Dialogue['review_by_client_en']['evaluation_id']
        IDs.append(ID)
        prompt = CreatePrompt_Hyp(Dialogue)
        request = {
            "model": model,
            "messages":[
                {
                    "role": "system",
                    "content": instruction
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        }
        if schema:
            request["response_format"] = format
    
        requests.append(
            {
                "custom_id": f"{ID}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": request
            }
        )
    if not IDs:
        print("No valid dialogues found. Please check the input data.")
        return [], ""

    IDs.sort()
    start = IDs[0]
    end = IDs[-1]
    file = f"{batch_path}{start}_{end}"
    with open(file+'.jsonl', "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    
    return requests, file


def FetchAnswer_gpt_batch(
    api_key:str,
    batch_file:str
):
    client = OpenAI(api_key=api_key) # Your API key here

    uploaded = client.files.create(
        file=open(batch_file+'.jsonl', "rb"),
        purpose="batch",
    )
    print(f"Batch input file uploaded: {uploaded.id}")
    print(uploaded)

    # begin batch process
    batch_start = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    print(f"Batch process started: {batch_start.id}")
    print(batch_start)

    # wait for batch process to complete
    while True:
        batch_process = client.batches.retrieve(batch_start.id)
        if batch_process.status in ["completed", "failed", "expired", "cancelled"]:
            break
        print(f"Job not finished. Current state: {batch_process.status}. Waiting 30 seconds...")
        time.sleep(30)

    print(f"Batch process finished with status: {batch_process.status}")

    if batch_process.status == "completed":
        if batch_process.output_file_id is None:
            print("No output file ID found.")
            error_response = client.files.content(batch_process.error_file_id)
            print(error_response.text)
            error_file = batch_file.replace('input', 'error') + '.json'
            Save_BatchOutput(error_response.text, error_file)
            return None
            
        response = client.files.content(batch_process.output_file_id)
        print(response.text)
        output_file = batch_file.replace('input', 'output') + '.json'
        Save_BatchOutput(response.text, output_file)

        return response.text
    elif batch_process.status in ["failed", "expired", "cancelled"]:
        print(f"Batch process failed with status: {batch_process.status}")
        print(batch_process)
        if batch_process.status != "failed":  # "failed": skip file
           client.batches.cancel(batch_start.id)


def Extract_GPT_BatchOutput(
    output_file:str
):
    with open(output_file, "r", encoding="utf-8") as f:
        all_outputs = json.load(f)

    Answers = {}
    for file_output in all_outputs:
        ID = int(file_output["custom_id"])
        try:
            content = file_output["response"]["body"]["choices"][0]["message"]["content"]
            content = json.loads(content)
            print(f"{ID}:\n")
            print(content)
            Answers[ID] = content
    
        except Exception as e:
            Answers[ID] = None
            print(f"Error processing output for file {ID}: {e}")

    return Answers



def main(
    api_key:str,
    IDs : list,
    target_language:str, # "English" or "Chinese",
    save_path:str = f"TestHyp/English/GPT"
):
    Instruction=f"""
# Data Description
- This is Japanese text counseling data from role-playing sessions where counselors acted as both counselor and client.
- Each line is separated by a colon (':'). The left side indicates the role name, and the right side is the utterance.

# Translation Instructions
- As a professional translator, translate this data into {target_language}.
- For the translation, please use polite and natural expressions that are appropriate for a counseling context.

# Output Format
- The output should be in unpretty-printed JSON format.
"""
    if target_language == "English":
        schema = to_strict_json_schema(Dialogue_En.schema())
    elif target_language == "Chinese":
        schema = to_strict_json_schema(Dialogue_Zh.schema())
    else:
        print("Invalid target language. Please choose 'English' or 'Chinese'.")
        return
    
    DialogueList = []
    for ID in IDs:
        Dialogue = ReadDialogue(f"../KokoroChat/kokorochat_dialogues/{ID}.json")
        if Dialogue:
            DialogueList.append(Dialogue)
    
    
    _, batch_file = Create_BatchInput(
        DialogueList, Instruction, schema=schema, model="gpt-5-2025-08-07",
        batch_path=f'Batch/{target_language}/input_gpt'
    )
    if os.path.exists(batch_file+'.jsonl'):
        answer = FetchAnswer_gpt_batch(
            api_key=api_key,
            batch_file=batch_file
        )

    output_file = batch_file.replace("input", "output") + '.json'
    print("Batch output saved to:", output_file)

    Answers = Extract_GPT_BatchOutput(output_file)

    for Dialogue in DialogueList:
        ID = Dialogue['review_by_client_en']['evaluation_id']
        if ID in Answers and Answers[ID]:
            SaveHypothesis(Dialogue, Answers[ID], target_language, save_path)
        else:
            print(f"No valid answer for dialogue ID {ID}. Skipping saving Gemini hypothesis.")