import os
import json

def ReadDialogue(file_path): # file_path: KokoroChat File Path
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            Dialogue = json.load(f)
        return Dialogue

"""
Example Prompt Format:
カウンセラー:こんにちは、今日はどのようなお話をされたいですか？
クライアント:最近、仕事のストレスが多くて、夜も眠れないんです。
"""
def CreatePrompt_Hyp(Dialogue):
    Prompt = ""
    for dialogue in Dialogue['dialogue']:
        Prompt += f"{dialogue['role']}:{dialogue['content']}\n"
    return Prompt.strip()


# Batch Output is in JSONL format, each line is a JSON object. 
# We need to read each line, decode it as JSON, and then save the collection of JSON objects into a single JSON file.
def Save_BatchOutput(file_content:str, output_path:str):
    lines = file_content.split('\n')
    
    # Attempt to decode each line as JSON and collect valid JSON objects
    decoded_data = []
    for line in lines:
        try:
            decoded_data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {line}\nError: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(decoded_data, f, ensure_ascii=False, indent=2)


