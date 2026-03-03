import os
import json

from Hypothesis_Gemini import gemini_batch

# Check if all LLM JSON files exist for this ID
def Exist_LLMsJson(ID_Range: list, LLMs:list, common_path:str) -> list:
    ID_List = []
    for i in ID_Range:
        # {LLM name}{ID}.json, e.g., GPT4.json, Gemini4.json, Grok4.json
        if all(os.path.exists(f"{common_path}{llm}{i}.json") for llm in LLMs):
            ID_List.append(i)
    return ID_List

# Read the JSON files for the IDs and LLMs creating hypotheses
def Read_LLMsJson(ID:int, LLMs:list, common_path:str) -> list:
    Dialogue = {}
    for llm in LLMs:
        file_path = f"{common_path}{llm}{ID}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                # key: LLM name, value: hypothesis dialogue created by that LLM
                Dialogue[llm] = json.load(file)
                if Dialogue[llm] is None:
                    print(file_path)
                    print(file.read())
                    raise ValueError(f"Failed to load JSON for {llm} at file number {ID}.")
    return Dialogue


"""
Example InputJson object:
{ID: 4, 'dialogue': [{'role': 'カウンセラー', 'source': 'こんにちは、今日はどのようなお話をされたいですか？', 'hypothesis1': 'Hello, what would you like to talk about today?', 'hypothesis2': 'Hi, what would you like to discuss today?', 'hypothesis3': 'Hello, what would you like to discuss today?'}]}

English: (hypothesis1: GPT, hypothesis2: Gemini, hypothesis3: Grok)
Chinese: (hypothesis1: GPT, hypothesis2: Gemini, hypothesis3: Qwen)
"""
def Create_InputJson(
    Dialogue: dict, 
    LLMs: list,
    target_language: str # English or Chinese
) -> dict:

    # Initialize: Store Japanese original text first
    InputJson = {
        'ID': Dialogue[LLMs[0]]['review_by_client']['評価ID'],
        'dialogue': [{'role': utterance['role'], 'source': utterance['content']} for utterance in Dialogue[LLMs[0]]['dialogue']]
    }
    # Add hypotheses for each LLM
    for i, llm in enumerate(LLMs):
        for Turn_idx, utterance in enumerate(Dialogue[llm]['dialogue']):
            if target_language not in utterance:
                raise ValueError(f"{llm}'s translation missing for utterance: {utterance['content']}")
            hypothesis = utterance[target_language]
            if InputJson['dialogue'][Turn_idx]['source'] != utterance['content']:
                raise ValueError(f"Source mismatch for utterance: {utterance['content']}")
            InputJson['dialogue'][Turn_idx][f'hypothesis{i+1}'] = hypothesis
    return InputJson


def Create_BatchInput(
    InputList:list,
    instruction:str,  
    batch_path:str = 'Batch/English/input_refine_gemini',
    schema:str = None,
    think_low:bool=False
)->tuple[list, str]:
    requests = []
    config = {}
    if schema:
        config = {
            "responseMimeType": "application/json",
            "responseSchema": schema
        }
    if think_low:
        config["thinkingConfig"] = {"thinkingLevel": "LOW"}

    IDs =[]
    for input in InputList:
        ID = input['review_by_client']['評価ID']
        IDs.append(ID)
        prompt = json.dumps(input['dialogue'], ensure_ascii=False)
        request = {
            "contents":[{
                "parts": [{'text': prompt}],
                "role": "user"
            }],
            "systemInstruction": {
                "parts": [{"text": instruction}]
            }
        }
        if config:
            request["generationConfig"] = config
        
        requests.append(
            {"key": f"{ID}", "request": request}
        )
    
    if not IDs:
        print("No new files to process. All files already exist.")
        return [], ""
    
    IDs.sort()
    start = IDs[0]
    end = IDs[-1]
    file = f"{batch_path}{start}_{end}" # File name based on the range of IDs in the batch
    with open(file+'.jsonl', "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    
    return requests, file


def main(
    IDs : list,
    target_language:str, # "English" or "Chinese"
):
    Instruction = f"""
# Data Description
- This data includes Japanese text counseling data and its {target_language} translation candidates.
- Japanese text counseling data was collected through role-playing between counselors acting as counselor and client.

# Input Data Format
- The input is a list of dictionary objects.
## Keys of each dictionary
    - 'role': Role of the speaker ('カウンセラー' or 'クライアント')
    - 'source': Japanese original text
    - 'hypothesis1', 'hypothesis2', 'hypothesis3': {target_language} translation candidates

# Evaluation Instructions
- You are a professional translator, evaluate the {target_language} translation candidates.
- For each utterance, follow these steps for evaluation:
    1. Analysis of Each Translation Candidate
    - Compare each translation candidate and describe specifically which parts are superior.
    - Describe specifically which parts need improvement.
    2. Construction of an Improved Translation
    - Based on your analysis, synthesize a revised translation by combining the strengths of both candidates.
    - Make corrections based on the areas for improvement you identified.
    - Ensure consistent terminology to maintain consistency throughout the translation.

# Output Format
- The output should be in unpretty-printed JSON format.
"""
    if target_language == "English":
        LLMs = ['GPT', 'Gemini', 'Grok']
    elif target_language == "Chinese":
        LLMs = ['GPT', 'Gemini', 'Qwen']
    else:
        print("Invalid target language. Please choose 'English' or 'Chinese'.")
        return
    
    # The common path where the JSON files for Hypotheses by all LLMs are stored
    common_path = f'TestHyp/{target_language}/'

    InputList = []
    ExistAllLLMs_IDs = Exist_LLMsJson(IDs, LLMs, common_path=common_path)
    for ID in ExistAllLLMs_IDs:
        Dialogue = Read_LLMsJson(ID, LLMs, common_path=common_path)
        InputList.append(Create_InputJson(Dialogue, LLMs, target_language))
    schema = { 
        "type": "OBJECT", 
        "required": ["dialogue"], 
        "properties": { 
            "dialogue": { 
                "type": "ARRAY", 
                "items": { 
                    "type": "OBJECT", 
                    "required": ["Source", "Think", "Translation"], 
                    "properties": { 
                        "Source": {"type": "STRING", "description": "The original Japanese utterance."}, 
                        "Think": {
                            "type": "OBJECT",
                            "required": ["hypothesis1", "hypothesis2", "hypothesis3", "overall"],
                            "properties": {
                                "hypothesis1": {"type": "STRING", "description": "The superior points and improvement points of hypothesis1"},
                                "hypothesis2": {"type": "STRING"},
                                "hypothesis3": {"type": "STRING"},
                                "overall": {"type": "STRING", "description": "A general assessment summarizing the comparison"}
                            }
                        }, 
                        "Translation": {"type": "STRING", "description": "Output the final, improved translation based on your analysis in the Think field."} 
                    }
                } 
            } 
        } 
    }
    _, batch_file = Create_BatchInput(
        InputList, Instruction, schema=schema, think_low=True,
        batch_path=f'Batch/{target_language}/input_refine_gemini'
    )

    answer = gemini_batch(batch_file, model="gemini-2.5-pro")
    if not answer:
        print("Batch processing failed or returned no output.")
        return
    output_file = batch_file.replace("input", "output") + '.json'
    print("Batch output saved to:", output_file)

    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(answer)
            with open(output_file, "r", encoding="utf-8") as f:
                answer = json.load(f)

    Answers = {}
    for output in answer:
        try:
            content = output["response"]["candidates"][0]["content"]["parts"][0]['text']
            content = json.loads(content)
            ID = output['key']
            print(f"{ID}:\n")
            print(content)
            Answers[ID] = content
        except Exception as e:
            Answers[ID] = None
            print(f"Error processing output for file {ID}: {e}")

    return Answers