import os
import json
import time

from google import genai
from google.genai import types

from utils import ReadDialogue, CreatePrompt_Hyp, Save_BatchOutput

def Retrieve_BatchOutput(
    batch_job,
    output_file:str,
    client:genai.Client
):
    print(f"Job finished with state: {batch_job.state.name}")

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':

        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a file
            result_file_name = batch_job.dest.file_name
            print(f"Results are in file: {result_file_name}")

            file_content = client.files.download(file=result_file_name)
            
            Save_BatchOutput(file_content.decode('utf-8'), output_file)
            return file_content.decode('utf-8')

        else:
            print("No results found (file).")
    else:
        print(f"Job did not succeed. Final state: {batch_job.state.name}")
        if batch_job.error:
            print(f"Error: {batch_job.error}")
        
        return None


def Create_BatchInput(
    DialogueList:list,
    instruction:str,  
    schema:str = None,
    batch_path:str = 'Batch/English/input_gemini',
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
    for Dialogue in DialogueList:
        ID = Dialogue['review_by_client']['評価ID']
        IDs.append(ID)
        prompt = CreatePrompt_Hyp(Dialogue)
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



def gemini_batch(
    batch_file:str,
    model: str = "gemini-2.5-pro",
):
    client = genai.Client(api_key='') # Your API key here

    uploaded_file = client.files.upload(
        file=batch_file+'.jsonl',
        config=types.UploadFileConfig(display_name=batch_file, mime_type='jsonl')
    )
    print(f"Uploaded file: {uploaded_file.name}")

    file_batch_job = client.batches.create(
        model=f"models/{model}",
        src=uploaded_file.name,
        config={
            'display_name': "file-upload-job-1",
        }
    )
    job_name = file_batch_job.name
    print(f"Created batch job: {job_name}")

    start = time.time()
    while True:
        batch_job = client.batches.get(name=job_name)
        if batch_job.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED'):
            break
        print(f"Job not finished. Current state: {batch_job.state.name}. Waiting 30 seconds...")
        current = time.time()
        if current - start > 3600*3:
            print("Job timed out after 3 hours.")
            client.batches.cancel(name=job_name)
            break

        time.sleep(30)
    
    output_file = batch_file.replace('input', 'output') + '.json'
    
    # If the job succeeded, retrieve the output file content and save it to the output path
    return Retrieve_BatchOutput(batch_job, output_file, client)


def main(
    IDs : list,
    target_language:str, # "English" or "Chinese"
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
    if target_language not in ["English", "Chinese"]:
        print("Invalid target language. Please choose 'English' or 'Chinese'.")
        return
    
    schema = { 
        "type": "OBJECT", 
        "required": ["dialogue"], 
        "properties": { 
            "dialogue": { 
                "type": "ARRAY", 
                "items": { 
                    "type": "OBJECT", 
                    "required": ["role", "Japanese", target_language], 
                    "properties": { 
                        "role": {"type": "STRING", "enum": ["カウンセラー", "クライアント"]}, 
                        "Japanese": {"type": "STRING"}, 
                        target_language: {"type": "STRING"} 
                    } 
                } 
            } 
        } 
    }

    DialogueList = []
    for ID in IDs:
        Dialogue = ReadDialogue(f"../{target_language}/Gemini{ID}.json")
        DialogueList.append(Dialogue)
    
    
    _, batch_file = Create_BatchInput(
        DialogueList, Instruction, schema=schema, think_low=True,
        batch_path=f'Batch/{target_language}/input_gemini'
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