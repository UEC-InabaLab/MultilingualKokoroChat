import os
from openai import OpenAI

from utils import ReadDialogue, CreatePrompt_Hyp, SaveHypothesis

def FetchAnswer_qwen(
    api_key:str,
    prompt:str,
    instruction:str,
    assistant:str,
    modelName:str,
    enable_thinking:bool = False
):
    client = OpenAI(
        api_key=api_key, # Your API key here
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1", # Singapore region
    )
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt},
    ]
    if assistant:
        messages.insert(1, {"role": "assistant", "content": assistant})
    
    translation_options = {
        "source_lang": "Japanese",
        "target_lang": "Chinese"
    }
    stream = False
    extra_body = {
        "enable_thinking": enable_thinking,
        "translation_options": translation_options
    }
    if enable_thinking:
        stream = True
        extra_body["result_format"] = "message"
        
    try:
        completion = client.chat.completions.create(
            model=modelName,
            messages=messages,
            stream=stream,
            extra_body=extra_body
        )
        if not stream:
            return completion.choices[0].message.content
        

        reasoning_content = ""
        content = ""

        for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
                continue

            delta = chunk.choices[0].delta

            # Collect only the reasoning content.
            if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                reasoning_content += delta.reasoning_content

            # After the content is received, start generating the response.
            if hasattr(delta, "content") and delta.content:
                content += delta.content
        
        return content, reasoning_content

    except Exception as e:
        print(f"Error: {e}")
        return None
    


def main(
    api_key:str,
    ID:str,
    save_path:str = 'TestHyp/Chinese/Qwen'
):
    Instructions = """
# Data Description
- This is Japanese text counseling data from role-playing sessions where counselors acted as both counselor and client.
- Each line is separated by a colon (':'). The left side indicates the role name, and the right side is the utterance.

# Translation Instructions
- As a professional translator, translate this data into Chinese.
- For the translation, please use polite and natural expressions that are appropriate for a counseling context.

# Output Format
- Your output MUST be a single, valid JSON object.
- Do not include any explanations or text outside the JSON.
- The JSON object MUST be in unpretty-printed format.
- The JSON MUST strictly conform to the schema below.
- "role" Must be either "カウンセラー" or "クライアント".
- "Japanese" Must copy the source utterance verbatim.

# JSON Schema
{"dialogue": [{"role": "string","Japanese": "string","Chinese": "string"}]}
"""

    Dialogue = ReadDialogue(f"../KokoroChat/kokorochat_dialogues/{ID}.json")
    Prompt = CreatePrompt_Hyp(Dialogue)

    Answer = FetchAnswer_qwen(
        api_key=api_key, # Your API key here
        prompt=Prompt,
        instruction=Instructions.strip(),
        assistant=None,
        modelName="qwen-plus-2025-07-28",
        enable_thinking=True
    )
    
    SaveHypothesis(Dialogue, Answer, 'Chinese', save_path)