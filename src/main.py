import os
import json

from Hypothesis_Gemini import main as run_Hyp_Gemini
from Hypothesis_GPT import main as run_Hyp_GPT
from Hypothesis_Grok import main as run_Hyp_Grok
from Hypothesis_Qwen import main as run_Hyp_Qwen
from Refine_Gemini import main as run_Refine_Gemini

# Return IDs of dialogues that have corresponding KokoroChat files.
def Exist_KokoroChatFile(IDs:list):
    Exists = []
    for ID in IDs:
        file_path = f"../KokoroChat/kokorochat_dialogues/{ID}.json"
        if os.path.exists(file_path):
            Exists.append(ID)
    return Exists

def ConfirmExecution(
    APIKeys: dict,
    IDs: list,
    LLM: str,
    target_language: str,
):
    print("IDs to process:", IDs)
    print(f"LLM: {LLM}")
    print(f"Target Language: {target_language}")
    print(f"{LLM} API Key: ***{APIKeys[LLM][-4:]}") # Show only last 4 characters of API keys for security
    confirm = input("Execute? Run[y], Cancel[n]").strip().lower()
    if confirm != 'y':
        print('Execution was cancelled.')
        return False
    return True


if __name__ == "__main__":
    # Load API keys
    APIKeys = json.load(open("config.json"))

    IDs = [] # You can specify the IDs of the dialogues you want to process, e.g., [4, 7, 8].
    RunLLM = "" # "Gemini", "GPT", "Grok", "Qwen", "Refine"
    target_language = "" # "English" or "Chinese"
    common_path = "" # Specify the common path for saving hypotheses results, e.g., f"TestHyp/{target_language}/"
    save_path_refine = "" # Specify the path to save the refine results, e.g., "{target_language}/Refine"

    # IDs existing in KokoroChat files
    IDs = Exist_KokoroChatFile(IDs)
    if not ConfirmExecution(APIKeys, IDs, RunLLM, target_language):
        exit(0)

    if RunLLM == "Gemini":
        save_path = common_path+"Gemini" #File path will be like "TestHyp/English/Gemini4.json"
        run_Hyp_Gemini(
            api_key=APIKeys[RunLLM], # API key for Gemini
            IDs=IDs,
            target_language=target_language,
            save_path=save_path
        )

    elif RunLLM == "GPT":
        save_path = common_path+"GPT"
        run_Hyp_GPT(
            api_key=APIKeys[RunLLM],
            IDs=IDs,
            target_language=target_language,
            save_path=save_path
        )

    elif RunLLM == "Grok": # Only for English translation
        save_path = common_path+"Grok"
        if target_language != "English":
            print("Grok is only used for English translation. Please set target_language to 'English'.")
            exit(1)

        for ID in IDs:
            run_Hyp_Grok(
                api_key=APIKeys[RunLLM],
                ID=ID,
                save_path=save_path
            )

    elif RunLLM == "Qwen": # Only for Chinese translation        
        save_path = common_path+"Qwen"
        if target_language != "Chinese":
            print("Qwen is only used for Chinese translation. Please set target_language to 'Chinese'.")
            exit(1)

        for ID in IDs:
            run_Hyp_Qwen(
                api_key=APIKeys[RunLLM],
                ID=ID, 
                save_path=save_path
            )

    elif RunLLM == "Refine":
        run_Refine_Gemini(
            api_key=APIKeys["Gemini"],
            IDs=IDs,
            target_language=target_language,
            save_path=save_path_refine,
            common_path=common_path
        )

    else:
        print("Invalid LLM choice. Please choose 'Gemini', 'GPT', 'Grok', 'Qwen', or 'Refine'.")