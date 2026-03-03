from Hypothesis_Gemini import main as run_Hyp_Gemini
from Hypothesis_GPT import main as run_Hyp_GPT
from Hypothesis_Grok import main as run_Hyp_Grok
from Hypothesis_Qwen import main as run_Hyp_Qwen
from Refine_Gemini import main as run_Refine_Gemini


if __name__ == "__main__":

    IDs = [] # You can specify the IDs of the dialogues you want to process, e.g., [4, 7, 8].
    RunLLM = "" # "Gemini", "GPT", "Grok", "Qwen", "Refine"
    target_language = "" # "English" or "Chinese"

    if RunLLM == "Gemini":
        run_Hyp_Gemini(
            IDs=IDs,
            target_language=target_language
        )
    elif RunLLM == "GPT":
        run_Hyp_GPT(
            IDs=IDs,
            target_language=target_language
        )
    elif RunLLM == "Grok": # Only for English translation
        for ID in IDs:
            run_Hyp_Grok(ID)
    elif RunLLM == "Qwen": # Only for Chinese translation
        for ID in IDs:
            run_Hyp_Qwen(ID)
    elif RunLLM == "Refine":
        run_Refine_Gemini(
            IDs=IDs,
            target_language=target_language
        )
    else:
        print("Invalid LLM choice. Please choose 'Gemini', 'GPT', 'Grok', 'Qwen', or 'Refine'.")