from Hypothesis_Gemini import main as run_Hyp_Gemini
from Hypothesis_GPT import main as run_Hyp_GPT
from Hypothesis_Grok import main as run_Hyp_Grok
from Hypothesis_Qwen import main as run_Hyp_Qwen
from Refine_Gemini import main as run_Refine_Gemini


if __name__ == "__main__":
    IDs = [] # You can specify the IDs of the dialogues you want to process, e.g., [4, 7, 8].
    RunLLM = "" # "Gemini", "GPT", "Grok", "Qwen", "Refine"
    target_language = "" # "English" or "Chinese"
    common_path = "" # Specify the common path for saving hypotheses results, e.g., f"TestHyp/{target_language}/"
    save_path_refine = "" # Specify the path to save the refine results, e.g., "{target_language}/Refine"

    if RunLLM == "Gemini":
        api_gemini_key = "" # Your API key here

        save_path = common_path+"Gemini" #File path will be like "TestHyp/English/Gemini4.json"
        run_Hyp_Gemini(
            api_key=api_gemini_key,
            IDs=IDs,
            target_language=target_language,
            save_path=save_path
        )

    elif RunLLM == "GPT":
        api_gpt_key = "" # Your API key here

        save_path = common_path+"GPT"
        run_Hyp_GPT(
            api_key=api_gpt_key,
            IDs=IDs,
            target_language=target_language,
            save_path=save_path
        )

    elif RunLLM == "Grok": # Only for English translation
        api_grok_key = "" # Your API key here

        save_path = common_path+"Grok"
        if target_language != "English":
            print("Grok is only used for English translation. Please set target_language to 'English'.")
            exit(1)

        for ID in IDs:
            run_Hyp_Grok(
                api_key=api_grok_key,
                ID=ID,
                save_path=save_path
            )

    elif RunLLM == "Qwen": # Only for Chinese translation
        api_qwen_key = "" # Your API key here

        save_path = common_path+"Qwen"
        if target_language != "Chinese":
            print("Qwen is only used for Chinese translation. Please set target_language to 'Chinese'.")
            exit(1)

        for ID in IDs:
            run_Hyp_Qwen(
                api_key=api_qwen_key,
                ID=ID, 
                save_path=save_path
            )

    elif RunLLM == "Refine":
        api_gemini_key = "" # Your API key here

        run_Refine_Gemini(
            api_key=api_gemini_key,
            IDs=IDs,
            target_language=target_language,
            save_path=save_path_refine,
            common_path=common_path
        )

    else:
        print("Invalid LLM choice. Please choose 'Gemini', 'GPT', 'Grok', 'Qwen', or 'Refine'.")