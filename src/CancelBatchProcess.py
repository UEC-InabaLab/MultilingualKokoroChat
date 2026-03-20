import json
import time
from openai import OpenAI
from google import genai


def user_input(message: str) -> str:
    while True:
        s = input(message).strip()
        if s:
            return s
        print("Input cannot be empty. Please try again.")


def Cancel_BatchProcess_Gemini(batch_job_id: str):
    # Load API keys
    APIKeys = json.load(open("config.json"))

    client = genai.Client(api_key=APIKeys['Gemini'])
    batch_job = client.batches.get(name=batch_job_id)
    print("before:", batch_job.state.name)
    client.batches.cancel(name=batch_job_id)
    time.sleep(30)
    print("after:", batch_job.state.name)


def Cancel_BatchProcess_GPT(batch_job_id: str):
    # Load API keys
    APIKeys = json.load(open("config.json"))

    client = OpenAI(api_key=APIKeys['GPT'])
    batch = client.batches.retrieve(batch_job_id)
    print("before: ", batch.status)
    batch = client.batches.cancel(batch_job_id)
    time.sleep(30)
    print("after: ", batch.status)



if __name__ == "__main__":
    LLM = user_input("Enter LLM name: ")

    if LLM == "Gemini":
        batch_job_id = user_input("Enter batch job id: ")
        Cancel_BatchProcess_Gemini(batch_job_id)
    elif LLM == "GPT":
        batch_job_id = user_input("Enter batch job id: ")
        Cancel_BatchProcess_GPT(batch_job_id)
    else:
        print("Unsupported LLM. Please enter 'Gemini' or 'GPT'.")