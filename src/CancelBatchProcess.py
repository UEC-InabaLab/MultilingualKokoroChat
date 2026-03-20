def _prompt_nonempty(prompt: str) -> str:
    while True:
        s = input(prompt).strip()
        if s:
            return s
        print("Input cannot be empty. Please try again.")


if __name__ == "__main__":
    llm_name = _prompt_nonempty("Enter LLM name: ")
    batch_job_id = _prompt_nonempty("Enter batch job id: ")

    # NOTE:
    # If Cancel_BatchProcess also requires the LLM name, change the call below to
    # something like Cancel_BatchProcess(llm_name, batch_job_id).
    # For now, keep the existing signature and pass only batch_job_id.
    print(f"LLM name: {llm_name}")
    print(f"Canceling batch job: {batch_job_id}")

    Cancel_BatchProcess(batch_job_id)