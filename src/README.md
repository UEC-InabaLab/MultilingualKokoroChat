This document explains how to reproduce our experiments.

Please follow the steps below.

### 1. Install dependencies
From the project root:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Prepare KokoroChat
Clone KokoroChat from the official repository (from the project root):

```bash
git clone https://github.com/UEC-InabaLab/KokoroChat.git
```

### 3. Prepare directories
Move to ``src/`` and create the required directories,

```bash
cd src

mkdir Batch
mkdir Batch/English
mkdir Batch/Chinese
```
These directories for saving the raw output and input of batch API processing

### 4. Configure `config.json`
Open `config.json` in the project root and set your API keys for each LLM.

### 5. Configure ``main.py``
Open ``main.py`` and fill in the following variables:

・``IDs``:
The list of evaluation dialogue IDs you want to translate. Example: 
```bash
IDs = [4, 7, 8]
IDs = list(range(4, 10))
```

・``RunLLM``: The LLM name used for translation.

・``target_language``: English or Chinese 

・``common_path``: Base directory path where hypothesis results will be saved. Exaple:

```bash
common_path = f"Results/{target_language}/"
```

・``save_path_refine``: Directory path where refined results will be saved. Example:

```bash
save_path_refine = f"Results/{target_language}/Refine"
```

### 6. Create output directories
Create the directories you specified in ``common_path`` and ``save_path_refine``.
If you use example I indicated before,
from ``src/``

```bash
mkdir Results
mkdir Results/English
mkdir Results/Chinese
```

Adjust the paths above so that they match your actual ``common_path`` and ``save_path_refine`` values.

### 7. Run the experiment
From ``src``:

```bash
python main.py
```
Repeat the execution while changing the LLM, language, and IDs as needed.
Once you have generated three types of hypotheses for each ID, set ``Refine`` to ``RunLLM``.
By running ``main.py`` in this configuration, you can perform translation using our proposed method.