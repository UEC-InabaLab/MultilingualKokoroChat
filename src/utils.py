import os
import json
import re
import unicodedata
from collections import defaultdict, deque
from typing import Dict, Any, List, Tuple

TILDE_CHARS = {"\u007E", "\uFF5E", "\u301C", "\u223C", "\u02DC"}  # ~  ～  〜  ∼  ˜
PROLONGED_SOUND_MARKS = {"\u30FC", "\uFF70"}  # ー, ｰ


def normalize_for_dedup(s: str) -> str:
    # unify full-width and half-width characters
    s = unicodedata.normalize("NFKC", s)
    # convert to lowercase (to ignore case differences in English, etc.)
    s = s.lower()
    # explicitly remove unwanted tilde and prolonged sound mark characters
    s = "".join(ch for ch in s if ch not in TILDE_CHARS and ch not in PROLONGED_SOUND_MARKS)
    # remove emojis, symbols, and punctuation (P = punctuation, S = symbol)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] not in ("P", "S"))
    # remove all whitespace
    s = re.sub(r"\s+", "", s)
    return s


def ReadDialogue(file_path): # file_path: KokoroChat File Path
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            Dialogue = json.load(f)
        return Dialogue
    return None

"""
Example Prompt Format:
カウンセラー:こんにちは、今日はどのようなお話をされたいですか？
クライアント:最近、仕事のストレスが多くて、夜も眠れないんです。
"""
def CreatePrompt_Hyp(Dialogue):
    Prompt = ""
    for utterance in Dialogue['dialogue']:
        role = utterance['role']
        if role == "counselor":
            role = "カウンセラー"
        elif role == "client":
            role = "クライアント"
        else:
            raise ValueError(f"Unexpected role: {role}")
        Prompt += f"{role}:{utterance['utterance']}\n"
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



def InsertTranslation2KokoroChat(
    Dialogue: Dict[str, Any],
    answer: Dict[str, Any],
    target_language: str = "Chinese",
    use_normalized_fallback: bool = True,
) -> Tuple[Dict[str, Any], str]:
    # 1) strict key buckets
    buckets: Dict[tuple, deque] = defaultdict(deque)
    # 2) normalized key -> list of original keys
    norm_index: Dict[tuple, List[tuple]] = defaultdict(list)
    # index-aware normalized key -> original keys (for multiple candidates)
    norm_index_with_index: Dict[tuple, List[tuple]] = defaultdict(list)

    # build buckets from the answer side
    for idx, item in enumerate(answer.get("dialogue", [])):
        role = item.get("role")

        # role name check / normalization
        if role not in ["カウンセラー", "クライアント"]:
            print(f"Checking role: '{role}, {item.get('Japanese')}'")
            print(item.get("Japanese"))
            print(f"Unrecognized role '{role}'. Skipping.")
            continue

        ja = item.get("Japanese")
        target = item.get(target_language)
        if not (isinstance(role, str) and isinstance(ja, str) and isinstance(target, str)):
            continue

        # original key
        key = (role, ja)
        buckets[key].append(target)
        
        if use_normalized_fallback:
            n_ja = normalize_for_dedup(ja)
            nkey = (role, n_ja)
            nkey_with_index = (role, n_ja, idx)
            # register original key reference (support multiple targets for the same source)
            if key not in norm_index[nkey]:
                norm_index[nkey].append(key)
            norm_index_with_index[nkey_with_index].append(key)

    # insert translations by exact match first
    for i, utt in enumerate(Dialogue.get("dialogue", [])):
        # skip if already translated
        if target_language in utt and isinstance(utt[target_language], str):
            continue

        role = utt.get("role")
        if role == "counselor":
            role = "カウンセラー"
        elif role == "client":
            role = "クライアント"

        ja = utt.get("utterance")
        if not (isinstance(role, str) and isinstance(ja, str)):
            continue

        key = (role, ja)
        q = buckets.get(key)

        chosen_key = None
        if q and len(q) > 0:
            chosen_key = key
        elif use_normalized_fallback:
            # try normalized-key fallback
            n_ja = normalize_for_dedup(ja)
            nkey = (role, n_ja)
            # only keys that still have remaining targets
            cand_keys = [k for k in norm_index.get(nkey, []) if len(buckets.get(k, ())) > 0]
            if len(cand_keys) == 1:
                chosen_key = cand_keys[0]
                print(f"{i}: Assigned by normalized match for '{ja}' (role={role}).")
            elif len(cand_keys) > 1:
                print(f"{i}: Multiple candidates found by normalized match for '{ja}' (role={role}).")
                nkey_with_index = (role, n_ja, i)
                keys = norm_index_with_index.get(nkey_with_index)
                if keys:
                    chosen_key = keys[0]
                    print(f"{i}: Assigned by normalized match with index for '{ja}' (role={role}).")
            else:
                print(f"{i}: No candidates found by normalized match for '{ja}' (role={role}). Skipping for now.")  

        if chosen_key:
            utt[target_language] = buckets[chosen_key].popleft()
        else:
            print(f"Warning {i}: '{ja}' not found (role={role}). Skipping.")

    # record only untranslated utterances in Miss_Japanese
    for i, utt in enumerate(Dialogue.get("dialogue", [])):
        if not (target_language in utt and isinstance(utt[target_language], str)):
            role = utt.get("role")
            ja = utt.get("utterance")
            Miss_Japanese += f"{role}:{ja}\n"

    return Dialogue, Miss_Japanese


# Create Final Dialogue Format
def FormatDialogue(
    OriginalDialogue:dict,
    TransDialogue:dict,
    target_language:str # English or Chinese
):
    FormattedDialogue = {
        "dialogue": [],
        "topic": OriginalDialogue['topic'],
        "review_by_client": OriginalDialogue['review_by_client_en']
    }
    for origin_utt, trans_utt in zip(OriginalDialogue['dialogue'], TransDialogue['dialogue']):
        formatted_utt = {
            "role": origin_utt['role'],
            "time": origin_utt['time'],
            "origin": origin_utt['utterance'],
            "content": trans_utt[target_language]
        }
        FormattedDialogue['dialogue'].append(formatted_utt)

    return FormattedDialogue


def SaveHypothesis(
    OriginalDialogue:dict,
    Answer:dict, # API Output after parsing
    target_language:str,
    save_path:str
):
    # Insert Translation to each utterance in KokoroChat Dialogue Format,
    # and also return any Japanese utterances that failed to find a match for debugging.
    Inserted, MissJapanese = InsertTranslation2KokoroChat(OriginalDialogue, Answer, target_language, use_normalized_fallback=True)

    # Format the dialogue into the final structure 
    FormattedDialogue = FormatDialogue(OriginalDialogue, Inserted, target_language)

    ID = OriginalDialogue['review_by_client_en']['evaluation_id']
    if not MissJapanese:
        with open(f"{save_path}{ID}.json", 'w', encoding='utf-8') as f:
            json.dump(FormattedDialogue, f, ensure_ascii=False, indent=2)
        print(f"Successfully processed file {ID}")
    else:
        print(f"Failed to insert translation for dialogue ID {ID} due to missing Japanese utterances:\n{MissJapanese}\n")