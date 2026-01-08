"""
Simple translation script for policy and transcript data.

Usage:
python translate_policy_transcripts.py --input_file data.csv --output_file translated.csv
python translate_policy_transcripts.py --input_file data.csv --output_file translated.csv --resume
"""

import os
import argparse
import pandas as pd
import re
import json
from tqdm.auto import tqdm
from deep_translator import GoogleTranslator
import time
from datetime import datetime

# Target languages by country
# Target languages mapping by country code
TARGET_LANGS = {
    "GHA": ["Akan", "Ewe"],
    "KEN": ["Swahili"],
    "MWI": ["Nyanja"],
    "NGA": ["Hausa", "Yoruba", "Igbo"],
    "SA": ["Zulu", "Xhosa"],
    "UGA": ["Luganda"],
    # Short codes
    "GH": ["Akan", "Ewe"],
    "KE": ["Swahili"],
    "MW": ["Nyanja"],
    "NG": ["Hausa", "Yoruba", "Igbo"],
    "SA": ["Zulu", "Xhosa"],
    "UG": ["Luganda"]
}

# Language codes for Google Translate
LANG_CODES = {
    "Akan": "ak",
    "Ewe": "ee",
    "Swahili": "sw",
    "Nyanja": "ny",
    # "Tumbuka": "auto",
    "Hausa": "ha",
    "Yoruba": "yo",
    "Igbo": "ig",
    "Zulu": "zu",
    "Xhosa": "xh",
    "Luganda": "lg"
}

def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language."""
    if GoogleTranslator is None:
        raise ModuleNotFoundError(
            "deep_translator is not installed. Install it or monkeypatch translate_text/translate_transcript for offline runs."
        )
    target_code = LANG_CODES.get(target_lang, "auto")
    if len(text.strip()) < 2:
        return text
    translator = GoogleTranslator(source="en", target=target_code)

    return translator.translate(text=text) + "\n"

def parse_transcript(transcript: str) -> list:
    """Parse transcript into User:/Agent: turns."""
    turns = []
    pattern = r'(User:|Agent:)'
    parts = re.split(pattern, transcript)
    parts = [p.strip() for p in parts if p.strip()]
    
    i = 0
    while i < len(parts):
        if parts[i] in ['User:', 'Agent:']:
            if i + 1 < len(parts) and parts[i + 1] not in ['User:', 'Agent:']:
                turns.append({'role': parts[i], 'text': parts[i + 1]})
                i += 2
            else:
                i += 1
        else:
            i += 1
    return turns

def translate_transcript(transcript: str, target_lang: str) -> str:
    """Translate transcript turn by turn, keeping User:/Agent: labels."""
    turns = parse_transcript(transcript)
    translated_turns = []
    
    for turn in turns:
        translated_text = translate_text(turn['text'], target_lang)
        translated_turns.append(f"{turn['role']} {translated_text}")
    
    return " ".join(translated_turns)

def load_checkpoint(checkpoint_file: str) -> set:
    """Load completed (row_id, lang) pairs from checkpoint."""
    if not os.path.exists(checkpoint_file):
        return set()

    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    return set(tuple(x) for x in data.get('completed', []))

def save_checkpoint(
    checkpoint_file: str,
    completed: set,
    *,
    total: int | None = None,
    written: int | None = None,
    errors: list[dict] | None = None,
):
    """Save progress to checkpoint.

    The checkpoint is meant to make progress recoverable:
    - 'completed': list of [row_id, lang]
    - 'stats': lightweight counters for visibility
    - 'errors': recent failures for debugging (bounded by code that calls this)
    """
    payload = {
        'completed': [list(x) for x in sorted(completed)],
        'updated_at': datetime.utcnow().isoformat() + 'Z',
    }
    if total is not None or written is not None:
        payload['stats'] = {
            'total': total,
            'written': written,
            'completed_pairs': len(completed),
        }
    if errors is not None:
        payload['errors'] = errors

    with open(checkpoint_file, 'w') as f:
        json.dump(payload, f, ensure_ascii=False)


def _checkpoint_path_for_output(output_file: str) -> str:
    """Place checkpoint next to output_file so it's easy to locate."""
    output_abs = os.path.abspath(output_file)
    base, _ = os.path.splitext(output_abs)
    return base + "_checkpoint.json"


def append_row_to_csv(row_dict: dict, output_file: str):
    """Append a single translated row to output_file.

    This makes work durable even if translation is interrupted.
    """
    out_df = pd.DataFrame([row_dict])
    write_header = (not os.path.exists(output_file)) or os.path.getsize(output_file) == 0
    out_df.to_csv(output_file, mode='a', header=write_header, index=False)


def load_completed_from_output(output_file: str) -> set:
    """Best-effort: infer completed (original_row_id, lang) pairs from output CSV.

    We expect row_id like '<original_row_id>_<lang_lower>' and a 'language' column.
    """
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return set()

    out = pd.read_csv(output_file, usecols=['row_id', 'language'])

    completed = set()
    for _, r in out.iterrows():
        rid = str(r.get('row_id', ''))
        lang = r.get('language', '')
        if not rid or not lang:
            continue
        if '_' not in rid:
            continue
        original_row_id = rid.rsplit('_', 1)[0]
        completed.add((original_row_id, str(lang)))
    return completed

def process_dataset(input_file: str, output_file: str, resume: bool = False):
    """
    Translate every row to all target languages for its country.
    
    For each row:
    1. Get target languages based on country_code
    2. Translate policy and transcript to each language
    3. Create new row with translated content and language column
    """
    checkpoint_file = _checkpoint_path_for_output(output_file)
    completed = load_checkpoint(checkpoint_file) if resume else set()
    errors: list[dict] = []
    written_rows = 0

    # If resuming, also consider what's already written to the output CSV to avoid duplicates.
    if resume:
        completed |= load_completed_from_output(output_file)
    
    if resume:
        print(f"Resuming: {len(completed)} translations already completed")
    
    print(f"Reading {input_file}...")
    if input_file.endswith('.jsonl'):
        df = pd.read_json(input_file, lines=True)
    else:
        df = pd.read_csv(input_file)
    df.fillna("", inplace=True)

    def get_country_code(row):
        if row['row_id'][:3] in TARGET_LANGS:
            return row['row_id'][:3]
        elif row['row_id'][:2] in TARGET_LANGS:
            return row['row_id'][:2]
        else:           
            return None

    df['country_code'] = df.apply(get_country_code, axis=1)
    df.loc[df['label']=='FAIL', 'policy'] = ""

    df = df[df['label']=='FAIL'].reset_index(drop=True)

    total_translations = sum(
        len(TARGET_LANGS.get(str(row.get('country_code', '')), [])) for _, row in df.iterrows()
    )
    
    print(f"\nTranslating {len(df)} rows into {total_translations} translations...")
    
    with tqdm(total=len(df), desc="Processing rows") as pbar:
        for _, row in df.iterrows():
            country_code = str(row.get('country_code', ''))
            target_langs = TARGET_LANGS.get(country_code, [])
            
            if not target_langs:
                print(f"\nWarning: No target languages for {country_code}")
                pbar.update(1)
                continue
            
            original_row_id = str(row.get('row_id', ''))
            
            for lang in target_langs:
                if (original_row_id, lang) in completed:
                    continue
                
                print(f"\n{original_row_id} -> {lang}")
                
                try:
                    # Translate policy
                    print("  Translating policy...")
                    policy_lines = row['policy'].split("\n")
                    pre_policy = ""
                    for line in policy_lines:
                        if line.strip() == "":
                            continue
                        else:
                            pre_policy += translate_text(line, lang) + "\n"
                            time.sleep(0.2)

                    translated_policy = pre_policy.strip()
                    
                    # Translate transcript
                    print("  Translating transcript...")
                    translated_transcript = translate_transcript(str(row.get('transcript', '')), lang)
                    
                    # Create new row
                    new_row = row.to_dict()
                    new_row['policy'] = translated_policy
                    new_row['transcript'] = translated_transcript
                    new_row['language'] = lang
                    new_row['row_id'] = f"{original_row_id}_{lang.lower()}"

                    append_row_to_csv(new_row, output_file)
                    written_rows += 1
                    
                    completed.add((original_row_id, lang))
                    save_checkpoint(
                        checkpoint_file,
                        completed,
                        total=total_translations,
                        written=written_rows,
                        errors=errors[-50:],
                    )
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    errors.append({
                        'row_id': str(original_row_id),
                        'lang': str(lang),
                        'error': str(e),
                    })
                    save_checkpoint(
                        checkpoint_file,
                        completed,
                        total=total_translations,
                        written=written_rows,
                        errors=errors[-50:],
                    )
                    continue
            
            pbar.update(1)
    
    print(f"\n=== Summary ===")
    print(f"Input rows: {len(df)}")
    print(f"Output rows written: {written_rows}")
    print(f"Translation completed!")

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Translate policy and transcript data")
    parser.add_argument("--input_file", type=str,
                        default="african_guardrail_test_cases_Afri_guard_all_transcripts_prompts.csv",
                        help="Input CSV file")
    parser.add_argument("--output_file", type=str,
                        default="policy_only_african_guardrail_test_cases_Afri_guard_all_transcripts_prompts_translated2.csv",
                        help="Output CSV file")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file '{args.input_file}' does not exist")
    
    process_dataset(args.input_file, args.output_file, args.resume)

if __name__ == "__main__":
    main()