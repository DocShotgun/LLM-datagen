import argparse
import json
import pathlib
import time
from functools import lru_cache

import httpx
from datasets import Dataset, load_dataset


def infer_completion(prompt, genparams: dict):
    """Generate response from API using completions endpoint"""
    while True:
        r = httpx.stream(
            "POST",
            f"{API_URL}/v1/completions",
            headers={"authorization": API_KEY, "x-api-key": API_KEY},
            json={"prompt": prompt, "stream": True, "model": MODEL, **genparams},
            timeout=None,
        )
        with r as r:
            if r.status_code == 200:
                generated_text = ""
                for chunk in r.iter_lines():
                    if chunk.startswith("data: "):
                        chunk = chunk.replace("data: ", "")
                        if chunk == "[DONE]":
                            break
                        chunk_data = json.loads(chunk)
                        if "choices" in chunk_data:
                            word = chunk_data["choices"][0]["text"]
                            generated_text += word
                return generated_text
            else:
                time.sleep(5)


def infer_chat_completion(messages, genparams: dict):
    """Generate response from API using chat completions endpoint"""
    while True:
        r = httpx.stream(
            "POST",
            f"{API_URL}/v1/chat/completions",
            headers={"authorization": API_KEY, "x-api-key": API_KEY},
            json={
                "messages": messages,
                "stream": True,
                "model": MODEL,
                "add_generation_prompt": True,
                **genparams,
            },
            timeout=None,
        )
        with r as r:
            if r.status_code == 200:
                generated_text = ""
                for chunk in r.iter_lines():
                    if chunk.startswith("data: "):
                        chunk = chunk.replace("data: ", "")
                        chunk_data = json.loads(chunk)
                        if "choices" in chunk_data:
                            if chunk_data["choices"][0]["finish_reason"]:
                                break
                            word = chunk_data["choices"][0]["delta"]["content"]
                            generated_text += word
                return generated_text
            else:
                time.sleep(5)


def format_prompt_jinja(
    messages, template: str, add_generation_prompt: bool, special_tokens: dict
):
    """Format prompt using Jinja2 template"""
    compiled_template = _compile_template(template)
    return compiled_template.render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        **special_tokens,
    )


# Inspired from
# https://github.com/huggingface/transformers/blob/main/src/transformers/tokenization_utils_base.py#L1761
@lru_cache
def _compile_template(template: str):
    """Compiles a Jinja2 template"""

    # Exception handler
    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception

    jinja_template = jinja_env.from_string(template)
    return jinja_template


def get_template_from_file(template_path_raw: str):
    """Get a template from a jinja file"""

    template_path = pathlib.Path(template_path_raw)
    if template_path.exists():
        with open(template_path, "r", encoding="utf8") as raw_template:
            return raw_template.read()
    else:
        raise FileNotFoundError(f'Template "{template_path_raw}" not found.')


def process(data):
    """Primary dataset building function"""
    system = ""
    prompt = ""
    chosen = ""
    rejected = ""
    convo = []
    for message in data["conversations"]:
        if message["from"] == "system" and not system:
            system = message["value"].strip()
            convo.append({"role": "system", "content": system})
        elif message["from"] == "human" and not prompt:
            prompt = message["value"].strip()
            convo.append({"role": "user", "content": prompt})
        elif message["from"] == "gpt" and not prompt:
            print("\nWARNING: Conversation does not begin with user turn - skipping.")
            data["system"] = None
            data["prompt"] = None
            data["chosen"] = None
            data["rejected"] = None
            return data
        elif message["from"] == "gpt" and not chosen:
            chosen = message["value"].strip()
            break

    if CHAT_COMPLETION:
        rejected = infer_chat_completion(convo, GEN_PARAMS)
    else:
        if PROMPT_TEMPLATE:
            # We don't need BOS token here because infer_completion already asks the endpoint to add it
            constructed_prompt = format_prompt_jinja(
                convo,
                PROMPT_TEMPLATE,
                True,
                {"eos_token": EOS_TOKEN},
            )
        else:
            # Default Mistral prompt fallback
            if system:
                constructed_prompt = f"[INST] {system}\n\n{prompt} [/INST]"
            else:
                constructed_prompt = f"[INST] {prompt} [/INST]"
        rejected = infer_completion(constructed_prompt, GEN_PARAMS)

    data["system"] = system
    data["prompt"] = prompt
    if args.chosen:
        # Reverse chosen and rejected if requested
        data["chosen"] = rejected
        data["rejected"] = chosen
    else:
        data["chosen"] = chosen
        data["rejected"] = rejected
    return data


# Setup args
script_dir = pathlib.Path(__file__).parent.resolve()
conf_path = script_dir / "config.json"
with open(conf_path, "r") as config_file:
    config = json.load(config_file)
API_URL = config.get("api_url", "http://127.0.0.1:5000")
API_KEY = config.get("api_key", None)
MODEL = config.get("model", "gpt-3.5-turbo")
CHAT_COMPLETION = config.get("chat_completion", False)
EOS_TOKEN = config.get("eos_token", "</s>")
GEN_PARAMS = config.get("gen_params", {})
if GEN_PARAMS.get("ban_eos_token"):
    # Don't use EOS token in jinja2 template if user bans EOS token
    EOS_TOKEN = None

parser = argparse.ArgumentParser(description="ShareGPT to DPO dataset creator")
parser.add_argument(
    "datafile",
    type=str,
    help="Dataset file in ShareGPT format, accepts .json/.jsonl/.parquet",
)
parser.add_argument(
    "-t",
    "--template",
    type=str,
    default=None,
    help="(Optional) Prompt template in Jinja2 format",
)
parser.add_argument(
    "-c",
    "--chosen",
    action="store_true",
    help="Generate 'chosen' response for DPO instead of 'rejected'",
)
args = parser.parse_args()

file = pathlib.Path(args.datafile)
datatype = None
if file.name.endswith(".json") or file.name.endswith(".jsonl"):
    datatype = "json"
elif file.name.endswith(".parquet"):
    datatype = "parquet"

PROMPT_TEMPLATE = None
if args.template:
    try:
        from jinja2 import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        template_file = pathlib.Path(args.template)
        PROMPT_TEMPLATE = get_template_from_file(template_file)
    except Exception:
        print("jinja2 template not available, using default prompt formatter (Mistral)")


# Load and process dataset
dataset = load_dataset(datatype, data_files=str(file))
dataset = dataset.map(process)
dataset = dataset.select_columns(["system", "prompt", "chosen", "rejected"])

filtered_data = []
for row in dataset["train"]:
    if row.get("prompt"):
        filtered_data.append(row)
dataset = Dataset.from_list(filtered_data)

dataset.to_json(f"{file.stem}-dpo.jsonl")
