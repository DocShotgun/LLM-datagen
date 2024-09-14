import argparse
import pathlib

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def process(data):
    convo = []
    newconvo = []
    for message in data["conversations"]:
        if message["from"] == "system":
            convo.append({"role": "system", "content": message["value"].strip()})
        elif message["from"] == "human":
            convo.append({"role": "user", "content": message["value"].strip()})
        elif message["from"] == "gpt":
            convo.append({"role": "assistant", "content": message["value"].strip()})

    while len(tokenizer.apply_chat_template(convo)) > args.length:
        del convo[-1]
        if len(convo) < 1:
            return {"conversations": []}
    while convo[-1].get("role") != "assistant":
        del convo[-1]
        if len(convo) < 1:
            return {"conversations": []}

    for message in convo:
        if message.get("role") == "system":
            newconvo.append({"from": "system", "value": message.get("content").strip()})
        elif message.get("role") == "user":
            newconvo.append({"from": "human", "value": message.get("content").strip()})
        elif message.get("role") == "assistant":
            newconvo.append({"from": "gpt", "value": message.get("content").strip()})

    data["conversations"] = newconvo
    return data


parser = argparse.ArgumentParser(description="ShareGPT dataset shortener")
parser.add_argument(
    "-d",
    "--datafile",
    type=str,
    help="Dataset file in ShareGPT format, accepts .json/.jsonl/.parquet",
)
parser.add_argument(
    "-t",
    "--tokenizer",
    type=str,
    help="Directory containing the model's tokenizer and tokenizer_config.json",
)
parser.add_argument(
    "-l",
    "--length",
    type=int,
    help="Context length to limit conversations to",
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

file = pathlib.Path(args.datafile)
datatype = None
if file.name.endswith(".json") or file.name.endswith(".jsonl"):
    datatype = "json"
elif file.name.endswith(".parquet"):
    datatype = "parquet"

# Load and process dataset
dataset = load_dataset(datatype, data_files=str(file))
dataset = dataset.map(process)

filtered_data = []
for row in dataset["train"]:
    if len(row.get("conversations")) > 1:
        filtered_data.append(row)
dataset = Dataset.from_list(filtered_data)

dataset.to_json(f"{file.stem}-shortened.jsonl")
