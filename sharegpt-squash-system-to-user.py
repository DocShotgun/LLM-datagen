import argparse
import pathlib

from datasets import Dataset, load_dataset


def process(data):
    for message in data["conversations"]:
        if message["from"] == "system":
            message["from"] = "human"

    convo = []
    prev_message = None

    for message in data["conversations"]:
        if prev_message and prev_message["from"] == message["from"]:
            prev_message["value"] = (
                prev_message["value"].strip() + "\n\n" + message["value"].strip()
            )
        else:
            convo.append(message)
            prev_message = message

    data["conversations"] = convo
    return data


parser = argparse.ArgumentParser(description="ShareGPT dataset shortener")
parser.add_argument(
    "-d",
    "--datafile",
    type=str,
    help="Dataset file in ShareGPT format, accepts .json/.jsonl/.parquet",
)
args = parser.parse_args()

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

dataset.to_json(f"{file.stem}-system-to-user-squashed.jsonl")
