# LLM-datagen
Scripts for generating synthetic LLM training data:

- sharegpt-to-dpo:
  - Takes an input dataset in ShareGPT format (supports `json`, `jsonl`, or `parquet`) and uses the first turn of the conversation to generate a DPO dataset consisting of 'chosen' examples (the original data) and 'rejected' examples (the newly generated data from your LLM API endpoint)
  - Designed for [TabbyAPI](https://github.com/theroyallab/tabbyAPI), however should function with any OAI-compatible completion endpoint
  - `-t`: specifies a Jinja2 template to use for formatting the prompt (ignored if using chat completions), otherwise uses the Mistral instruction template by default
  - `-c`: reverses the 'chosen' and 'rejected' samples
- sharegpt-shorten:
  - Takes an input dataset in ShareGPT format (supports `json`, `jsonl`, or `parquet`) and shortens each conversation to a specified token length using a specified tokenizer/chat template
  - `-d`: specifies the path to the dataset
  - `-t`: specifies the path to a Huggingface transformers tokenizer used to process the data; the chat template should be embedded in the `tokenizer_config.json`
  - `-l`: specifies the number of tokens to limit the conversation length to

## Installation:
1. Clone this repository
2. Create or enter your Python venv or conda environment
3. Install the script's requirements with `pip install -r requirements.txt`
4. Make a copy of the script's `config.example.json` named `config.json` and edit settings as appropriate
5. Run the desired script, i.e. `python sharegpt-to-dpo.py <path_to_sharegpt_dataset> -t <path_to_jinja2_prompt_template>`

Jinja2 prompt templates are available at https://github.com/theroyallab/llm-prompt-templates
