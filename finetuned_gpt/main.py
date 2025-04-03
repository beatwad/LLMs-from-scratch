import os
import urllib.request

import tiktoken
import torch
from safetensors.torch import load_file
from src.config import BASE_CONFIG, CHOOSE_MODEL, model_configs
from src.model import GPTModel, load_weights_into_gpt

from finetuned_gpt.src.utils import generate, text_to_token_ids, token_ids_to_text

if __name__ == "__main__":
    torch.manual_seed(2042025)

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    URL_DIR = {
        "gpt2-small (124M)": "gpt2",  # works ok
        "gpt2-medium (355M)": "gpt2-medium",  # this file seems to have issues via `generate`
        "gpt2-large (774M)": "gpt2-large",  # works ok
        "gpt2-xl (1558M)": "gpt2-xl",  # works ok
    }

    url = f"https://huggingface.co/openai-community/{URL_DIR[CHOOSE_MODEL]}/resolve/main/model.safetensors"
    output_file = f"model-{URL_DIR[CHOOSE_MODEL]}.safetensors"

    # Download file
    if not os.path.exists(output_file):
        print(f"Downloading {url} to {output_file}")
        urllib.request.urlretrieve(url, output_file)

    # Load weights
    print("Loading weights")
    state_dict = load_file(output_file)
    gpt = GPTModel(BASE_CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading weights into GPT")
    load_weights_into_gpt(gpt, state_dict)
    print("Moving to device")
    gpt.to(device)

    print("Getting tokenizer")
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=gpt.to(device),
        idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
        max_new_tokens=30,
        context_size=BASE_CONFIG["context_length"],
        top_k=1,
        temperature=1.0,
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
