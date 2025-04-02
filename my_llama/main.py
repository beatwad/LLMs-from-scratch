import os
import torch
from src.config import LLAMA32_CONFIG
from src.model import Llama3Model
from src.model import load_weights_into_llama
from src.tokenizer import Tokenizer, ChatFormat, text_to_token_ids, token_ids_to_text, generate
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


if __name__ == "__main__":
    
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Reduce the context length so the model would work fine on weaker GPUs
    old_context_length = LLAMA32_CONFIG["context_length"]
    LLAMA32_CONFIG["context_length"] = 8192

    def rescale_theta(theta_old, context_length_old, context_length_new):
        scaling_factor = context_length_new / context_length_old
        theta_new = theta_old * scaling_factor
        return theta_new

    LLAMA32_CONFIG["rope_base"] = rescale_theta(
        LLAMA32_CONFIG["rope_base"],
        old_context_length,
        LLAMA32_CONFIG["context_length"]
    )

    LLAMA_SIZE_STR = "1B" if LLAMA32_CONFIG["emb_dim"] == 2048 else "3B"

    print("New RoPE theta:", LLAMA32_CONFIG["rope_base"])

    # load model
    model = Llama3Model(LLAMA32_CONFIG)
    
    # load tokenizer
    tokenizer_file_path = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
        filename="original/tokenizer.model",
        local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    )

    # load weights
    if LLAMA_SIZE_STR == "1B":
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
            filename="model.safetensors",
            local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
        )
        combined_weights = load_file(weights_file)
    else:
        combined_weights = {}
        for i in range(1, 3):
            weights_file = hf_hub_download(
                repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
                filename=f"model-0000{i}-of-00002.safetensors",
                local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
            )
            current_weights = load_file(weights_file)
            combined_weights.update(current_weights)

    # load weights into model
    load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)
    model.to(device)
    del combined_weights

    # generate text
    PROMPT = "What do llamas eat?"
    tokenizer = Tokenizer(tokenizer_file_path)
    chat_tokenizer = ChatFormat(tokenizer)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(PROMPT, chat_tokenizer).to(device),
        max_new_tokens=150,
        context_size=LLAMA32_CONFIG["context_length"],
        top_k=1,
        temperature=0.
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    print(output_text)
