import argparse
import math
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import tiktoken
import torch
import wandb
from constants import GPT_CONFIG_124M
from src.dataloader import create_dataloader
from src.model import GPTModel
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def read_text_file(file_path: str) -> str:
    """Read text data from a file.

    Parameters
    ----------
    file_path : str
        Path to the text file to read.

    Returns
    -------
    str
        The contents of the text file as a string.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def text_2_token_ids(text: str, tokenizer: Any) -> Tensor:
    """Convert text to token IDs using the provided tokenizer.

    Parameters
    ----------
    text : str
        Input text to tokenize.
    tokenizer : Any
        Tokenizer object that implements encode method.

    Returns
    -------
    Tensor
        Tensor of token IDs with shape (1, sequence_length).
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_2_text(ids: Tensor, tokenizer: Any) -> str:
    """Convert token IDs back to text using the provided tokenizer.

    Parameters
    ----------
    ids : Tensor
        Tensor of token IDs to decode.
    tokenizer : Any
        Tokenizer object that implements decode method.

    Returns
    -------
    str
        Decoded text from the token IDs.
    """
    decoded = tokenizer.decode(ids.squeeze(0).tolist())
    return decoded


def generate_text(
    model: Any,
    ids: Tensor,
    context_size: int,
    num_steps: int = 30,
    temperature: float = 0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> Tensor:
    """Generate text from input token IDs using the provided model.

    Parameters
    ----------
    model : Any
        The language model to use for generation.
    ids : Tensor
        Initial token IDs to start generation from.
    context_size : int
        Maximum context length for the model.
    num_steps : int, optional
        Number of tokens to generate, by default 30.
    temperature : float, optional
        Temperature parameter for sampling, by default 0.
    top_k : Optional[int], optional
        Top-k sampling parameter, by default None.
    top_p : Optional[float], optional
        Top-p (nucleus) sampling parameter, by default None.

    Returns
    -------
    Tensor
        Generated token IDs including the input context.
    """
    for _ in range(num_steps):
        ids = ids[:, -context_size:]

        with torch.no_grad():
            logits = model(ids)

        logits = logits[:, -1, :]

        if top_k is not None:
            logits_vals = torch.topk(logits, k=top_k, dim=-1).values
            min_val = logits_vals[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            # in case when the most probable token has p > top_p no one token left to predict
            # so the must probable token must always left
            sorted_indices_to_remove[:, 0] = False
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            logits = torch.gather(sorted_logits, -1, sorted_indices.argsort(dim=-1))

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        ids = torch.concat((ids, idx_next), dim=-1)

    return ids


def calc_loss(
    model: Any, dataloader: DataLoader, device: torch.device, loss_fn: Any, eval_num: int
) -> float:
    """Calculate the average loss over a specified number of batches.

    Parameters
    ----------
    model : Any
        The model to evaluate.
    dataloader : DataLoader
        DataLoader containing the evaluation data.
    device : torch.device
        Device to perform computations on.
    loss_fn : Any
        Loss function to use for evaluation.
    eval_num : int
        Number of batches to evaluate.

    Returns
    -------
    float
        Average loss over the evaluated batches.
    """
    i, loss = 0, 0
    for train, test in dataloader:
        train, test = train.to(device), test.to(device)
        logits = model(train)
        input = logits.flatten(0, 1)
        labels = test.flatten()
        loss_ = loss_fn(input, labels)
        loss += loss_.item()
        i += 1
        if i >= eval_num:
            break
    if i == 0:
        return float("inf")
    return loss / i


def evaluate_model(
    model: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: Any,
    eval_num: int,
) -> Tuple[float, float]:
    """Evaluate the model on both training and validation data.

    Parameters
    ----------
    model : Any
        The model to evaluate.
    train_loader : DataLoader
        DataLoader containing training data.
    val_loader : DataLoader
        DataLoader containing validation data.
    device : torch.device
        Device to perform computations on.
    loss_fn : Any
        Loss function to use for evaluation.
    eval_num : int
        Number of batches to evaluate.

    Returns
    -------
    Tuple[float, float]
        Tuple containing (train_loss, val_loss).
    """
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss(model, train_loader, device, loss_fn, eval_num)
        val_loss = calc_loss(model, val_loader, device, loss_fn, eval_num)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    start_context: str,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
) -> None:
    """Generate and print a sample text using the model.

    Parameters
    ----------
    model : Any
        The model to use for generation.
    tokenizer : Any
        Tokenizer for encoding/decoding text.
    device : torch.device
        Device to perform computations on.
    start_context : str
        Initial text to start generation from.
    temperature : float, optional
        Temperature parameter for sampling, by default 1.0.
    top_k : int, optional
        Top-k sampling parameter, by default 50.
    top_p : float, optional
        Top-p (nucleus) sampling parameter, by default 0.95.
    """
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_2_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(
            model=model,
            ids=encoded,
            context_size=context_size,
            num_steps=50,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        decoded_text = token_ids_2_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ").replace("\r", ""))
    model.train()


def train(
    model: Any,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    n_epochs: int,
    lr: float,
    start_context: str,
    eval_num: int = 10,
    global_step: int = -1,
    tokens_seen: int = 0,
    warmup_steps: int = 100,
    model_last_step: int = -1,
    eval_period: int = 100,
    print_sample_period: int = 1000,
    save_ckpt_period: int = 1000,
    model_dir: Union[str, Path] = "model_checkpoints",
) -> Tuple[int, int]:
    """Train the model with the specified parameters.

    Parameters
    ----------
    model : Any
        The model to train.
    optimizer : torch.optim.Optimizer
        Optimizer for training.
    train_loader : DataLoader
        DataLoader containing training data.
    val_loader : DataLoader
        DataLoader containing validation data.
    tokenizer : Any
        Tokenizer for text processing.
    device : torch.device
        Device to perform computations on.
    n_epochs : int
        Number of epochs to train.
    lr : float
        Learning rate.
    start_context : str
        Initial text for text sample generation.
    eval_num : int, optional
        Number of batches to evaluate on, by default 10.
    global_step : int, optional
        Starting global step counter, by default -1.
    tokens_seen : int, optional
        Starting number of token that was seen by model, by default 0.
    warmup_steps : int, optional
        Number of warmup steps, by default 100.
    model_last_step : int, optional
        Last training step of the model, is used to continue training from the last checkpoint, by default -1.
    eval_period : int, optional
        Steps between evaluations, by default 100.
    print_sample_period : int, optional
        Steps between printing text samples, by default 1000.
    save_ckpt_period : int, optional
        Steps between saving checkpoints, by default 1000.
    model_dir : Union[str, Path], optional
        Directory to save model checkpoints, by default "model_checkpoints".

    Returns
    -------
    Tuple[List[float], List[float], List[int]]
        Tuple containing (gloabl_step, tokens_seen).
    """
    loss_fn = CrossEntropyLoss()

    try:
        for _ in range(n_epochs):
            model.train()

            for train_batch, target_batch in train_loader:
                global_step += 1
                if global_step < model_last_step:
                    continue

                train_batch, target_batch = train_batch.to(device), target_batch.to(device)
                optimizer.zero_grad()

                logits = model(train_batch)
                input = logits.flatten(0, 1)
                labels = target_batch.flatten()

                loss = loss_fn(input, labels)
                loss.backward()

                tokens_seen += train_batch.numel()

                # is used to prevent model get shocked by too much value of the gradient
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # set the learning rate fot this iteration
                new_lr = get_lr(global_step, warmup_steps, lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr
                optimizer.step()

                # Log training metrics to wandb
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/learning_rate": new_lr,
                        "train/grad_norm": norm.item(),
                        "train/tokens_seen": tokens_seen,
                        "train/global_step": global_step,
                    }
                )

                if global_step % eval_period == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, loss_fn, eval_num
                    )
                    print(f"Train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")

                    # Log evaluation metrics to wandb
                    wandb.log(
                        {
                            "eval/train_loss": train_loss,
                            "eval/val_loss": val_loss,
                            "eval/tokens_seen": tokens_seen,
                            "eval/global_step": global_step,
                        }
                    )

            if global_step % print_sample_period == 0 and global_step >= model_last_step:
                generate_and_print_sample(model, tokenizer, device, start_context)

            if global_step % save_ckpt_period == 0 and global_step >= model_last_step:
                file_name = model_dir / f"model_pg_{global_step}.pth"
                # Save and load model
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                    },
                    file_name,
                )
                print(f"Saved {file_name}")

    except KeyboardInterrupt:
        file_name = model_dir / f"model_pg_{global_step}.pth"
        # Save and load model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            file_name,
        )
        raise KeyboardInterrupt

    return gloabl_step, tokens_seen


def get_lr(step: int, warmup_steps: int, max_lr: float) -> float:
    """Calculate learning rate based on current step with warmup and cosine decay.

    Parameters
    ----------
    step : int
        Current training step.
    warmup_steps : int
        Number of warmup steps.
    max_lr : float
        Maximum learning rate.

    Returns
    -------
    float
        Current learning rate.
    """
    min_lr = 0.1 * max_lr
    max_steps = 100 * warmup_steps

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT Model Training Configuration")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="gutenberg_preprocessed",
        help="Directory containing the training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_folder",
        help="Directory where the model checkpoints will be saved",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--print_sample_period",
        type=int,
        default=1000,
        help="Iterations between printing sample outputs",
    )
    parser.add_argument(
        "--eval_freq", type=int, default=100, help="Frequency of evaluations during training"
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay for the optimizer")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for training")
    parser.add_argument(
        "--train_val_ratio",
        type=float,
        default=0.9,
        help="Train / validation datasets ratio (e.g. if 0.9 then train size = 0.9*len(dataset) and val = 0.1*len(dataset))",
    )
    parser.add_argument("--warmup", type=int, default=100, help="Number of warmup training steps")
    parser.add_argument(
        "--wandb_project", type=str, default="gpt-train", help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="beatwad",
        help="Weights & Biases entity (username or team name)",
    )
    parser.add_argument(
        "--wandb_name", type=str, default="wandb_run", help="Weights & Biases run name"
    )

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config={
            "model_config": GPT_CONFIG_124M,
            "learning_rate": args.lr,
            "weight_decay": args.wd,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "train_val_ratio": args.train_val_ratio,
            "warmup_steps": args.warmup,
        },
    )

    # Set device and Pytorch settings
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Using {device}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")

        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:  # Volta (7.0+), Turing (7.5+), Ampere (8.0+), Hopper (9.0+)
            torch.set_float32_matmul_precision("high")
            print("Uses tensor cores")
        else:
            print("Tensor cores not supported on this GPU. Using default precision.")
    print(f"Uses tensor cores: {torch.cuda.is_available()}")

    # Log system info to wandb
    wandb.log(
        {
            "system/pytorch_version": torch.__version__,
            "system/cuda_available": torch.cuda.is_available(),
            "system/cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "system/device": str(device),
            "system/tensor_cores": torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 7,
        }
    )

    # Get settings
    start_context = "Every effort moves you"
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    train_val_ratio = args.train_val_ratio

    context_size = GPT_CONFIG_124M["context_length"]

    # Load and initialize model and optimizer
    model = GPTModel(GPT_CONFIG_124M)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
        fused=True,
    )

    cache_dir = Path(args.output_dir) / "cache"
    model_dir = Path(args.output_dir) / "model_checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Get the latest saved model (if exists)
    model_files = list(model_dir.glob("*.pth"))
    max_step_num = -1

    for f in model_files:
        step_num = str(f).split("_")[-1].split(".")[0]
        if step_num.isdigit():
            max_step_num = max(max_step_num, int(step_num))

    model_name = f"model_pg_{max_step_num}.pth" if max_step_num > 0 else ""

    if model_name:
        print(f"Loading model from {model_dir / model_name}")
        checkpoint = torch.load(model_dir / model_name, map_location=device, weights_only=True)
        state_dict = checkpoint["model_state_dict"]
        # Remove "_orig_mod." prefix from keys
        model_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")
            model_state_dict[new_key] = v

        model.load_state_dict(model_state_dict)

    model.to(device).to(torch.bfloat16)
    model = torch.compile(model)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load the number of last file on which model was trained before
    data_dir = args.data_dir
    all_files = [
        os.path.join(path, name)
        for path, subdirs, files in os.walk(data_dir)
        for name in files
        if name.endswith((".txt"))
    ]
    total_files = len(all_files)

    if total_files == 0:
        print("No training text files found. Make sure you " "selected the correct input directory")
        quit()
    print("Total files:", total_files)
    all_files.sort()

    gloabl_step = -1
    tokens_seen = 0
    # Iterate over the books in the training corpus
    for index, file_path in tqdm(enumerate(all_files)):
        text_data = read_text_file(file_path) + " <|endoftext|> "
        print(f"Tokenizing file {index} of {total_files}: {file_path}")

        # Split on train and val dataloaders
        split_idx = int(len(text_data) * train_val_ratio)
        if batch_size > (len(text_data) - split_idx) // context_size:
            continue

        train_loader = create_dataloader(
            text_data[:split_idx],
            tokenizer,
            batch_size,
            max_length=context_size,
            shuffle=True,
            drop_last=True,
            cache_dir=cache_dir,
        )

        val_loader = create_dataloader(
            text_data[split_idx:],
            tokenizer,
            batch_size,
            max_length=context_size,
            shuffle=False,
            drop_last=False,
            cache_dir=cache_dir,
        )

        # Train model on file
        tokens_seen, gloabl_step = train(
            model,
            optimizer,
            train_loader,
            val_loader,
            tokenizer,
            device,
            n_epochs,
            args.lr,
            start_context,
            eval_num=10,
            global_step=gloabl_step,
            warmup_steps=args.warmup,
            model_last_step=max_step_num,
            eval_period=args.eval_freq,
            print_sample_period=args.print_sample_period,
            save_ckpt_period=1000,
            model_dir=model_dir,
        )
