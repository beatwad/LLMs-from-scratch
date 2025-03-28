from calendar import c
from typing import Any, List
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from model import GPTModel, GPT_CONFIG_124M
from dataloader import create_dataloader_v1

import matplotlib.pyplot as plt
import tiktoken


torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text_2_token_ids(text: str, tokenizer: Any) -> torch.tensor:
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_2_text(ids: torch.tensor, tokenizer: Any) -> str:
    decoded = tokenizer.decode(ids.squeeze(0).tolist())
    return decoded

def generate_text(
        model: Any, 
        ids: List[int], 
        context_size: int, 
        num_steps: int = 30, 
        temperature: float = 0,
        top_k: int = None,
        top_p: int = None,
        ) -> List[int]:
    
    for _ in range(num_steps):
        ids = ids[:, -context_size:]
        
        with torch.no_grad():
            logits = model(ids)
        
        logits = logits[:, -1, :]
        
        if top_k is not None:
            logits_vals = torch.topk(logits, k=top_k, dim=-1).values
            min_val = logits_vals[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits)
            
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

def calc_loss(model, dataloader, loss_fn, eval_num) -> float:
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

def evaluate_model(model, train_loader, val_loader, loss_fn, eval_num) -> float:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss(model, train_loader, loss_fn, eval_num)
        val_loss = calc_loss(model, val_loader, loss_fn, eval_num)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(
        model, 
        tokenizer, 
        device, 
        start_context, 
        temperature=1.0, 
        top_k=50, 
        top_p=0.95,
        ):
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
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def train(
        model: Any, 
        train_loader: Any, 
        val_loader: Any, 
        n_epochs: int,
        start_context: str,
        tokenizer: str,
        eval_period: int = 10,
        eval_num: int = 10,
        verbose: bool = False,
        ):
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    loss_fn = CrossEntropyLoss()

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1
    
    for _ in range(n_epochs):
        model.train()

        for train_batch, target_batch in train_loader:
            train_batch, target_batch = train_batch.to(device), target_batch.to(device)
            optimizer.zero_grad()
            
            logits = model(train_batch)
            input = logits.flatten(0, 1)
            labels = target_batch.flatten()
            
            loss = loss_fn(input, labels)
            loss.backward()
            
            optimizer.step()
        
            tokens_seen += train_batch.numel()
            global_step += 1
            
            if global_step % eval_period == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, loss_fn, eval_num)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Train loss: {train_loss:.4f}, val loss: {val_loss:.4f}")
           
        if verbose:
            generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


if __name__ == "__main__":
    TRAIN_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1,
        "train_test_ratio": 0.9,
    }

    model = GPTModel(GPT_CONFIG_124M)

    tokenizer = tiktoken.get_encoding("gpt2")
    with open("the-verdict.txt") as f:
        text_data = f.read()

    n_epochs = TRAIN_SETTINGS["num_epochs"]
    batch_size = TRAIN_SETTINGS["batch_size"]
    train_test_ratio = TRAIN_SETTINGS["train_test_ratio"]
    
    start_context = "Every effort moves you"
    context_size = GPT_CONFIG_124M["context_length"]

    split_idx = int(len(text_data) * train_test_ratio)
    train_loader = create_dataloader_v1(
        text_data[:split_idx], 
        tokenizer, 
        batch_size, 
        max_length=context_size,
        shuffle=True, 
        drop_last=True,
        )
    
    val_loader = create_dataloader_v1(
        text_data[split_idx:], 
        tokenizer, 
        batch_size, 
        max_length=context_size,
        shuffle=False, 
        drop_last=False,
        )

    train_losses, val_losses, tokens_seen = train(
        model, 
        train_loader, 
        val_loader, 
        n_epochs, 
        start_context, 
        tokenizer, 
        verbose=True,
        )

    epochs_tensor = torch.linspace(0, TRAIN_SETTINGS["num_epochs"], len(train_losses))
    # import code; code.interact(local=locals())
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")
