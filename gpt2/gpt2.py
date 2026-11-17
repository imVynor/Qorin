import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from torch.utils.data import DataLoader, Dataset


# Model hyperparameters
GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

# Data and training hyperparameters
DATA_PATH = "shakespeare.txt"
TRAIN_RATIO = 0.90
BATCH_SIZE = 2
STRIDE = GPT_CONFIG["context_length"]
NUM_WORKERS = 0
NUM_EPOCHS = 10
LEARNING_RATE = 0.0004
WEIGHT_DECAY = 0.1
EVAL_FREQ = 5
EVAL_ITER = 5
RANDOM_SEED = 123
START_CONTEXT = "Every effort moves you"
CHECKPOINT_PATH = "gpt2_model.pth"
RUN_TRAINING = True


class GPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(
            text,
            allowed_special={"<|endoftext|>"},
        )

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def create_dataloader(text, tokenizer, batch_size, max_length, stride, shuffle, drop_last):
    dataset = GPTDataset(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=NUM_WORKERS,
    )


class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(-2, -1)
        mask = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(mask, float("-inf"))
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5,
            dim=-1,
        )
        attention_weights = self.dropout(attention_weights)

        context = (attention_weights @ values).transpose(1, 2)
        context = context.contiguous().view(batch_size, num_tokens, self.d_out)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            dropout=cfg["drop_rate"],
            num_heads=cfg["n_heads"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.drop_shortcut(self.att(x))
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.drop_shortcut(self.ff(x))
        return x + shortcut


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *(TransformerBlock(cfg) for _ in range(cfg["n_layers"]))
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, input_ids):
        _, sequence_length = input_ids.size()
        token_embeddings = self.token_emb(input_ids)
        position_embeddings = self.pos_emb(
            torch.arange(sequence_length, device=input_ids.device)
        )
        x = self.drop_emb(token_embeddings + position_embeddings)
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")

    num_batches = len(data_loader) if num_batches is None else min(
        num_batches, len(data_loader)
    )
    total_loss = 0.0
    for index, (input_batch, target_batch) in enumerate(data_loader):
        if index >= num_batches:
            break
        total_loss += calc_loss_batch(
            input_batch, target_batch, model, device
        ).item()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=EVAL_ITER
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=EVAL_ITER
        )
    model.train()
    return train_loss, val_loss


def text_to_token_ids(text, tokenizer):
    return torch.tensor([tokenizer.encode(text)], dtype=torch.long)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.squeeze(0).tolist())


def generate_text(model, input_ids, max_length, context_length):
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    for _ in range(max_length):
        input_context = input_ids[:, -context_length:]
        with torch.no_grad():
            logits = model(input_context)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


def generate_and_print_sample(model, tokenizer, device):
    model.eval()
    token_ids = generate_text(
        model,
        text_to_token_ids(START_CONTEXT, tokenizer),
        max_length=50,
        context_length=GPT_CONFIG["context_length"],
    )
    print(token_ids_to_text(token_ids, tokenizer).replace("\n", " "))
    model.train()


def train_model(model, train_loader, val_loader, optimizer, device, tokenizer):
    tokens_seen = 0
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            if global_step % EVAL_FREQ == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device
                )
                print(
                    f"Epoch {epoch + 1} (step {global_step:06d}): "
                    f"train loss {train_loss:.3f}, val loss {val_loss:.3f}"
                )
            global_step += 1

        generate_and_print_sample(model, tokenizer, device)

    return tokens_seen


def main():
    tokenizer = tiktoken.get_encoding("gpt2")
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text_data = file.read()

    split_index = int(TRAIN_RATIO * len(text_data))
    train_loader = create_dataloader(
        text_data[:split_index],
        tokenizer,
        BATCH_SIZE,
        GPT_CONFIG["context_length"],
        STRIDE,
        shuffle=True,
        drop_last=True,
    )
    val_loader = create_dataloader(
        text_data[split_index:],
        tokenizer,
        BATCH_SIZE,
        GPT_CONFIG["context_length"],
        STRIDE,
        shuffle=False,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG).to(device)

    if RUN_TRAINING:
        torch.manual_seed(RANDOM_SEED)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        train_model(model, train_loader, val_loader, optimizer, device, tokenizer)
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f"Saved trained model to {CHECKPOINT_PATH}")
    else:
        model.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        )
        print(f"Loaded model from {CHECKPOINT_PATH}; training skipped.")


if __name__ == "__main__":
    main()
