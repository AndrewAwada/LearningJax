import jax.numpy as jnp
import jax
from jax import random
import equinox as eqx
import equinox.nn as nn
import optax
from jax import vmap, jit, grad
from typing import List
from functools import reduce

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 128
n_head = 4
n_layer = 4
dropout_rate = 0.2
# ------------

# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# This is our very simple tokenizer
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Now encode entire dataset and store it in a jax array
data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

key = jax.random.PRNGKey(42)

def get_batch(split, key):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class Head(eqx.Module):
    key_: nn.Linear
    query: nn.Linear
    value: nn.Linear
    tril: jnp.array = eqx.field(static=True)
    dropout: nn.Dropout

    def __init__(self, n_embd, head_size, key):
        key_q, key_v = random.split(key)

        self.key_ = nn.Linear(n_embd, head_size, use_bias=False, key=key)
        self.query = nn.Linear(n_embd, head_size, use_bias=False, key=key_q)
        self.value = nn.Linear(n_embd, head_size, use_bias=False, key=key_v)
        self.tril = jnp.tril(jnp.ones((block_size, block_size)))
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, x, enable_dropout=False, key=None):
        B, T, C = x.shape

        k = vmap(vmap(self.key_))(x)  # (B, T, head_size)
        q = vmap(vmap(self.query))(x)  # (B, T, head_size)

        # compute attention scores (affinities)
        wei = q @ jnp.transpose(k, axes=(0, 2, 1)) * C**-0.5  # (B, T, T)

        wei = jnp.where(self.tril[:T, :T] == 0, -jnp.inf, wei)
        wei = jax.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei, inference=not enable_dropout, key=key)

        # Perform the weighted aggregation of the values
        v = vmap(vmap(self.value))(x)  # (B, T, head_size)
        out = wei @ v

        return out


class MultiHeadAttention(eqx.Module):
    # Multiple heads of self attention in parallel
    heads: List[Head]
    projection: nn.Linear
    dropout: nn.Dropout

    def __init__(self, num_heads, head_size, key, n_embd=n_embd):
        key_l, key_p = jax.random.split(key)
        layer_keys = jax.random.split(key_l, num=num_heads)
        self.heads = [Head(n_embd, head_size, layer_key) for layer_key in layer_keys]
        self.projection = nn.Linear(n_embd, n_embd, key=key_p)
        self.dropout = nn.Dropout(dropout_rate)

    def __call__(self, x, enable_dropout=False, key=None):
        if enable_dropout:
            key_h, key_d = random.split(key)
            dropout_keys = random.split(key_d, num=len(self.heads))
        else:
            dropout_keys = [None] * len(self.heads)
            key_h = None

        out = jnp.concat([h(x, enable_dropout, k) for h, k in zip(self.heads, dropout_keys)], axis=-1)
        out = vmap(vmap(self.projection))(out)
        out = self.dropout(out, inference=not enable_dropout, key=key_h)
        return out
    

class FeedForward(eqx.Module):
    # A simple linear layer followed by non-linearity
    mlp: nn.Linear
    projection: nn.Linear
    dropout: nn.Dropout

    def __init__(self, n_embd, key):
        mlp_key, key_p = random.split(key)
        self.mlp = nn.Linear(n_embd, 4 * n_embd, key=mlp_key)
        self.projection = nn.Linear(4 * n_embd, n_embd, key=key_p)
        self.dropout = nn.Dropout(dropout_rate)
        
    def __call__(self, x, enable_dropout=False, key=None):
        hidden = vmap(vmap(self.mlp))(x)
        hidden = jax.nn.relu(hidden)
        out = vmap(vmap(self.projection))(hidden)
        out = self.dropout(out, inference=not enable_dropout, key=key)
        return out


class Block(eqx.Module):
    # Transformer block: communication followed by computation
    self_attention: MultiHeadAttention
    feed_forward: FeedForward
    ln1: nn.LayerNorm
    ln2: nn.LayerNorm

    def __init__(self, n_embd, n_head, key):
        # n_embd: embedding dimension, n_head: number of heads we want
        key_a, key_ffwd = random.split(key)
        head_size = n_embd//n_head

        self.self_attention = MultiHeadAttention(n_head, head_size, key_a)
        self.feed_forward = FeedForward(n_embd, key_ffwd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def __call__(self, x, enable_dropout=False, key=None):
        if enable_dropout:
            key_attention, key_ffwd = random.split(key)
        else:
            key_attention = None
            key_ffwd = None

        # note: adding x as a skip connection
        x = x + self.self_attention(vmap(vmap(self.ln1))(x), enable_dropout, key_attention)
        x = x + self.feed_forward(vmap(vmap(self.ln2))(x), enable_dropout, key_ffwd)
        return x


class BigramLanguageModel(eqx.Module):
    token_embedding_table: nn.Embedding
    position_embedding_table: nn.Embedding
    # self_attention_heads: MultiHeadAttention
    # feed_forward: FeedForward
    blocks: List[Block]
    ln: nn.LayerNorm
    lm_head: nn.Linear

    def __init__(self, vocab_size, n_embd, key):
        key_emb, key_pos, key_lm, key_b = random.split(key, num=4)
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, key=key_emb)
        self.position_embedding_table = nn.Embedding(vocab_size, n_embd, key=key_pos)

        # self.self_attention_heads = MultiHeadAttention(4, n_embd//4, key)
        # self.feed_forward = FeedForward(n_embd, key)
        n_block = n_layer
        key_blocks = random.split(key_b, num=n_block)
        self.blocks = [Block(n_embd, n_head=n_head, key=key_block) for key_block in key_blocks]
        self.ln = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, key=key_lm)

    def __call__(self, idx, enable_dropout=False, key=None):
        # idx is a (B,T) tensor of integers
        B, T = idx.shape

        tok_emb = vmap(vmap(self.token_embedding_table))(idx)  # (B, T, n_embd)
        pos_emb = vmap(self.position_embedding_table)(jnp.arange(T))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        # x = self.self_attention_heads(x)  # apply to one of the self attention heads. (B, T, C)
        # x = vmap(self.feed_forward)(x)
        if enable_dropout:
            key_blocks = random.split(key, len(self.blocks))
        else:
            key_blocks = [None] * len(self.blocks)
        x = reduce(lambda acc, layer: layer[0](acc, enable_dropout, layer[1]), zip(self.blocks, key_blocks), x)
        x = vmap(vmap(self.ln))(x)
        logits = vmap(vmap(self.lm_head))(x)  # (B, T, vocab_size)
        return logits
    
    def generate(self, key, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Generate new random key
            key, _ = random.split(key)
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self(idx_cond)
            #focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = jax.nn.softmax(logits, axis=-1) # (B, C)
            # sample from the distribution
            def jax_multinomial(probabilities):
                return jax.random.choice(key, a=vocab_size, shape=(1, ), p=probabilities)
            idx_next = vmap(jax_multinomial)(probs)
            # idx_next = torch.multinomial(torch.tensor(probs.tolist()), num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = jnp.concat((idx, idx_next), axis=1) # (B, T+1)
        return idx

@eqx.filter_value_and_grad
def compute_loss(model, idx, targets):
    logits = model(idx, True, key)
    B, T, C = logits.shape
    logits = jnp.reshape(logits, (B*T, C))
    targets = jnp.reshape(targets, (B*T))
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
    return loss

def compute_loss_slow(model, idx, targets):
    logits = model(idx)
    B, T, C = logits.shape
    logits = jnp.reshape(logits, (B*T, C))
    targets = jnp.reshape(targets, (B*T))
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, targets))
    return loss

m = BigramLanguageModel(vocab_size, n_embd, key)

optim = optax.adamw(learning_rate=learning_rate)
opt_state = optim.init(m)

@eqx.filter_jit
def make_step(model, x, y, opt_state):
    loss, grads = compute_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state, m)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state

def estimate_loss(key):
    out = {}
    for split in ['train', 'val']:
        losses = []
        for k in range(eval_iters):
            X, Y = get_batch(split, key)
            loss = compute_loss_slow(m, X, Y)
            losses.append(loss)
        out[split] = jnp.mean(jnp.array(losses))
    return out

for i in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    key, _ = random.split(key)
    if i % eval_interval == 0:
        losses = estimate_loss(key)
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train', key)

    # evaluate the loss and update model params
    loss, m, opt_state = make_step(m, xb, yb, opt_state)

print(f"Final Training Loss: {loss.item()}\n")

print(decode(m.generate(key, idx = jnp.zeros((1, 1), dtype=jnp.int32), max_new_tokens=500)[0].tolist()))
