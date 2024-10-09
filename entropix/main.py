import math
from pathlib import Path

import jax
import jax.numpy as jnp
import tyro
import numpy as np

from entropix.config import LLAMA_1B_PARAMS
from entropix.kvcache import KVCache
from entropix.model import xfmr
from entropix.sampler import SamplerConfig, sample
from entropix.prompts import create_prompts_from_csv, prompt
from entropix.sampler import sample
from entropix.tokenizer import Tokenizer
from entropix.utils import stable_softmax
from entropix.weights import load_weights

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / '../weights'

@jax.jit
def stable_softmax(x, axis=-1):
    """Numerically stable softmax."""
    shifted_x = x - jax.lax.stop_gradient(x.max(axis=axis, keepdims=True))
    return jax.nn.softmax(shifted_x, axis=axis)

def apply_scaling(freqs: jax.Array):
  SCALE_FACTOR = 8
  LOW_FREQ_FACTOR = 1
  HIGH_FREQ_FACTOR = 4
  OLD_CONTEXT_LEN = 8192  # original llama3 length

  low_freq_wavelen = OLD_CONTEXT_LEN / LOW_FREQ_FACTOR
  high_freq_wavelen = OLD_CONTEXT_LEN / HIGH_FREQ_FACTOR

  def scale_freq(freq):
    wavelen = 2 * math.pi / freq

    def scale_mid(_):
      smooth = (OLD_CONTEXT_LEN / wavelen - LOW_FREQ_FACTOR) / (HIGH_FREQ_FACTOR - LOW_FREQ_FACTOR)
      return (1 - smooth) * freq / SCALE_FACTOR + smooth * freq

    return jax.lax.cond(
      wavelen < high_freq_wavelen,
      lambda _: freq,
      lambda _: jax.lax.cond(wavelen > low_freq_wavelen, lambda _: freq / SCALE_FACTOR, scale_mid, None),
      None
    )

  return jax.vmap(scale_freq)(freqs)


def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, use_scaled: bool = False, dtype: jnp.dtype = jnp.float32) -> jax.Array:
  freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(dtype) / dim))
  if use_scaled:
    freqs = apply_scaling(freqs)
  t = jnp.arange(end, dtype=dtype)
  freqs = jnp.outer(t, freqs)
  return jnp.exp(1j * freqs)


def build_attn_mask(seqlen: int, start_pos: int) -> jax.Array:
  mask = jnp.zeros((seqlen, seqlen), dtype=jnp.float32)
  if seqlen > 1:
    mask = jnp.full((seqlen, seqlen), float('-inf'))
    mask = jnp.triu(mask, k=1)
    mask = jnp.hstack([jnp.zeros((seqlen, start_pos)), mask], dtype=jnp.float32)
  return mask


def main(weights_path: Path = DEFAULT_WEIGHTS_PATH.joinpath('1B-Instruct'), max_seq_len: int = 2048): # Reduced default max_seq_len
    model_params = LLAMA_1B_PARAMS
    # Enforce static shapes for weights
    xfmr_weights = load_weights(weights_path.absolute())


    tokenizer = Tokenizer('entropix/tokenizer.model')

    # Create the batch of tokens
    def generate(xfmr_weights, model_params, tokens, key): # Added PRNG key
        gen_tokens = None
        cur_pos = 0
        tokens = jnp.array([tokens], jnp.int32)
        bsz, seqlen = tokens.shape

        # Precompute freqs_cis outside the loop and slice as needed
        freqs_cis = precompute_freqs_cis(model_params.head_dim, max_seq_len + 1, model_params.rope_theta, model_params.use_scaled_rope)

        kvcache = KVCache.new(model_params.n_layers, bsz, max_seq_len, model_params.n_local_kv_heads, model_params.head_dim)

        attn_mask = build_attn_mask(seqlen, cur_pos) # Move mask creation here

        logits, kvcache, _, _ = xfmr(xfmr_weights, model_params, tokens, cur_pos, freqs_cis[:seqlen], kvcache, attn_mask=attn_mask)

        # Check for NaN in logits
        if np.isnan(logits).any():
            raise RuntimeError("NaN values encountered in logits.")

        next_token = jnp.argmax(logits[:, -1], axis=-1, keepdims=True).astype(jnp.int32)
        gen_tokens = next_token
        print(tokenizer.decode([next_token.item()]), end='', flush=True)
        cur_pos = seqlen
        stop = jnp.array([128001, 128008, 128009])
        sampler_cfg = SamplerConfig()
        while cur_pos < max_seq_len: # Use max_seq_len argument
            cur_pos += 1
            logits, kvcache, scores, stats = xfmr(xfmr_weights, model_params, next_token, cur_pos, freqs_cis[cur_pos:cur_pos+1], kvcache)

            # Check for NaN in logits and scores
            if np.isnan(logits).any() or np.isnan(scores).any():
                raise RuntimeError("NaN values encountered in logits or scores.")

            next_token, key = sample(gen_tokens, logits, scores, cfg=sampler_cfg, key=key) # Pass and update key
            gen_tokens = jnp.concatenate((gen_tokens, next_token))
            print(tokenizer.decode(next_token.tolist()[0]), end='', flush=True)
            if jnp.isin(next_token, stop).any():
                break
        return gen_tokens, key # Return updated key

    csv_path = Path('entropix/data/prompts.csv')
    prompts = create_prompts_from_csv(csv_path)
    PROMPT_TEST = False

    key = jax.random.PRNGKey(0) # Initialize PRNG key

    if PROMPT_TEST:
        for p in prompts:
            print(p)
            tokens = tokenizer.encode(p,  bos=False, eos=False, allowed_special='all')
            _, key = generate(xfmr_weights, model_params, tokens, key) # Update key after each generation
    else:
        print(prompt)
        tokens = tokenizer.encode(prompt,  bos=False, eos=False, allowed_special='all')
        _, key = generate(xfmr_weights, model_params, tokens, key) # Update key after generation

if __name__ == '__main__':
  tyro.cli(main)
