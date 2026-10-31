# S0 Information-Fidelity Chain Probe

Locates where, along the LACES encoding chain, an answer-deciding token loses its
signal. For paired samples that differ in exactly one decisive token (a digit, operand
order, or `<` vs `<=`), it measures separability `sep = 1 - cos` at each stage:

    full RWKV hidden  ->  mean-pooled hidden  ->  S0 mu (32-dim)

Read-only probe; S0/RWKV frozen. Results below are from the 2.9B step-30k checkpoint
(`laces-2.9B-dynlowrank-r32-s0frozen-joint-b4-50k-fla03/step_00030000`), run on a single H100.

## Result

| pair | full hidden | mean-pooled | S0 mu (32-dim) |
| --- | ---: | ---: | ---: |
| lt->le (math) | 1.4e-1 | 3.4e-3 | 2.0e-5 |
| a-b -> b-a | 3.8e-2 | 1.1e-3 | 1.7e-5 |
| digit 42->43 | 9.6e-3 | 4.8e-4 | 2.5e-5 |
| add->sub | 1.1e-2 | 8.5e-4 | 2.2e-5 |
| code digit 16->18 | 1.6e-2 | 2.2e-4 | 2.4e-5 |

(separability = 1 - cosine; higher = more separable)

## Conclusion

Both stages drop the decisive-token signal, multiplicatively:

- **mean pooling** keeps only 2%-7% of the full-hidden separability (13-43x loss),
  consistent with the `/L` dilution of a single-token delta.
- **32-dim projection** keeps only 0.6%-5% of the pooled separability (20-160x loss).
- After both, `mu` sits at `cos ~ 0.99998`, i.e. `sep ~ 1e-5` — the encoder no longer
  distinguishes samples whose answer differs.

This explains the measured GSM8K collapse (raw RWKV 19.0% vs LACES singlez 0.85%): the
token that decides the answer is effectively erased before S1/S2 ever see it.

## Implication

A single mean vector is an information hole for exact-symbol tasks. The proposed fix is to
keep a positional axis inside the per-segment plan (2-4 positional latent slots, as in
Cola-DLM's 32x16 layout) plus a key-symbol reconstruction objective, rather than replacing
the VAE outright. Re-running this probe after the change should lift `sep_mu` from ~1e-5
back toward ~1e-2.
