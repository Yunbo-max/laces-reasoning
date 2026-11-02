# S0 Readability Probe

Locates which stage of the LACES encoding chain erases an answer-deciding symbol.
For paired samples differing only in one decisive token (digit, operator, operand
order, `<` vs `<=`), a linear probe reads the symbol from (a) mean-pooled RWKV hidden
(2560-dim) and (b) the S0 32-dim latent. Templates are split train/test so the probe
cannot memorize surface wording.

Frozen S0/RWKV; 2.9B step-30k checkpoint, single H100. Companion to
`experiments/s0_info_fidelity` (cosine separability) — this one measures *readability*.

## Result (linear-probe test accuracy, train/test template split)

| task | pooled (2560-dim) | z32 (32-dim latent) | chance |
| --- | ---: | ---: | ---: |
| digit (8-way) | 0.859 | 0.063 | 0.125 |
| operator (3-way) | 0.958 | 0.292 | 0.333 |
| order (2-way) | 0.938 | 0.563 | 0.500 |
| boundary `<` vs `<=` (2-way) | 0.938 | 0.250 | 0.500 |

## Conclusion

- **mean pooling is NOT the culprit.** The decisive symbol is still linearly decodable
  from the mean-pooled 2560-dim hidden (86%-96% test accuracy across templates).
- **The 2560 -> 32 projection is where the symbol dies.** At the 32-dim latent, digit and
  boundary drop *below chance* (6.3% vs 12.5%, 25% vs 50%), i.e. the projection not only
  loses the signal but inverts it.

Per the decision rule (pooled reads well but 32-dim fails), the fix priority is the
**32-dim projection / S0 training objective** (widen the latent, or add a key-symbol
reconstruction objective), NOT adding positional latent slots.
