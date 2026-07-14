# Overall Accuracy Summary

Accuracy-type metrics extracted from the per-model reports in this folder
(precision / recall / F1 omitted — accuracy only).

- **Clean test set:** 2342 samples (2075 background + 267 drone).
- **Clean accuracy** = overall accuracy on the clean test set.
- **ASR** (Attack Success Rate) = % of the 267 real drones misclassified as background when the trigger is present.
- **CDA** (Clean Data Accuracy) and **ASR** in the pruning table come from `eval_pruning_dataset.py` (267 drones), before vs after pruning.

## 1. Stealth & attack effectiveness

| Model | Poison rate | Clean accuracy | ASR — Continuous test | ASR — Pulsed test |
|---|---|:---:|:---:|:---:|
| Clean (baseline) | —    | 99% | 3.0%  | 3.0%  |
| Continuous       | 1.6% | 99% | 93.3% | 56.6% |
| Continuous       | 5%   | 99% | 100%  | 100%  |
| Pulsed           | 1%   | 99% | 100%  | 99.6% |
| Pulsed           | 5.5% | 99% | 100%  | 100%  |

## 2. Pruning defense (34% of `layer3[0].conv1`)

| Model | CDA before → after | ASR before → after |
|---|:---:|:---:|
| Continuous 1.6% | 97.4% → 97.0% | 92.1% → 54.3% |
| Continuous 5%   | 98.5% → 98.5% | 100%  → 26.6% |
| Pulsed 1%       | 97.0% → 98.5% | 100%  → 88.4% |
| Pulsed 5.5%     | 98.9% → 97.4% | 100%  → 98.9% |

*(The clean baseline has no backdoor, so no pruning was applied to it.)*

## Key takeaways

- **Stealth:** every poisoned model keeps **~99% clean accuracy**, identical to the clean baseline — the backdoor is invisible on clean data.
- **Control:** the clean model's ASR is only **~3%**, so the trigger alone barely fools a healthy model — the high ASR of the poisoned models comes from the backdoor, not from the trigger degrading the input.
- **Attack:** ASR grows with the poison rate; it reaches **~100%** at ≥5% (continuous) and already at ~1% (pulsed).
- **Defense:** pruning lowers ASR while keeping CDA ~97-98%, but its effect **weakens on the strongly-poisoned models** (Continuous 5%: 100% → 27%; Pulsed 5.5%: only 100% → 99%).

> Note: `eval_pruning_dataset.py` injects a **continuous** trigger by default; keep this in mind when reading the pruning rows of the *pulsed* models (their pruning ASR is measured against a continuous trigger unless the script was changed per run).
