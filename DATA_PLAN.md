# Cyber-LLM — Data & Training Plan

A local, security-specialized assistant: QLoRA fine-tune of **Foundation-Sec-8B-Instruct**
on a curated ~25k-example mix, trained on Kaggle's free T4, run locally on an RTX 4060 (8 GB)
via Ollama. Personal/local use (non-commercial datasets allowed; each row keeps its license tag).

## Pipeline

```
13 raw datasets ──adapt──> one canonical record ──normalize──> Llama-3.1 format
   ──dedup/convert──> ~25k mix + held-out eval ──QLoRA (Kaggle T4)──>
   merge (CPU) ──> GGUF Q4_K_M + Modelfile ──> Ollama on your 4060
```

Downloads happen on **Kaggle's** network, not yours. The only file you pull locally is the
final ~5 GB GGUF.

## The mix (edit `mix_config.yaml`, not code)

Target balance: ~28% tool-use/agentic, ~37% defensive, ~18% offensive, ~14% knowledge, ~4% general.
Offensive is boosted as far as *clean* data allows (offensive SFT data is scarce); tool-use is
heavy because it is the hardest capability to instill.

| Source | Rows | Bucket | License |
|---|---|---|---|
| SEvenLLM-Instruct | 4000 | defensive + reasoning | apache-2.0 |
| xLAM-function-calling | 3000 | tool-use | cc-by-4.0 |
| cyberdata-2-large | 3000 | detection/vuln/CTI | mixed (Sigma=DRL) |
| pentesting-explanations | 3000 | offensive | apache-2.0 (HackTricks-NC source) |
| security-agentic-sft-corpus | 2500 | agentic | mixed |
| Fenrir v2.1 | 2500 | general knowledge | apache-2.0 |
| ToolACE | 1500 | tool-use | apache-2.0 |
| DiverseVul | 1500 | code-vuln | cc-by-4.0 |
| Primus-Reasoning | 1000 | reasoning | odc-by |
| CyberMetric | 1000 | knowledge | apache-2.0 |
| Dolly-15k (general) | 1000 | anti-forgetting | cc-by-sa-3.0 |
| Primus-Instruct | 835 | defensive | odc-by |
| pentest-checklist-en | 160 | offensive | cc-by-4.0 |
| **Held out for eval:** SecQA (200), CyberMetric-80 (80) | — | eval | — |

## Four hard rules (the silent failure points)

1. **One tool-call format.** All tool-use data normalizes to Llama-3.1's format; never mix syntaxes.
2. **De-duplicate** by strong id (commit hash via `id_fields`, CVE via `extract_cve`) or exact text-hash; `cyberdata-2` already embeds some PrimeVul/CVEfixes rows. Paraphrase-level overlap with no shared id is not caught.
3. **MCQ → reasoning+answer**, and hold the quiz benchmarks OUT as eval — never train on what you test.
4. **Keep the general slice** (Dolly) so the model doesn't get dumber at everything non-security.

## License notes (personal use is clear; commercial would need pruning)

- Non-commercial / restricted: HackTricks-derived (`pentesting-explanations`), `oi-uae` (unused),
  SecQA (eval), the one NC-SA file inside `security-agentic` (dropped at load), SigmaHQ rows in
  `cyberdata-2` (DRL 1.1). Each row carries `meta.license`, so a commercial subset can be filtered later.
- Clean to build from directly: MITRE ATT&CK (attribution), PayloadsAllTheThings (MIT), NVD/CVE (public domain).

## Base model & why

`fdtn-ai/Foundation-Sec-8B-Instruct` — Llama-3.1-8B continued-pretrained by Cisco on security
corpora. Because it already knows security, the fine-tune spends its budget on *behavior,
reasoning, and tool-use*, not on cramming facts.

## Known limit, by design

`field_map` values reflect each dataset's **documented** schema. If a live dataset's columns
differ, its adapter raises a clear schema-drift error on the first Kaggle run — adjust the
column names in `mix_config.yaml`. This is expected, not a bug: it is cheaper to fix one YAML
line than to silently train on empty records.

## Not "pretty good" by data alone

An 8B model has a ceiling, and fine-tuning bakes in knowledge up to the data cutoff. "Good" is
proven by the held-out eval, then improved by (a) adding data where weak and (b) adding **RAG**
for current facts (new CVEs). RAG is phase 2, after v1 works.
