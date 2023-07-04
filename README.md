# Flacuna: A Vicuna made of Flan

![Alt Text](flacuna5.png)

Flacuna was developed by fine-tuning Vicuna on Flan-mini, a comprehensive instruction collection encompassing various tasks. Vicuna is already an excellent writing assistant, and the intention behind Flacuna was to enhance Vicuna's problem-solving capabilities. To achieve this, we curated a dedicated instruction dataset called Flan-mini.

| Dataset Name                | Source                 | Dataset Size |
|-----------------------------|------------------------|--------------|
| Flan2021                    | Flan                   | 388K         |
| Public Pool of Prompts      | Flan                   | 320K         |
| Natural instructions v2     | Flan                   | 200K         |
| CoT                         | Flan                   | 100K         |
| Code Search                 | husain2019codesearchnet | 100K         |
| Code Contest                | li2022competition      | 50K          |
| Apps                        | hendrycksapps2021      | 50K          |
| GPT4-Alpaca                 | GPT-4                  | 52K          |
| Code-Alpaca                 | ChatGPT                | 20K          |
| ShareGPT                    | ChatGPT                | 60K          |
| Total                       | -                      | 1.34M        |


As a result of this fine-tuning process, Flacuna exhibited notable performance improvements in problem-solving across multiple benchmark datasets, both in few-shot and zero-shot settings.

| **Model** | **Size** | **MMLU (5-shot)** | **BBH (3-shot)** | **DROP (3-shot)** | **CRASS (3-shot)** | **HumanEval (0-shot)** | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StableVicuna | 13B | 49.2 (+3.0) | 37.5 (+0.4) | 34.3 (-1.0) | 67.5 (+8.7) | 15.9 (+2.5) | 40.9 (+2.7) |
| Vicuna | 13B | 50.6 (+4.5) | 37.6 (+0.5) | 32.6 (-3.0) | 60.9 (+2.1) | 11.6 (-1.8) | 38.8 (+0.6) |
| Flacuna | 13B | 51.1 (+5.0) | 39.3 (+2.2) | 43.6 (+8.0) | 74.1 (+15.3) | 11.0 (-2.4) | 43.8 (+5.6) |

| Model | Size | **MMLU (0-shot)** | **BBH (0-shot)** | **CRASS (0-shot)** |
| --- | --- | --- | --- | --- |
| StableVicuna | 13B | 47.5 | 18.5 | 64.2 |
| Vicuna | 13B | 48.3 | 28.3 | 65.7 |
| Flacuna | 13B | 49.4 | 32.5 | 67.9 |


During training, Flacuna employed a maximum input sequence length of 1280. We utilized LoRA for parameter-efficient fine-tuning.
