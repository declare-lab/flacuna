# Flacuna: A Vicuna made of Flan

[Paper](https://arxiv.org/abs//2307.02053) | [Model](https://huggingface.co/declare-lab/flacuna-13b-v1.0) | [Dataset](https://huggingface.co/datasets/declare-lab/flan-mini)

📣 We still have numerous experiments awaiting completion (details are [here](https://arxiv.org/abs//2307.02053)), requiring additional computing resources in our lab. If any industry professionals reading this are willing to provide assistance, please feel free to reach out to us at sporia@sutd.edu.sg.

<p align="center">

<img src="https://declare-lab.net/assets/images/logos/flacuna5.png" alt="Image" width="200" height="335">

</p>

Flacuna was developed by fine-tuning Vicuna on Flan-mini, a comprehensive instruction collection encompassing various tasks. Vicuna is already an excellent writing assistant, and the intention behind Flacuna was to enhance Vicuna's problem-solving capabilities. To achieve this, we curated a dedicated instruction dataset called Flan-mini.

| Dataset Name                | Source                 | Dataset Size |
|-----------------------------|------------------------|--------------|
| Flan2021                    | Flan                   | 388K         |
| Public Pool of Prompts      | Flan                   | 320K         |
| Natural instructions v2     | Flan                   | 200K         |
| CoT                         | Flan                   | 100K         |
| Code Search                 | HF/code_search_net | 100K         |
| Code Contest                | HF/deepmind/code_contests      | 50K          |
| Apps                        | HF/codeparrot/apps      | 50K          |
| GPT4-Alpaca                 | GPT-4                  | 52K          |
| Code-Alpaca                 | ChatGPT                | 20K          |
| ShareGPT                    | ChatGPT                | 60K          |
| Total                       | -                      | 1.34M        |


## Quickstart Guide
```python
from flacuna import FLACUNA
model = FLACUNA("declare-lab/flacuna-13b-v1.0")

prompt = (
  "A chat between a curious user and an artificial intelligence assistant. "
  "The assistant gives helpful, detailed, and polite answers to the user's questions. "
  "USER: You are tasked to demonstrate your writing skills in professional or work settings for the following question.\n"
  "Can you help me write a speech for a graduation ceremony, inspiring and motivating the graduates to pursue their dreams and make a positive impact on the world?\n"
  "Output: ASSISTANT: "
)
decoded = model.generate(prompt)
print (decoded)
```

The output is the following:

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: You are tasked to demonstrate your writing skills in professional or work settings for the following question.
Can you help me write a speech for a graduation ceremony, inspiring and motivating the graduates to pursue their dreams and make a positive impact on the world?
Output: ASSISTANT: ​​Dear Graduates,

It is my honor to stand before you here today, to celebrate your achievements and to acknowledge your hard work and dedication that you have shown here at the University. I am proud of each one of you, for all that you have accomplished thus far, and for the potential that lies within each of you.

As you move on from this stage, I want to remind you that your journey has just begun. The real world awaits you, and with it, comes a myriad of opportunities and challenges. But I believe in each one of you, and I know that you are more than capable of facing whatever lies ahead.

You are the future of this world, and I have no doubt that you will make a positive impact on it, in whatever field you choose to pursue. Whether it is in science, technology, education, healthcare, or any other field, your contributions will be valuable and meaningful, and will make a difference in the lives of others.

So, as you move forward, I urge you to stay true to your passions, to pursue your dreams, and to never give up on what you believe in. Remember, it is not the obstacles that define us, but how we overcome them that truly matters.

You have the power to change the world, and I have no doubt that you will. So, go out there, and make the most of every opportunity that comes your way. Make a difference, and leave your mark on this world.

Congratulations once again, and best of luck in all that lies ahead.

Thank you.
```

## Problem Solving Ability

As a result of this fine-tuning process, Flacuna exhibited notable performance improvements in problem-solving across multiple benchmark datasets, both in few-shot and zero-shot settings.

| **Model** | **Size** | **MMLU (5-shot)** | **BBH (3-shot)** | **DROP (3-shot)** | **CRASS (3-shot)** | **HumanEval (0-shot)** | **Avg.** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| StableVicuna | 13B | 49.2 (+3.0) | 37.5 (+0.4) | 34.3 (-1.0) | 67.5 (+8.7) | 15.9 (+2.5) | 40.9 (+2.7) |
| Vicuna | 13B | 50.6 (+4.5) | 37.6 (+0.5) | 32.6 (-3.0) | 60.9 (+2.1) | 11.6 (-1.8) | 38.7 (+0.6) |
| Flacuna | 13B | 51.1 (+5.0) | 39.3 (+2.2) | 43.6 (+8.0) | 74.1 (+15.3) | 11.0 (-2.4) | 43.8 (+5.6) |

| **Model** | **Size** | **MMLU (0-shot)** | **BBH (0-shot)** | **CRASS (0-shot)** |
| --- | --- | --- | --- | --- |
| StableVicuna | 13B | 47.5 | 18.5 | 64.2 |
| Vicuna | 13B | 48.3 | 28.3 | 65.7 |
| Flacuna | 13B | 49.4 | 32.5 | 67.9 |


During training, Flacuna is a 13B checkpoint of LLaMA and employed a maximum input sequence length of 1280. We utilized LoRA for parameter-efficient fine-tuning.

## Chatbot / Writing Assistant

While Flacuna primarily excels in problem-solving tasks, we made efforts to maintain the impressive writing and chatting ability of Vicuna. To achieve this, we incorporated conversational datasets generated by GPT-4, such as GPT-4-Alpaca and ShareGPT, into the Flan-mini collection.
To use Flacuna as a chatbot or writing assistant, we recommend you use the following template:

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {definition of the task}.\n\n
{question}\n
Output: ASSISTANT:

```
**Please note that we still recommend using Vicuna as your preferred Chatbot or Writing Assistant, over Flacuna. Flacuna's primary strength lies in problem-solving tasks, making it ideal for such applications.**

The following table presents the writing performance of Flacuna on the IMPACT dataset, which is a component of the InstructEval evaluation suite. The generated responses have been evaluated by ChatGPT, and their relevance and coherence have been scored on a scale of 1 to 5.


| **Model** | **Size** | **Informative Rel.** | **Informative Coh.** | **Professional Rel.** | **Professional Coh.** | **Argumentative Rel.** | **Argumentative Coh.** | **Creative Rel.** | **Creative Coh.** | **Avg. Rel.** | **Avg. Coh.** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ChatGPT | - | 3.34 | 3.98 | 3.88 | 3.96 | 3.96 | 3.82 | 3.92 | 3.94 | 3.78 | 3.93 |
| Flan-Alpaca | 11B | 3.56 | 3.46 | 3.54 | 3.70 | 3.22 | 3.28 | 3.70 | 3.40 | 3.51 | 3.46 |
| Flan-T5 | 11B | 2.64 | 3.24 | 2.62 | 3.22 | 2.54 | 3.40 | 2.50 | 2.72 | 2.58 | 3.15 |
| Dolly-V2 | 12B | 3.54 | 3.64 | 2.96 | 3.74 | 3.66 | 3.20 | 3.02 | 3.18 | 3.30 | 3.44 |
| StableVicuna | 13B | 3.54 | 3.64 | 2.96 | 3.74 | 3.30 | 3.20 | 3.02 | 3.18 | 3.21 | 3.44 |
| Vicuna | 13B | 3.60 | 3.96 | 3.74 | 3.82 | 3.82 | 3.56 | 3.82 | 3.92 | 3.75 | 3.82 |
| Flacuna | 13B | 3.02 | 3.42 | 3.48 | 3.52 | 3.38 | 3.02 | 3.92 | 3.80 | 3.45 | 3.44 |


## Training Flacuna
Navigate to the `data` directory and download the Flan-Mini dataset:
```bash
cd data
wget https://huggingface.co/datasets/declare-lab/flan-mini/resolve/main/flan_mini.json.zip
unzip flan_mini.json.zip
cd ..
```

You can then use the `train.sh` script for fine-tuning Vicuna on the Flan-Mini dataset:
```bash
bash train.sh
```

## Citation

```bibtex
@misc{ghosal2023flacuna,
      title={Flacuna: Unleashing the Problem Solving Power of Vicuna using FLAN Fine-Tuning}, 
      author={Deepanway Ghosal and Yew Ken Chia and Navonil Majumder and Soujanya Poria},
      year={2023},
      eprint={2307.02053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
