# FactCloze
Improving Factual Error Correction for Abstractive Summarization via Data Distillation and Conditional-generation Cloze

You can use main.py as a reference to use FactCloze.
We propose the related models as below:

| Model | dataset | Post Alert | URL |
| ------- | ------- | ------- | ------- |
| t5-base | CNN/DM  | No      | https://huggingface.co/KenLee/t5_base_cnndm_sd |
| t5-base | CNN/DM  | Yes     | https://huggingface.co/KenLee/t5_base_cnndm_sd_pa |
| bart-large | XSum  | No     | https://huggingface.co/KenLee/bart_large_xsum_sd |
| bart-large | Xsum  | Yes    | https://huggingface.co/KenLee/bart_large_xsum_sd_pa |

