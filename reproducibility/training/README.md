# SFT/LoRA training

`train_qwen3vl_lora.py` is the entry point used for the content-specific SFT ablation.
It trains language-attention LoRA modules while freezing the visual encoder.
The exact settings are in `sft_lora_config.json`; the frozen subsets and run
records are in `../../experiments/sft_ablation/`.

Install the training additions with:

```bash
pip install -r reproducibility/training/requirements.txt
```

The script accepts either a Hugging Face model ID or a local checkpoint path.
Pass `--local-files-only` for an offline run. It verifies that every training
row has an available video before starting a full run.

This release supports exact reconstruction of the ablation training setup.
It does not currently include the original adapters or complete historical
configuration records for every Escher row in the main model table; those rows
are marked accordingly in `../models/model_registry.csv`.
