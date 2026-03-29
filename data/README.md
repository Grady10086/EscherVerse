# Data access

The released EscherVerse files are hosted on Hugging Face:

- https://huggingface.co/datasets/Gradygu3u/EscherVerse-Data

Recommended download command:

```bash
huggingface-cli download Gradygu3u/EscherVerse-Data \
  --repo-type dataset \
  --local-dir ./
```

To download only the benchmark and metadata files:

```bash
huggingface-cli download Gradygu3u/EscherVerse-Data Escher-Bench.json video_list.json \
  --repo-type dataset \
  --local-dir ./
```

To download the instruction-tuning annotations:

```bash
huggingface-cli download Gradygu3u/EscherVerse-Data Escher-sft.jsonl Escher-GRPO-Subset.jsonl \
  --repo-type dataset \
  --local-dir ./
```

## Notes

- The benchmark and training files use different schemas, so direct file download is recommended instead of relying on the dataset viewer.
- The underlying raw clips are derived from third-party online platforms and are not redistributed as an unrestricted public download.
- Access to retained source clips is controlled and subject to availability and source-platform terms.

## Video filenames

Each processed clip filename follows the format:

```text
{youtube_id}_{clip_index}_{start_time}_to_{end_time}.mp4
```

Example:

```text
N0b5SvS9k0E_66_0_12_19_238_to_0_12_50_803.mp4
```
