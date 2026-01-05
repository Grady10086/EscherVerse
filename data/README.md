# Data

Download the dataset from Hugging Face:

```bash
huggingface-cli download Gradygu3u/Escher-Data --local-dir ./
```

Or use Python:

```python
from huggingface_hub import hf_hub_download

# Download benchmark
hf_hub_download(
    repo_id="Gradygu3u/Escher-Data",
    filename="Escher-Bench.json",
    repo_type="dataset",
    local_dir="./"
)
```

## Video Files

Videos are sourced from YouTube. Contact the authors for access to the processed video clips.

Each video filename follows the format:
```
{youtube_id}_{clip_index}_{start_time}_to_{end_time}.mp4
```

Example: `N0b5SvS9k0E_66_0_12_19_238_to_0_12_50_803.mp4`
