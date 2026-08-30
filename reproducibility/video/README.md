# Video reconstruction

`reconstruct_clips.py` reads the public `video_list.json`, reconstructs the
recorded interval, writes the exact benchmark filename, validates the result
with `ffprobe`, and emits a CSV availability report.

The preferred route is `--source-dir`, which uses retained source media already
available to the researcher. `--download` is an optional best-effort route for
sources that remain public. Users are responsible for complying with source
licenses and platform terms. Failed or inaccessible sources are reported and
are not silently replaced.
