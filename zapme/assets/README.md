# Audio assets

Voice clips played by the runtime through the speaker.

The runtime expects five clips by default (all `.mp3`). Anything missing
is silently skipped (just logged), so it's safe to omit any of these
without breaking the loop:

| File             | When it plays                                                       |
| ---------------- | ------------------------------------------------------------------- |
| `bootup.mp3`     | Once at process startup, after the gate / watchdog are wired.       |
| `firstwarn.mp3`  | First detection of a sustained slouch — gentle nudge.               |
| `finalwarn.mp3`  | Slouch persisted past `--warn-to-warn` seconds.                     |
| `zapwarn.mp3`    | Plays immediately before the EMS pulse fires (sequence step 1).     |
| `zapscream.mp3`  | Plays right after `zapwarn` finishes — the actual zap (sequence 2). |

`zapwarn` then `zapscream` play **sequentially** in a background
worker, so the second clip starts the moment the first subprocess
exits — no overlap, no blocking the inference loop.

### Variants (multiple takes for one clip)

For any clip name, drop in additional files matching `<name>*.mp3`
and the runtime will pick one at random per play. Example:

```
zapscream.mp3
zapscream2.mp3
zapscream3.mp3
```

… and `play("zapscream")` chooses one of the three uniformly each
zap. This works for any clip: `firstwarn2.mp3`, `bootup_alt.mp3`,
etc. Variant discovery happens at startup; the runtime logs which
clips have multiple variants on boot. To go back to a single take,
delete the extra files.

Format: MP3. The platform binary used is:

- Linux (Pi): `mpg123 -q` — `sudo apt install -y mpg123`
- macOS: `afplay`
- Windows: `ffplay -nodisp -autoexit` (install ffmpeg)

If you want different filenames, edit `DEFAULT_CLIPS` /
`DEFAULT_CLIP_EXT` in `zapme/src/__main__.py`.

To skip audio entirely (e.g. for silent benchtop testing), pass
`--no-audio` to the runtime.
