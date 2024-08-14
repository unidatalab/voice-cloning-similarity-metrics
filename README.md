# Voice Cloning Metrics

## Pyannote token

1. Accept [pyannote/segmentation-3.0](https://hf.co/pyannote/segmentation-3.0) user conditions
2. Accept [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) user conditions
3. Create access token at [hf.co/settings/tokens](https://hf.co/settings/tokens).

Then make token as environment variable
```bash
PYANNOTE_TOKEN=PLACE_GENERATE_TOKEN
``` 

## Input data requirements

Sample rate = 16 kHz

Number of channels = 1 (mono)

Audio format = .wav

Audio of originals and cloned voices must be placed in different folders. All audio files must be named as 0.wav, 1.wav, 2.wav and so on.

**NOTE**: For better similarity calculation, audio files with the original and cloned voices located 
in different folders but having the same name should voice the same text. If this condition is met, 
then it is possible to use `--correction_type=interim`.

## Install requirements

```bash
pip install -r requirements.txt
```

## Usage example
```bash
python3 similarity_calc.py --orig_dir examples/jordan_peters/origs \
--cloned_dir examples/jordan_peters/11labs \
--output_dir examples/jordan_peters/results \
--correction_type final # final or interim
```