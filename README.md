# Emotional AI Research App

This project includes:
- A React chat UI (Vite)
- A sycophantic system prompt for chatbot behavior
- A persona generator for controlled emotion/intensity/topic scenarios

## Run the chat app

```bash
npm install
npm run dev
```

Set your key in `.env`:

```bash
VITE_OPENAI_API_KEY=your_openai_api_key_here
```

## Persona generator (Step 1: trajectory mode)

The generator creates personas for:
- Emotions: `joy`, `anger`
- Intensity: `low`, `high`
- Topics: `work_performance`, `relationship`, `life_event`, `public_event`

This yields **16 scenarios**. Default run creates **100 personas per scenario** (1600 total).

Each persona record includes:
- scenario metadata
- a 5-turn persona trajectory (`turns[0]`..`turns[4]`) about the same issue
- validation metadata (fallback/retry context)

### Commands

Dry-run without API calls:

```bash
npm run persona:dry-run
```

Generate with OpenAI:

```bash
npm run persona:generate
```

### Environment variables

- `OPENAI_API_KEY` (preferred; required for real generation)
- `PERSONAS_PER_SCENARIO` (default `100`)
- `PERSONA_MODEL` (default `gpt-4o-mini`)
- `PERSONA_OUTPUT_DIR` (default `data/personas`)
- `PERSONA_REQUEST_DELAY_MS` (default `120`)
- `PERSONA_MAX_RETRIES` (default `3`)
- `PERSONA_DUPLICATE_RETRIES` (default `6`)
- `PERSONA_DUPLICATE_THRESHOLD` (default `0.72`)
- `PERSONA_TEMPERATURE` (default `0.9`)

Example:

```bash
OPENAI_API_KEY=sk-... PERSONAS_PER_SCENARIO=100 PERSONA_TEMPERATURE=0.9 npm run persona:generate
```

### Output

The script writes timestamped files to `data/personas`:
- `personas_<timestamp>.json`
- `personas_<timestamp>.jsonl`

Each row contains:
- `persona_id`
- `scenario_id`
- `emotion`
- `intensity`
- `topic`
- `background`
- `persona_style`
- `turns` (exactly 5 turns in sequence)
- `metadata`

## Emotion scoring (Hugging Face, Python)

Scores each persona **per round** (seed = `turns[0]`, follow-ups = `turns[1:]`) and the **concatenated** full text using **`huggingface_hub.InferenceClient`** ([Inference Providers](https://huggingface.co/docs/api-inference); the old `api-inference.huggingface.co` URL returns **410 Gone**). The default model is **[j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)** (labels include **joy**, **anger**, **neutral** among seven classes). Each `hf` block includes full `labels`, **entropy**, and **`joy_anger_neutral`** (`seven_class_marginals` + `three_way_renormalized`).

### Setup

On macOS, Homebrew Python is often **PEP 668** (“externally managed”), so install into a **venv** (do not use bare `pip install` system-wide).

```bash
cd /path/to/research
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-emotion.txt
# installs huggingface_hub (InferenceClient), not raw requests to the legacy inference URL
```

Use **`python3`** if `python` is not found. After activating the venv, `python` will work and point at `.venv`.

```bash
./.venv/bin/python emotion_score_hf.py --input data/personas/persona_final.json --output data/emotion_scores/out.json --limit 5
```

Set a token with Inference API access (create at [Hugging Face settings](https://huggingface.co/settings/tokens)):

```bash
export HF_TOKEN=hf_...
# optional:
export HF_EMOTION_MODEL=j-hartmann/emotion-english-distilroberta-base
```

Or add `HF_TOKEN` to `.env` (the script loads it via `python-dotenv` if installed).

### Run

```bash
python emotion_score_hf.py \
  --input data/personas/persona_final.json \
  --output data/emotion_scores/persona_emotions.json
```

Options:

- `--limit N` — process only the first N personas (dry run).
- `--sleep-ms M` — delay between API calls (rate limits).
- `--concat-sep " "` — separator for the concatenated text (default space).
- `--model MODEL` — override `HF_EMOTION_MODEL`.

Output JSON: one object per persona with `per_round` (each with `hf.labels`, `hf.entropy`, `hf.top`) and `concatenated` with the same `hf` shape.
