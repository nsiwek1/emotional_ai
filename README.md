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
