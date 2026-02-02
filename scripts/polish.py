name: Phase 2 – Cadence & Beauty Polish

on:
  workflow_dispatch:
    inputs:
      book:
        description: "Book filename to polish (e.g., genesis)"
        required: true
        default: "genesis"

jobs:
  polish:
    runs-on: ubuntu-latest

    steps:
      - name: Check out repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install openai

      - name: Verify inputs exist
        run: |
          test -f output/${{ github.event.inputs.book }}.jsonl
          test -f phase2/targets.jsonl
          test -f charter/phase2_charter.txt

      - name: Run Phase 2 polish
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          mkdir -p output_phase2
          python scripts/polish.py \
            --in output/${{ github.event.inputs.book }}.jsonl \
            --targets phase2/targets.jsonl \
            --charter charter/phase2_charter.txt \
            --out output_phase2/${{ github.event.inputs.book }}.jsonl

      - name: Commit polished output
        run: |
          git config user.name "github-actions"
          git config user.email "github-actions@github.com"
          git add output_phase2/${{ github.event.inputs.book }}.jsonl
          git commit -m "Phase 2 polish: ${{ github.event.inputs.book }}"
          git push
