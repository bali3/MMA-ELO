AGENTS.md

## Mission
Build a production-grade MMA fight prediction and betting-edge research system with strict chronological integrity and zero data leakage.

## Non-negotiable rules
- Never use future information in feature generation
- Never use random train/test splits
- Always validate chronologically
- Prefer simple, testable modules over clever shortcuts
- Preserve raw source data before transformation
- All model inputs for a bout must be available before the bout starts

## Modeling priorities
- Profitability and calibration over headline accuracy
- Baselines before complexity
- UFC-only first
- Continuous style vectors over hard-coded style categories
- Elo as a prior, not the whole model

## Code standards
- Python with type hints
- Modular src/ layout
- Tests for chronology and leakage prevention
- Clear docs and assumptions
- Small PR-sized changes

## Do not
- Do not hard-code subjective fighter style labels
- Do not compute rolling stats on the full dataset before splitting
- Do not mix exploratory notebook logic into production modules
- Do not optimize for one backtest period at the expense of robustness

## Preferred workflow
1. Scaffold
2. Schema
3. Ingestion
4. Entity resolution
5. Sequential features
6. Ratings
7. Baselines
8. Walk-forward evaluation
9. Betting simulation
10. Advanced models