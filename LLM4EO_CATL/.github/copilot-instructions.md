# Copilot Instructions for LLM4EO

## Project Overview
- **LLM4EO** is an evolutionary optimization framework for Flexible Job Shop Scheduling Problems (FJSSP), leveraging Large Language Models (LLMs) for operator generation and improvement.
- Main algorithm logic is in `LLM4EO.py`. LLM API integration and configuration are handled in `run_LLM4EO.py`.
- Data instances are stored in `Instances/` (see `Instances/README.md` for format and references).

## Key Components
- `LLM4EO.py`: Core evolutionary algorithm, including crossover, mutation, and operator adaptation using LLM-generated strategies.
- `Parameter.py`: Defines the `Paras` class for all tunable parameters (e.g., population size, LLM API settings).
- `DataProcess/`: Contains modules for data reading (`ReadingData.py`), initial population creation (`CreateInitGroup.py`), fitness evaluation (`Fitness.py`), and translation between solution and data (`Translate.py`).
- `LLM/Strategy.py`: Handles LLM-based strategy generation and improvement.

## Developer Workflows
- **Run the algorithm:**
  - Edit LLM API settings in `run_LLM4EO.py` (endpoint, key, model).
  - Select an instance file from `Instances/` and update the path in `run_LLM4EO.py`.
  - Run with: `python run_LLM4EO.py`
- **Add new instances:**
  - Place files in the appropriate subfolder under `Instances/`.
  - Update `Instances/instances.json` if needed.
- **Parameter tuning:**
  - Modify `Parameter.py` or pass new values via `Paras` in `run_LLM4EO.py`.

## Patterns & Conventions
- Chromosomes are dictionaries with keys: `job`, `operation`, `machine`, and `perturbate_rate`.
- Operator logic (crossover/mutation) adapts dynamically using LLM-generated Python code (see `Strategy.py`).
- Solution evaluation is always performed via `Translate.main()`.
- Selection methods: 'roulette' and 'binary tournament' (see `LLM4EO.py`).
- All external LLM calls are abstracted in `LLM/Strategy.py`.

## Integration Points
- LLM API: Configure endpoint, key, and model in `Paras`.
- Data: Instance files must follow the format described in `Instances/README.md`.
- Results: Best chromosome and score are returned by `LLM4EO.main()`.

## Examples
- See `run_LLM4EO.py` and the main usage block in `README.md` for end-to-end workflow.
- For new operator logic, update or extend `LLM/Strategy.py`.

---

For questions about unclear conventions or missing documentation, ask the user for clarification or examples from their workflow.
