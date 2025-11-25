# Multi-Agents-Metric

A multi-agent evaluation framework for machine translation quality assessment using Large Language Models (LLMs). The system employs specialized agents (Specialist, Supervisor, and Orchestrator) to evaluate translations across multiple quality dimensions including accuracy, linguistic conventions, localization, style, and terminology consistency.

## Features

- **Multi-Agent Architecture**: Implements specialists, supervisors, and orchestrators for nuanced evaluation
- **Modular Evaluation Categories**:
  - Accuracy (Accuracy)
  - Linguistic Conventions (Linc)
  - Localization (Loc)
  - Style (Sty)
  - Terminology (Term)
- **Scoring System**: Computes final translation quality scores with predefined penalty systems
- **Threaded Processing**: Supports parallel evaluation of multiple translation pairs
- **CSV/TSV Data Handling**: Processes machine translation hypotheses vs. ground truth
- **Prompt Engineering**: Uses structured prompts to guide agent behaviors

## Installation

### Prerequisites
- Python 3.7+
- OpenAI API key
- Google Colab (preferred) or Jupyter Notebook environment

### Dependencies
```bash
pip install openai python-dotenv
```

Additional imports (may need installation):
- `csv`, `argparse`, `os`, `zipfile`, `json`, `threading` (built-in Python)
- `google.colab` (for Colab environments)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/MaiRodrigues/Multi-Agents-Metric.git
cd Multi-Agents-Metric
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Colab Environment (Recommended)
1. Open `Agents/system_multi_agents.ipynb` in Google Colab
2. Upload your data files and prompts zip
3. Set your OpenAI API key in the notebook
4. Run the cells sequentially

### Data Format
- `source.tsv`: Source language texts
- `target.tsv`: Ground truth translations
- `hyp_source.tsv`: Hypothesis source texts
- `hyp_target.tsv`: Machine translation hypotheses

### Prompt Structure
Upload a zip file containing prompt templates organized as `system_<category>_<subcategory>.txt`

## Project Structure

```
Multi-Agents-Metric/
│
├── Agents/
│   └── system_multi_agents.ipynb          # Main multi-agent system implementation
│
├── Data/
│   ├── Dataset_Processing.ipynb           # Data preprocessing scripts
│   ├── source.tsv                         # Source texts
│   ├── target.tsv                         # Ground truth translations
│   ├── hyp_source.tsv                     # Hypothesis sources
│   ├── hyp_target.tsv                     # Machine translation outputs
│   └── final_*.txt                        # Processed datasets
│
├── Evaluation/
│   ├── Metrics_Comparison.ipynb           # Comparative analysis notebooks
│   └── Metrics_Evaluation.ipynb          # Evaluation utilities
│
├── Prompts/
│   ├── system_*.txt                       # Specialized prompt templates
│   ├── complement_*.txt                   # Supplementary prompts
│   └── Prompts.zip                        # Archived prompts
│
└── README.md                              # Project documentation
```

## Agent Types

### Specialist Agent
- Evaluates translations for specific error types
- Provides severity ratings and correction suggestions

### Supervisor Agent
- Reviews specialist evaluations
- Ensures consistency across evaluation categories

### Orchestrator Agent
- Manages the evaluation workflow
- Handles disagreements between agents through debate mechanisms
- Includes self-reflection capabilities

## Scoring

The system computes scores starting from 100, applying penalties:
- No error: 0 points
- Minor error: 2 points deduction
- Severe error: 5 points deduction

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source.
