# 🧠 Mental Wellness Companion

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code of Conduct](https://img.shields.io/badge/Code%20of%20Conduct-Contributor%20Covenant-orange.svg)](CODE_OF_CONDUCT.md)

An innovative Reinforcement Learning-powered mental health support system that learns to provide personalized, empathetic conversations through experience. This project combines Proximal Policy Optimization (PPO) and Contextual Bandits with a novel LLM-in-the-Loop training approach.

## 🎯 Project Overview

The Mental Wellness Companion addresses the mental health crisis affecting 1 in 5 adults annually by providing:
- **24/7 Availability**: Scalable support without waitlists
- **Personalized Interactions**: Learns optimal strategies for each user
- **Privacy-First**: No real patient data used in training
- **Safety Guaranteed**: Zero tolerance for harmful responses

### 🏆 Key Achievements
- **79% User Engagement Rate** (75% improvement over baseline)
- **9.8% Mood Improvement** (390% better than random approaches)
- **0 Safety Violations** across 476+ training episodes
- **60% Conversation Completion Rate** (2x baseline)

## 🚀 Key Features

### 🤖 Dual Reinforcement Learning Approach
1. **PPO (Proximal Policy Optimization)**: Learns conversation strategies
2. **Contextual Bandits (Thompson Sampling)**: Optimizes resource recommendations

### 💡 Novel LLM-in-the-Loop Training
- Uses GPT-4o-mini to generate synthetic training data
- Eliminates need for sensitive patient data
- Creates unlimited, diverse training scenarios

### 🛡️ Comprehensive Safety System
- Multi-layer safety monitoring
- Automatic crisis detection and escalation
- Hard constraints prevent harmful responses
- Professional referral triggers

### 🎨 Production-Ready Architecture
- FastAPI backend for high-performance API
- Streamlit UI for intuitive interaction
- Docker-ready for containerization
- Horizontally scalable design

## 📊 Performance Metrics

| Metric | Our System | Baseline | Improvement |
|--------|------------|----------|-------------|
| Engagement Rate | 79% | 45% | **+75%** |
| Mood Improvement | 9.8% | 2% | **+390%** |
| Completion Rate | 60% | 30% | **+100%** |
| Safety Violations | 0 | 15% | **-100%** |

## 🏗️ System Architecture


## 🛠️ Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (optional but recommended)
- OpenAI API key
- 8GB+ RAM

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mental-wellness-companion.git
cd mental-wellness-companion
```

2. Create conda environment

```bash
conda create -n mental-wellness python=3.10
conda activate mental-wellness
```

3. Install PyTorch

```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# For Mac M1/M2
conda install pytorch torchvision torchaudio -c pytorch
```

4. Install dependencies

```bash
pip install -r requirements.txt
```

5. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

6. Verify installation

```bash
python verify_setup.py
```

## 🎮 Usage

### Training the Model

```bash
# Full training (200 episodes)
python scripts/train_improved.py

# Quick training (50 episodes)
python scripts/train.py --episodes 50 --batch-size 32

# Resume from checkpoint
python scripts/train.py --load-checkpoint data/models/checkpoint.pt
```

### Running the Application

Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

### Launch the UI

```bash
streamlit run ui/app.py --server.port 8501
```

### Access the application

UI: http://localhost:8501  
API Docs: http://localhost:8000/docs

### Evaluation

```bash
# Run evaluation on test personas
python scripts/evaluate.py --num-personas 20

# Generate performance report
python scripts/generate_report.py
```

### 📁 Project Structure

```
mental-wellness-companion/
├── src/
│   ├── agents/           # Multi-agent implementations
│   ├── rl/              # PPO and Contextual Bandit
│   ├── simulation/      # LLM-based user simulation
│   ├── safety/          # Safety monitoring system
│   └── utils/           # Helper functions
├── api/                 # FastAPI backend
├── ui/                  # Streamlit interface
├── config/              # Configuration files
├── scripts/             # Training and evaluation scripts
├── data/               # Data and trained models
├── docs/               # Documentation
├── tests/              # Unit and integration tests
└── notebooks/          # Jupyter notebooks for analysis
```

## 📈 Training Results

### Learning Curves

- Episodes: 476 completed
- Best Reward: 3.60
- Policy Loss: Reduced from 0.20 to 0.05 (75% reduction)
- Value Loss: Converged from 3.5 to 2.0

### Strategy Distribution (Learned)

- Validation: 70.2%
- Cognitive Behavioral: 24.6%
- Psychoeducation: 5.26%

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test module
pytest tests/test_agents.py
```

## 🤝 Contributing
We welcome contributions! Please see our Contributing Guidelines and Code of Conduct.  

## Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Format code
black src/ tests/

# Run linter
pylint src/
```

## 📊 Experimental Results

### Ablation Study Results
| Configuration | Performance| Safety Score | 
|--------|------------|----------|
| Full System | 79% | 100% |
| Without PPO | 62% (-27%) | 95% |
| Without Bandits | 71% (-10%) | 100% |
| Without Safety | 75% | 0% |

### Statistical Validation

- T-test: p < 0.001 (highly significant)
- Cohen's d: 2.3 (large effect size)
- 95% CI: [0.76, 0.82] for engagement

## 🔒 Safety & Ethics

This project prioritizes user safety:

- No Real Patient Data: All training uses synthetic personas
- Crisis Detection: 100% catch rate for crisis keywords
- Professional Referral: Automatic escalation at risk threshold
- Transparent AI: Clear about AI nature and limitations
- Privacy First: No conversation data stored

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{mental_wellness_companion_2025,
  author = {Vedant Mane},
  title = {Mental Wellness Companion: RL-Powered Mental Health Support},
  year = {2025},
  url = {https://github.com/vedantmane12/mental-wellness-companion}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Course instructors for guidance on RL implementation
- OpenAI for GPT-4o-mini API access
- Anthropic's Claude for development assistance
- Open-source community for foundational libraries
- Mental health professionals who provided domain expertise

## ⚠️ Disclaimer

This system is designed for educational and research purposes. It is not a replacement for professional mental health care. If you're experiencing a mental health crisis, please contact:

- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741
- International Crisis Lines: findahelpline.com