# Efficient Sphere-Effect based Information Diffusion Prediction on Large-scale Social Networks

This repository contains the implementation of our paper "Efficient Sphere-Effect based Information Diffusion Prediction on Large-scale Social Networks" (SIGKDD 2025).



## 📁 Repository Structure

```
.
├── main.py              # Main training and evaluation script
├── model.py             # Core model architecture implementation
├── module.py            # Implementation of individual model components
├── dataLoader.py        # Data loading and preprocessing utilities
├── efficiency.py        # Efficiency analysis and benchmarking tools
├── config.py            # Configuration parameters and hyperparameters
├── framework.png        # Framework overview diagram
├── utils/               # Utility functions
│   ├── Setup.py        # Environment and experiment setup utilities
│   ├── Metric.py       # Evaluation metrics implementation
│   └── Optim.py        # Optimization utilities
├── dataset/            # Dataset storage directory
└── checkpoint/         # Model checkpoint storage directory
```

## 🔮 Getting Started

### ⚙️ Environment Setup

**Hardware Requirements:**  
- Tested on NVIDIA RTX 4090 GPUs (24GB VRAM)

**Software Requirements:**
- Python 3.11.7
- PyTorch 1.13.1
- torch_geometric 2.2.0
- numpy 1.21.6

**Installation:**
1. Install PyTorch and dependencies from [pytorch.org](http://pytorch.org)
2. Install PyTorch Geometric following the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

### ⚡️ Training and Evaluation

To train and evaluate the model:

```bash
python main.py
```

For detailed configuration options, please refer to:
- `config.py`: Model and training hyperparameters
- `utils/Setup.py`: Experiment setup and environment configuration

### 📊 Datasets

The implementation supports multiple datasets with an 8:1:1 train-validation-test split ratio. Due to GitHub's file size limitations (25MB), we currently provide:
1. Christianity dataset (included in this repository)
2. Raw Weibo dataset (available at [AMiner](www.aminer.cn/influencelocality))


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📝 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{SILN2025,
  title={Efficient Sphere-Effect Based Information Diffusion Prediction on Large-scale Social Networks},
  author={Zihan Feng, Yajun Yang, Xin Huang, Hong Gao, Liping Jing, Qinghua Hu},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```

## 📧 Contact

For any questions or concerns, please open an issue in this repository.
