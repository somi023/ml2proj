# What makes equally accurate fine-tuned models disagree?

CS 536 (Machine Learning II, Rutgers, Fall 2026) semester project.

Fine-tune a pretrained model twice with different random seeds and you get two models with the same accuracy that still disagree on individual test images. This project measures how that seed-to-seed disagreement depends on how a pretrained vision transformer is adapted (linear head, LoRA at rank 4 and 64, or full fine-tuning), both in distribution and under distribution shift (Gaussian noise and blur, CIFAR-100-C). It also tests whether drift from the pretrained features predicts disagreement better than the number of trained parameters.

**Setup:** ViT-S/16 pretrained on ImageNet-21k (`vit_small_patch16_224.augreg_in21k`), a fixed 10,000-image CIFAR-100 training subset, 4 adaptation conditions × 5 seeds.

## Install

Python 3.12 is required (the numpy and scipy pins need it). On iLab, create the venv with `/common/system/venv/python312/bin/python`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Development

```bash
make setup   # pinned runtime + dev tools, and the git pre-commit hook (once per clone)
make lint    # ruff lint and format check
make test    # CPU unit tests, including a check that installed versions match requirements.txt
```

CI runs `make lint` and `make test` on every push and pull request. Every commit runs ruff, a large-file guard (500 kB) and a check for hard-coded home-directory paths.

## Status

Day 1 scaffold. Install, data, run and reproduce instructions are added as each piece lands.


## License

MIT
