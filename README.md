# Cosine Fusion

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/CookieMonsteriOS/CosineFusion)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/CookieMonsteriOS/CosineFusion/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)]()


## Overview
Cosine Fusion is a lightweight C++/Python cosine similarity engine using **pybind11**. It provides fast, vectorized similarity calculations between user and item feature matrices.

## Features
- C++ backend for high performance
- Python interface via pybind11
- Easy integration into Python projects
- Simple example with user-item preference vectors

## Use Cases / Applications
- Recommender systems for e-commerce or media content
- Personalization engines based on user preferences
- Fast similarity search for AI/ML feature matching
- Any project requiring high-performance cosine similarity computation

## Installation
Clone the repository and install the package:

```bash
git clone https://github.com/CookieMonsteriOS/CosineFusion.git
cd CosineFusion
pip install pybind11
pip install .
```

## Usage Example
```python
import numpy as np
import core_init


# Example item features
items = np.array([
[0, 1, 0, 0, 1], # Tea
[0, 1, 0, 0, 1], # Coffee
[1, 0, 1, 1, 0], # Jaffa Cake
[1, 0, 1, 0, 0], # Biscuit
[1, 0, 1, 1, 1], # Chocolate Bar
[0, 1, 0, 0, 1], # Espresso
])


item_names = ["Tea", "Coffee", "Jaffa Cake", "Biscuit", "Chocolate Bar", "Espresso"]


user = np.array([[0, 0, 0, 1, 0]]) # User likes sweet + chocolate
res = core_init.cosine_similarity(user, items)
sim = res["similarity_matrix"]


top_indices = np.argsort(sim[0])[::-1]
print("Top recommendations for user:")
for i in top_indices:
print(f"{item_names[i]}: {sim[0][i]:.2f}") # Sample output showing relationships:

# Top recommendations for user:
# Chocolate Bar: 0.89
# Jaffa Cake: 0.75 <- shows some similarity to Tea in sweetness
# Biscuit: 0.65
# Tea: 0.45
# Coffee: 0.45
# Espresso: 0.43
```

## Project Structure
```
CosineFusion/
 ├── src/
 │    ├── cpp/core_init.cpp
 │    └── python/core_demo.py
 ├── tests/test_bridge.py
 ├── setup.py
 ├── pyproject.toml
 ├── README.md
 ├── LICENSE
 └── requirements.txt
```

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---
**Author:** Sam Chaudry  
**GitHub:** [CookieMonsteriOS](https://github.com/CookieMonsteriOS)
