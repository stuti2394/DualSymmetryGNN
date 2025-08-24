# DualSymmetryGNN

A Graph Neural Network (GNN) based model for detecting and quantifying symmetry in polygonal objects.

## Overview

Traditional symmetry detection methods rely on geometric rules and transformations, but they often fail when data is noisy, irregular, or complex.
DualSymmetryGNN addresses this by representing polygons as graphs and applying Graph Neural Networks (GIN/GCN) to learn both local and global structural patterns.

This project was awarded **3rd Prize at the Spark 2025 Internship Program**.

---

## Features

* Conversion of polygonal objects into graph representations (nodes = coordinates, edges = structural connections)
* Implementation of Graph Convolutional Networks (GCN) and Graph Isomorphism Networks (GIN)
* Custom node pooling strategies to capture repeating and symmetric patterns
* Multi-task learning setup:

  * Symmetry Classification (predicting number of symmetry axes)
  * Symmetry Quantification (measuring degree of symmetry)
* Hyperparameter optimization with Optuna
* Comparison against traditional rule-based symmetry detection methods

---

## Tech Stack

* Python
* PyTorch
* PyTorch Geometric (PyG)
* Optuna (for hyperparameter tuning)
* NetworkX (graph representation)
* NumPy / Pandas / Matplotlib (data handling and visualization)

---

## Results

* Demonstrated robustness on noisy and irregular polygons
* Outperformed traditional geometric rule-based methods
* Achieved accurate classification of symmetry axes and reliable quantification of symmetry degree

---

## Future Work

* Extend the approach to 3D geometric objects
* Explore graph transformers for enhanced structural learning
* Build a web-based tool for symmetry detection

---

## Acknowledgements

This project was developed as part of the Spark 2025 Internship Program, where it received the **3rd Prize**. Guidance from mentors and peers contributed greatly to its success.

---

## Contact

**Stuti Srivastava**
Email: [stutisrivastava0923@gmail.com](mailto:stutisrivastava0923@gmail.com)
LinkedIn: [linkedin.com/in/stutisrivastava23](https://www.linkedin.com/in/stutisrivastava23)

---
