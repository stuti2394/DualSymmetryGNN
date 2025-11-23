# DualSymmetryGNN

A Graph Neural Network (GNN) based model for detecting and quantifying symmetry in polygonal objects.

## Overview

Traditional symmetry detection methods rely on geometric rules and transformations, but they often fail when data is noisy, irregular, or complex.
DualSymmetryGNN addresses this by representing polygons as graphs and applying Graph Neural Networks (GIN/GCN) to learn both local and global structural patterns.

This project was awarded **3rd Prize at the Spark 2025 Internship Program**.

---

## Features

* Conversion of polygonal objects into graph representations (nodes = coordinates, edges = structural connections)
* Implementation of multiple combinations of various GNN layer types
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

* MAE: 0.2322
* RMSE: 0.7347
* R2: 0.9081
* Exact: 0.9107
* BinAcc: 0.9920
* BinF1: 0.9130
* ConfMat: [[479   3]
           [  1  21]]

<img width="555" height="455" alt="image" src="https://github.com/user-attachments/assets/fa000dff-3144-4f03-bf25-1ea2a2647a7d" />

<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/fc10a5ee-dba2-4612-b673-cb700ec30f97" />



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
