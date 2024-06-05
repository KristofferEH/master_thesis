# Master thesis ⚗️ 
Repository for my master's thesis in chemical engineering.


| Title                 | **Advancing Solvent Development for CO2 Capture with Deep Learning**                                 |
|-----------------------|------------------------------------------------------------------------------------|
| Subtitle              | **Developing Domain-Reinforced Graph Neural Networks for VLE Predictions of Aqueous Amines** |
| Author                | **Kristoffer Elstad Hansen**                                                                   |
| Supervisor            | **Hanna Katariina Knuutila**                                                      |
| Co-supervisor         | **Idelfonso Bessa dos Reis Nogueira, Vinicius Viena Santana** |
| Date                  | **June 2024**                                                                     |


# Abstract

This thesis aims to combine traditional chemical engineering methods with advanced machine learning techniques. Traditional chemical engineering models, such as the Antoine equation for vapor pressure and the NRTL activity coefficient model, are derived from laboratory experiments and theoretical physics. These models have been validated over decades of research and are deeply rooted in our fundamental understanding of thermodynamics and molecular interactions. However, traditional methods often require specific parameters that are not available for new or poorly characterized compounds. 

In contrast, machine learning excels at pattern recognition and finding complex, non-linear relationships in data, allowing it to generate predictions for molecules it has never seen before. This capability is valuable for estimating properties, such as the vapor pressure, for new and poorly characterized compounds where experimental data is scarce. However, machine learning models often lack a fundamental grounding in the physical phenomena they aim to predict, which can result in physically impossible predictions, such as negative vapor pressures. To address this, this thesis focuses on developing domain-reinforced neural networks that utilize graph neural networks to analyze the molecular structure. By integrating the robust, data-driven capabilities of machine learning with the fundamental understanding of traditional chemical engineering, this approach constrains the model and ensures it makes thermodynamically consistent predictions.

In this work, a model predicting the saturation pressure of amines was developed by first training on a diverse range of compounds and then fine-tuning it specifically for amines. The model achieved a root mean square error (RMSE) of 0.221 log(mmHg), with 90\% of the residuals ranging from -0.31 to 0.36 log(mmHg). This specialized amine model significantly outperformed the model it was benchmarked against, highlighting the value of the model developed in this work. 

Furthermore, the saturation pressure model was extended to create a framework for predicting the VLE of aqueous amines. This model achieved a RMSE of 1.125 for predicting the amine vapor fraction and 0.136 for predicting the water fraction, with 90\% of the residuals falling within the range of -1.86 to 1.76 for the amine fraction and -0.17 to 0.27 for the water fraction. These errors are given as the natural logarithm of the vapor fractions. Due to the lack of publicly available models for predicting VLE data based on molecular structures, no suitable model was found to benchmark against. This highlights the need for further development of robust and reliable models within this field. The model developed in this thesis could be used as a foundation for future research comparisons. Although both the volatility and VLE model could be further refined to enhance accuracy and reliability, this work demonstrates that machine learning can indeed be used to capture the complex relationship between phase behavior and molecular structures.

## Requirements
Run the following command to install all of the packages listed in `environment.yml`. Note that this assumes that you have conda installed. 

```bash
conda env create -f environment.yml
```

## Description

The folders "General GNN model," "Specialized GNN model," and "VLE model" contain the developed machine-learning models and the code used to create them. To run these models and generate predictions, use the "run_models.ipynb" file. This file is designed to utilize the machine-learning models for prediction purposes.

Please note that the file paths in the "run_models.ipynb" are configured to match my environment. To use these models on your computer, you will need to update these paths to correspond to your own setup. Specifically, you need to modify the code to indicate where you have stored the machine-learning models from the folders mentioned above on your system.

Additionally, the code used to preprocess my datasets is located in the folder "Code for Preprocessing." However, since I have not published the data used to develop the machine-learning models due to confidentiality concerns, these files might not be very useful. Nonetheless, I chose to include them just in case.
