

# SymptoGuard: An Explainable Machine Learning Framework for Multiclass Disease Diagnosis

## Introduction

**SymptoGuard** is an Explainable Artificial Intelligence (XAI)-powered disease prediction framework designed to assist in multiclass disease diagnosis using patient symptoms. The system applies machine learning algorithms to analyze binary symptom data and predict the most probable disease among 100 disease categories. To enhance transparency and trust, SymptoGuard integrates SHAP (SHapley Additive exPlanations), allowing users to understand which symptoms influenced each prediction. The framework is deployed as an interactive Streamlit web application, making it accessible for research, educational purposes, and AI-assisted healthcare demonstrations.

🌐 **Live Demo:** [https://symptoguard.streamlit.app/](https://symptoguard.streamlit.app/)



## What is SymptoGuard?

SymptoGuard is a research-based machine learning framework that predicts diseases from user-selected symptoms. It combines classical machine learning, explainable AI, and a web-based interface to create an interpretable clinical decision-support prototype.

The framework was developed using a processed version of the publicly available Kaggle Diseases and Symptoms dataset, containing over **101,000 patient records**, **227 binary symptom features**, and **100 disease classes**.

Users can access the deployed application online to explore symptom-based disease prediction with explainable AI:

**🔗 Website:** [https://symptoguard.streamlit.app/](https://symptoguard.streamlit.app/)



## How Does It Work?

The framework follows a complete machine learning pipeline:

* Collects and preprocesses symptom-based medical data.
* Converts symptoms into binary feature vectors.
* Trains and evaluates multiple machine learning models.
* Optimizes model performance using Optuna.
* Selects the best-performing model (Optimized Bernoulli Naive Bayes).
* Predicts the most likely disease based on user-selected symptoms.
* Generates SHAP-based explanations to show how each symptom contributed to the prediction.
* Displays results through the interactive Streamlit web application.



## Key Features

* Predicts **100 different diseases** from symptom inputs.
* Uses **227 binary symptom features** for classification.
* Compares multiple machine learning algorithms.
* Hyperparameter optimization using **Optuna**.
* SHAP-based **local and global explainability**.
* Interactive **Streamlit web application**.
* Fully reproducible research implementation.
* High prediction performance with **89.60% accuracy**.
* Searchable symptom selection interface.
* Real-time disease prediction with confidence information.
* User-friendly web deployment accessible from any modern browser.



## Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Optuna
* SHAP
* Streamlit
* XGBoost
* CatBoost
* LightGBM
* Matplotlib



## Performance

The final optimized **Bernoulli Naive Bayes** model achieved:

* **Accuracy:** **89.60%**
* **Weighted F1-Score:** **89.66%**
* **Macro F1-Score:** **89.62%**

Among all evaluated models, it provided the best balance between prediction accuracy, computational efficiency, and interpretability.



## Repository Contents

This repository includes:

* Dataset preprocessing scripts
* Model training and evaluation code
* Hyperparameter optimization
* Trained machine learning models
* SHAP explainability implementation
* Streamlit web application
* Documentation and reproducible research workflow



## Web Application

The trained model is publicly available through a Streamlit-based web application.

**🌐 Live Website:** [https://symptoguard.streamlit.app/](https://symptoguard.streamlit.app/)

The application allows users to:

* Select symptoms from a searchable list.
* Predict the most probable disease.
* View the model's confidence score.
* Understand the prediction through SHAP-based feature explanations.
* Explore an explainable AI-powered healthcare decision-support prototype.



## Applications

SymptoGuard can be used for:

* AI-assisted disease prediction research
* Explainable AI (XAI) demonstrations
* Machine learning education
* Healthcare decision-support research
* Academic and university projects
* Benchmarking multiclass classification models

> **Note:** This project is intended for research and educational purposes only and should not be used as a substitute for professional medical diagnosis.



## Future Work

Several improvements can further enhance the framework:

* Validate the model using real-world clinical datasets.
* Increase the number of supported diseases.
* Incorporate laboratory test results and patient medical history.
* Explore deep learning and transformer-based architectures.
* Develop Android and iOS mobile applications.
* Integrate Electronic Health Record (EHR) systems.
* Support multilingual interfaces.
* Improve prediction confidence estimation and uncertainty analysis.
* Evaluate the framework in real clinical environments with healthcare professionals.
* Deploy the system as a cloud-based clinical decision-support platform.



## Conclusion

SymptoGuard demonstrates that combining machine learning with Explainable Artificial Intelligence (XAI) can provide an effective and transparent framework for symptom-based disease prediction. By integrating high-performing classification models with SHAP-based explanations and a user-friendly web interface, the project bridges the gap between predictive accuracy and interpretability. The publicly available web application further demonstrates the practical implementation of the proposed framework, making it accessible for researchers, students, and AI enthusiasts. SymptoGuard serves as a reproducible research platform and provides a strong foundation for future advancements in AI-assisted healthcare and explainable clinical decision-support systems.

**🔗 Live Demo:** [https://symptoguard.streamlit.app/](https://symptoguard.streamlit.app/)
