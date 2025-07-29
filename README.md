# 🍽️ Predictive Restaurant Recommender

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-red.svg)](https://xgboost.readthedocs.io/)

## 📋 Overview

A comprehensive machine learning system for restaurant recommendation that predicts customer preferences and order values using advanced ML algorithms. This project implements multiple models with F1 score evaluation and comprehensive tree visualizations to provide accurate and interpretable recommendations.

## 🎯 Key Features

- **Multi-Model Approach**: Random Forest, XGBoost, and Neural Network implementations
- **Comprehensive Evaluation**: F1 Score, accuracy, precision, recall, and regression metrics
- **Tree Visualizations**: Interactive decision tree plots and feature importance analysis
- **Advanced Feature Engineering**: 20+ engineered features including geographic, temporal, and behavioral patterns
- **Industry Standards**: Cross-validation, early stopping, and hyperparameter optimization
- **Business Insights**: Interpretable decision rules and prediction explanations

## 🏗️ Project Structure

```
DS Assignment - 2025/
├── 📓 restaurant_recommendation_engine.ipynb    # Main analysis notebook
├── 📊 Train/
│   ├── orders.csv                              # Training order data
│   ├── train_customers.csv                     # Customer information
│   ├── train_locations.csv                     # Location data
│   └── vendors.csv                             # Vendor details
├── 📊 Test/
│   ├── test_customers.csv                      # Test customer data
│   └── test_locations.csv                      # Test location data
├── 📄 SampleSubmission.csv                     # Submission format
├── 📋 VariableDefinitions.pdf                  # Data dictionary
└── 📖 README.md                                # Project documentation
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
pip install graphviz pydotplus dtreeviz  # For tree visualizations
```

### Running the Analysis

1. **Clone the repository**
   ```bash
   git clone https://github.com/SidddhantJain/Predictive-Restaurant-Recommender.git
   cd Predictive-Restaurant-Recommender
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook restaurant_recommendation_engine.ipynb
   ```

3. **Run all cells** to execute the complete analysis pipeline

## 🧠 Machine Learning Models

### 🌲 Random Forest Regressor
- **Purpose**: Baseline ensemble model with robust performance
- **Features**: 
  - 150 estimators with max depth 15
  - Comprehensive F1 score evaluation via target binning
  - Feature importance analysis
  - Out-of-bag error estimation

### ⚡ XGBoost Regressor
- **Purpose**: Advanced gradient boosting for superior accuracy
- **Features**:
  - Early stopping and regularization (L1/L2)
  - Learning curves and iteration optimization
  - Tree visualization and boosting analysis
  - F1 score with classification metrics

### 🧠 Neural Network (MLPRegressor)
- **Purpose**: Deep learning approach for complex patterns
- **Features**:
  - 4-layer architecture (256-128-64-32 neurons)
  - Adaptive learning rate and early stopping
  - Comprehensive evaluation including F1 score
  - Loss curve analysis and convergence monitoring

## 📊 Model Performance Metrics

| Model | RMSE | MAE | R² | F1 Score | Accuracy |
|-------|------|-----|----|---------:|----------|
| Random Forest | ✅ | ✅ | ✅ | ✅ | ✅ |
| XGBoost | ✅ | ✅ | ✅ | ✅ | ✅ |
| Neural Network | ✅ | ✅ | ✅ | ✅ | ✅ |

*All models include comprehensive evaluation with both regression and classification metrics*

## 🌳 Tree Visualization Features

- **🌲 Random Forest Trees**: Individual tree plots with depth analysis
- **⚡ XGBoost Trees**: Boosting sequence visualization  
- **🌳 Decision Trees**: Full tree structure with decision rules
- **📊 Feature Importance**: Cross-model comparison and ranking
- **📈 Depth Analysis**: Performance vs complexity optimization
- **🛤️ Decision Paths**: Sample prediction explanations

## 🔧 Feature Engineering

### 📍 Geographic Features
- Customer-vendor distance calculations
- Location clustering and density analysis
- Geographic proximity scoring

### ⏰ Temporal Features  
- Time-based ordering patterns
- Seasonal and weekly trends
- Peak hour identification

### 👤 Customer Behavior
- Order frequency and recency
- Preference learning and loyalty scoring
- Demographic segmentation

### 🏪 Vendor Characteristics
- Category and tag analysis
- Rating and popularity metrics
- Service area optimization

## 📈 Business Impact

### 🎯 Recommendation Accuracy
- **Improved Customer Satisfaction**: Higher prediction accuracy leads to better recommendations
- **Increased Revenue**: Optimized order value predictions
- **Enhanced User Experience**: Personalized restaurant suggestions

### 📊 Operational Insights
- **Feature Importance**: Identify key factors driving customer choices
- **Decision Rules**: Interpretable business logic from tree models
- **Performance Monitoring**: Comprehensive evaluation framework

## 🛠️ Technical Implementation

### Data Processing Pipeline
1. **Data Loading & Validation**: Robust data ingestion with error handling
2. **Feature Engineering**: Advanced feature creation and selection
3. **Model Training**: Multi-algorithm approach with cross-validation
4. **Evaluation**: Comprehensive metrics including F1 scores
5. **Visualization**: Interactive plots and decision tree analysis

### Code Quality
- **Modular Design**: Object-oriented model classes
- **Error Handling**: Robust exception management
- **Documentation**: Comprehensive inline documentation
- **Reproducibility**: Fixed random seeds and version control

## 📋 Requirements

### Core Dependencies
```
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
xgboost >= 1.5.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
jupyter >= 1.0.0
```

### Visualization Dependencies
```
graphviz >= 0.19.0
pydotplus >= 2.0.2
dtreeviz >= 1.3.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Siddhant Jain**
- GitHub: [@SidddhantJain](https://github.com/SidddhantJain)
- LinkedIn: [Siddhant Jain](https://linkedin.com/in/your-profile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Dataset provided by DS Assignment 2025
- Scikit-learn and XGBoost communities for excellent ML libraries
- Jupyter Project for interactive development environment

## 📊 Project Status

- ✅ **Data Analysis**: Complete
- ✅ **Feature Engineering**: Complete  
- ✅ **Model Development**: Complete
- ✅ **F1 Score Implementation**: Complete
- ✅ **Tree Visualizations**: Complete
- ✅ **Documentation**: Complete
- 🔄 **Model Deployment**: In Progress

---

⭐ **Star this repository if you found it helpful!**

📧 **Questions?** Open an issue or reach out directly!
