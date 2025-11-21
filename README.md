# 🏠 Mortgage Credit Risk Modelling: Assessing Performance and Explainability

**Out-of-Universe Evaluation of Credit Scoring for Banking Using Machine Learning**

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 🌟 Project Overview

This repository contains the full source code and documentation for the academic research project: **"Mortgage Credit Risk Modelling: Assessing Performance under Out-of-Universe Evaluation."**

The project focuses on developing, evaluating, and demonstrating a machine learning (ML) model—specifically a **Random Forest Classifier**—to predict mortgage default risk (DFlag). A key component of this research is the deployment of a proof-of-concept application to address the critical regulatory and ethical challenges of **model explainability** and **financial inclusion** in credit decision-making, particularly in the **South African context**.

### 🎓 Academic Context

**Project Title**: Out-of-Universe Evaluation of Credit Scoring for Banking Using Machine Learning

**Author**: Ayanda Mhlauli

**Institution**: Department of Economics and Finance, The University of the Free State, South Africa

**Degree**: B.Com Honours in Business and Financial Analytics

**Year**: 2025

---

## 📋 Table of Contents

- [Key Objectives](#-key-objectives)
- [Deployed Application](#-deployed-application)
- [Repository Structure](#️-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Information](#-model-information)
- [Features](#-features)
- [Dependencies](#-dependencies)
- [How to Reproduce](#-how-to-reproduce-and-run-locally)
- [Citation](#-citation-and-references)
- [Contributing](#-contributing)
- [License](#-license)
- [Disclaimer](#️-disclaimer)
- [Contact](#-contact--support)

---

## 🎯 Key Objectives

1. **Develop**: Create a high-performance Random Forest model for binary classification (Default vs. No Default)

2. **Evaluate**: Rigorously test the model's performance using metrics like AUC-ROC against traditional statistical models

3. **Explain**: Integrate a system of feature contribution visualisation (simulated SHAP values) to address the National Credit Act's requirement for explaining credit denial decisions

4. **Discuss**: Analyse the trade-offs between ML complexity and regulatory explainability, and the implications for financial inclusion

---

## 🚀 Deployed Application (Proof-of-Concept)

The final, validated model is deployed as an interactive application to showcase its practical utility.

| Component | Description |
|-----------|-------------|
| **Application Title** | Mortgage Credit Predictor: Loan Approval Simulator |
| **Deployment Tool** | Streamlit |
| **Live URL** | [https://mortgage-credit-predictor-loan-approval-simulator-hwhnfxzbheat.streamlit.app/](https://mortgage-credit-predictor-loan-approval-simulator-hwhnfxzbheat.streamlit.app/) |

### Application Features

The application allows users to:
- Input core applicant metrics (Credit Score, DTI, LTV, etc.)
- Instantly receive a predicted probability of default
- View a binary loan approval/denial decision
- Access visual explanations showing which factors (features) had the greatest positive or negative impact on the decision

---

## ⚙️ Repository Structure

```
mortgage-credit-risk-modelling/
│
├── main_thesis.pdf                    # Complete research paper (thesis document)
├── app.py                             # Streamlit web application code
├── OOU_DF_25_selected.csv            # Anonymised, cleaned dataset (25 features + 1 target)
├── best_random_forest_model.pkl      # Trained Random Forest model (serialised)
├── README.md                          # This file
├── requirements.txt                   # Python package dependencies
│
└── screenshots/                       # Application screenshots (optional)
    ├── main_interface.png
    └── decision_output.png
```

### File Descriptions

| File/Folder | Description |
|-------------|-------------|
| `main_thesis.pdf` | The complete, final research paper (thesis document) |
| `app.py` | Python code for the Streamlit web application with MockRFModel logic and prediction pipeline |
| `OOU_DF_25_selected.csv` | The anonymised, cleaned dataset used for training and testing the model (25 features + 1 target) |
| `best_random_forest_model.pkl` | The serialised, trained Random Forest model object |
| `README.md` | Project documentation (this file) |
| `requirements.txt` | List of necessary Python packages (streamlit, pandas, scikit-learn, numpy, plotly) |

---

## 🚀 Installation

### Prerequisites

To run the Streamlit application and reproduce the model's environment, you need:
- **Python 3.8+**
- The packages listed in `requirements.txt`

### Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/mortgage-credit-risk-modelling.git
cd mortgage-credit-risk-modelling
```

2. **Create a Virtual Environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Model File**
```bash
# Ensure best_random_forest_model.pkl is in the project root
# If not available, the app will use a simulated fallback model
```

---

## 💻 Usage

### Running the Application Locally

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Enter Applicant Information**: Use the sidebar controls to input applicant details
   - Debt-to-Income Ratio (0-60%)
   - Credit Score (300-850)
   - Loan-to-Value Ratio (50-100%)
   - Loan Term (15 or 30 years)
   - Mortgage Insurance (Yes/No)
   - Loan Servicer type
   - Property Location Risk

2. **Analyze Application**: Click the "🔍 Analyze Application" button

3. **Review Results**: 
   - View the default probability percentage
   - See the loan approval/denial decision
   - Examine the feature contribution chart
   - Review the application summary

### Example Workflow

**Typical Approved Applicant Profile:**
```python
Debt-to-Income: 35%
Credit Score: 720
Loan-to-Value: 80%
Loan Term: 30 years
Mortgage Insurance: Yes
Servicer: Standard
Location: Standard Area

Expected Result: LOAN GRANTED (Default Risk < 50%)
```

---

## 🤖 Model Information

### Architecture
- **Model Type**: Random Forest Classifier
- **Training Data**: OOU_DF_25_selected.csv (anonymised mortgage dataset)
- **Features**: 25 engineered features
- **Target Variable**: DFlag (Default Flag: 0 = No Default, 1 = Default)
- **Decision Threshold**: 50% default probability
- **Output**: Binary classification (Approve/Deny)

### Feature Engineering

The model uses a strict feature order with **25 variables**:

**Continuous Features (4):**
- `Debt_to_income`: Percentage of income allocated to debt payments
- `Credit_Score`: FICO credit score (300-850)
- `OLoan_to_value`: Original loan-to-value ratio
- `Loan_term`: Duration of the loan (15 or 30 years)

**Binary Categorical Features (21):**
- Servicer indicators: `Servicer_47`, `Servicer_35`, `Servicer_26`, `Servicer_19`, `Servicer_28`
- Metropolitan Statistical Area (MSA) indicators: `MSA_32820`, `MSA_36740`, `MSA_40380`, `MSA_35300`, `MSA_36084`
- Postal Code indicators: `PostalCode_20900`, `PostalCode_7600`, `PostalCode_25300`, `PostalCode_18900`, `PostalCode_60900`, `PostalCode_67000`, `PostalCode_23300`, `PostalCode_12300`, `PostalCode_33700`, `PostalCode_62700`
- `Mortgage_Insurance`: Presence of mortgage insurance

### Key Risk Factors (by importance)

1. **Credit Score** (inverse relationship: lower score = higher risk)
2. **Debt-to-Income Ratio** (higher ratio = higher risk)
3. **Loan-to-Value Ratio** (higher LTV = higher risk)
4. **High-Risk Servicer** (Servicer_47)
5. **Property Location Risk** (MSA_32820)
6. **Mortgage Insurance** (protective factor)

### Model Performance

The model is evaluated using:
- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve
- **Confusion Matrix**: True/False Positives and Negatives
- **Precision, Recall, F1-Score**: Classification performance metrics
- **Out-of-Universe Testing**: Performance on unseen data distributions

*(See `main_thesis.pdf` for detailed performance metrics and comparison with traditional statistical models)*

### Explainability

The application implements **simulated SHAP-style feature contributions** to:
- Comply with the National Credit Act's transparency requirements
- Provide clear explanations for credit decisions
- Enable applicants to understand the denial reasons
- Support fair lending practices

---

## ✨ Features

### Core Functionality
- **Intelligent Risk Assessment**: Uses machine learning to predict default probability
- **Automated Decisions**: Applies a 50% default probability threshold for loan approval
- **Feature Analysis**: Displays which factors contribute most to the decision
- **Model Flexibility**: Supports both trained models (pickle file) and simulated fallback
- **Responsive Design**: Mobile-friendly interface with professional styling
- **Real-time Predictions**: Instant feedback on loan applications

### Regulatory Compliance
- **Explainable AI**: Feature contribution charts for decision transparency
- **Fair Lending Support**: Objective, data-driven decision-making
- **Audit Trail**: Clear documentation of decision factors
- **South African Context**: Aligned with National Credit Act requirements

---

## 📦 Dependencies

Create a `requirements.txt` file with:

```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.14.0
scikit-learn>=1.3.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Reproduce and Run Locally

### Full Reproduction Steps

1. **Clone the Repository**
```bash
git clone https://github.com/yourusername/mortgage-credit-risk-modelling.git
cd mortgage-credit-risk-modelling
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Streamlit Application**

The `app.py` script contains the core simulation logic:

```bash
streamlit run app.py
```

This command will open the application in your default web browser (usually at `http://localhost:8501`)

4. **Explore the Dataset** (Optional)

```python
import pandas as pd

# Load the training dataset
df = pd.read_csv('OOU_DF_25_selected.csv')
print(df.head())
print(df.describe())
```

5. **Load and Test the Model** (Optional)

```python
import pandas as pd
import numpy as np

# Load the trained model
model = pd.read_pickle('best_random_forest_model.pkl')

# Make a prediction
sample_features = np.array([[35, 680, 0, 80, 0, 30, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
prediction = model.predict_proba(sample_features)
print(f"Default Probability: {prediction[0][1]:.2%}")
```

---

## 📚 Citation and References

If you reference this work in your own research, please cite the research document and the application.

### Streamlit Application Citation

```bibtex
@misc{MhlauliA2025App,
    author = {Mhlauli, A},
    title = {{M}ortgage {C}redit {P}redictor: {L}oan {A}pproval {S}imulator},
    howpublished = {\url{https://mortgage-credit-predictor-loan-approval-simulator-hwhnfxzbheat.streamlit.app/}},
    year = {2025},
    note = {[Accessed 21-11-2025]},
}
```

### Academic Research Citation

```bibtex
@mastersthesis{MhlauliA2025Thesis,
    author = {Mhlauli, Ayanda},
    title = {Mortgage Credit Risk Modelling: Assessing Performance under Out-of-Universe Evaluation},
    school = {University of the Free State},
    year = {2025},
    type = {B.Com Honours Thesis},
    address = {Bloemfontein, South Africa},
    note = {Department of Economics and Finance}
}
```

*(Refer to `main_thesis.pdf` for the full bibliographic details of the academic paper)*

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting
- Ensure compliance with fair lending principles

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Ayanda Mhlauli

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## ⚠️ Disclaimer

**IMPORTANT**: This application is designed for **research and educational purposes only**. It should NOT be used for:

- Making actual lending decisions without proper validation
- Replacing professional credit analysis
- Financial advice or recommendations
- Production lending systems without regulatory approval

The model predictions are based on historical data and may not reflect current market conditions, regulatory requirements, or fair lending practices.

### Ethical Considerations

This research addresses critical issues in responsible AI deployment:

1. **Fair Lending Compliance**: Ensure adherence to anti-discrimination laws
2. **Model Transparency**: Provide clear explanations for credit decisions
3. **Financial Inclusion**: Balance risk management with access to credit
4. **Regulatory Alignment**: Meet National Credit Act requirements (South Africa)
5. **Bias Auditing**: Regular testing for disparate impact on protected groups
6. **Human Oversight**: Maintain human review for edge cases and appeals

### Regulatory Context (South Africa)

The National Credit Act (NCA) requires that:
- Credit providers must provide reasons for credit denial
- Scoring models must be transparent and explainable
- Consumers have the right to challenge credit decisions
- Credit assessment must be fair and non-discriminatory

This application demonstrates how ML models can meet these requirements through explainable AI techniques.

---

## 📞 Contact & Support

**Author**: Ayanda Mhlauli

**Institution**: University of the Free State, South Africa

**Department**: Economics and Finance

**GitHub**: [@Acemhlauli]([https://github.com/yourusername](https://github.com/Acemhlauli/Mortgage-Credit-Predictor-Loan-Approval-Simulator/))

**Issues**: [Report bugs or request features](https://github.com/yourusername/mortgage-credit-risk-modelling/issues)

For academic inquiries or collaboration opportunities, please use the contact information provided in `main_thesis.pdf`.

---

## 🙏 Acknowledgments

This project was completed in fulfilment of B.Com Honours in Business and Financial Analytics at the Department of Economics and Finance, The University of the Free State, South Africa.

**Special Thanks To:**
- The University of the Free State, Department of Economics and Finance
- Thesis supervisors and academic advisors
- Contributors to open-source ML and data science tools

**Built With:**
- [Streamlit](https://streamlit.io/) - Web application framework
- [Plotly](https://plotly.com/) - Interactive visualizations
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [NumPy](https://numpy.org/) - Numerical computing

---

## 📈 Future Enhancements

- [ ] Implement actual SHAP values for precise feature explanations
- [ ] Add A/B testing framework for model comparison
- [ ] Include batch prediction functionality
- [ ] Develop a comprehensive model monitoring dashboard
- [ ] Implement automated model retraining pipeline
- [ ] Add PDF report generation for loan decisions
- [ ] Include fairness metrics and bias detection tools
- [ ] Expand to support multiple credit products
- [ ] Integrate with real-time credit bureau data
- [ ] Add multi-language support for broader accessibility

---

## 📊 Research Impact

This research contributes to:

1. **Academic Literature**: Bridging the gap between ML performance and regulatory compliance
2. **Industry Practice**: Demonstrating practical implementation of explainable credit scoring
3. **Policy Development**: Informing fair lending regulations in emerging markets
4. **Financial Inclusion**: Improving access to credit through objective assessment
5. **Responsible AI**: Showcasing ethical ML deployment in high-stakes domains

---

**⭐ If you find this project useful for your research or practice, please consider citing it and giving it a star!**

*Last Updated: November 2024*

---

**Project Status**: ✅ Complete and Deployed

**Live Application**: [Mortgage Credit Predictor](https://mortgage-credit-predictor-loan-approval-simulator-hwhnfxzbheat.streamlit.app/)
