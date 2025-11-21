import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Feature order - CRITICAL: Model expects features in this exact order
FEATURE_ORDER = [
    'Debt_to_income', 'Credit_Score', 'Servicer_47', 'OLoan_to_value', 'Servicer_35',
    'Loan_term', 'PostalCode_20900', 'MSA_32820', 'MSA_36740', 'MSA_40380',
    'PostalCode_7600', 'Mortgage_Insurance', 'PostalCode_25300', 'Servicer_26',
    'PostalCode_18900', 'Servicer_19', 'MSA_35300', 'PostalCode_60900', 'MSA_36084',
    'Servicer_28', 'PostalCode_67000', 'PostalCode_23300', 'PostalCode_12300',
    'PostalCode_33700', 'PostalCode_62700'
]

class MockRFModel:
    """Simulates a Random Forest classifier for mortgage default prediction"""
    
    def __init__(self):
        self.feature_names = FEATURE_ORDER
        
    def predict_proba(self, X):
        """
        Simulate RF model predictions based on key risk factors
        Returns: array of [P(no default), P(default)]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Extract key features by index
        debt_to_income = X[:, 0]  # Index 0
        credit_score = X[:, 1]     # Index 1
        servicer_47 = X[:, 2]      # Index 2
        loan_to_value = X[:, 3]    # Index 3
        servicer_35 = X[:, 4]      # Index 4
        loan_term = X[:, 5]        # Index 5
        msa_32820 = X[:, 7]        # Index 7
        mortgage_ins = X[:, 11]    # Index 11
        
        # Risk scoring logic (higher score = higher default risk)
        risk_score = np.zeros(len(X))
        
        # Debt-to-income: normalized 0-60 range
        risk_score += (debt_to_income / 60.0) * 0.30
        
        # Credit score: inverse normalized (lower score = higher risk)
        risk_score += ((850 - credit_score) / 550.0) * 0.35
        
        # Loan-to-value: normalized 50-100 range
        risk_score += ((loan_to_value - 50) / 50.0) * 0.20
        
        # High-risk servicer (significant penalty)
        risk_score += servicer_47 * 0.25
        
        # Alternative servicer (moderate penalty)
        risk_score += servicer_35 * 0.12
        
        # High-risk location
        risk_score += msa_32820 * 0.15
        
        # Longer loan term (slight increase)
        risk_score += (loan_term == 30) * 0.08
        
        # Mortgage insurance (protective factor)
        risk_score -= mortgage_ins * 0.10
        
        # Add some realistic randomness
        np.random.seed(42)
        noise = np.random.normal(0, 0.02, len(X))
        risk_score += noise
        
        # Clip to [0, 1] probability range
        default_prob = np.clip(risk_score, 0, 1)
        no_default_prob = 1 - default_prob
        
        # Return as [P(class 0), P(class 1)]
        return np.column_stack([no_default_prob, default_prob])

@st.cache_resource
def load_model():
    """
    Load the Random Forest model
    First tries to load from pickle file, falls back to mock model if unavailable
    """
    try:
        # Attempt to load the actual trained model
        model = pd.read_pickle("best_random_forest_model.pkl")
        st.sidebar.success("✅ Loaded trained model from pickle file")
        return model, True  # Return model and flag indicating it's real
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Pickle file not found. Using simulated model.")
        return MockRFModel(), False  # Return mock model and flag
    except Exception as e:
        st.sidebar.error(f"⚠️ Error loading pickle file: {str(e)}. Using simulated model.")
        return MockRFModel(), False

def create_feature_vector(user_inputs):
    """
    Create 25-feature vector from user inputs following FEATURE_ORDER
    """
    feature_dict = {feature: 0.0 for feature in FEATURE_ORDER}
    
    # Map user inputs to feature vector
    feature_dict['Debt_to_income'] = user_inputs['debt_to_income']
    feature_dict['Credit_Score'] = user_inputs['credit_score']
    feature_dict['OLoan_to_value'] = user_inputs['loan_to_value']
    feature_dict['Loan_term'] = user_inputs['loan_term']
    feature_dict['Mortgage_Insurance'] = 1.0 if user_inputs['mortgage_insurance'] else 0.0
    
    # Servicer mapping
    if user_inputs['servicer'] == 'High-Risk Servicer':
        feature_dict['Servicer_47'] = 1.0
    elif user_inputs['servicer'] == 'Alternative Servicer':
        feature_dict['Servicer_35'] = 1.0
    
    # Location mapping
    if user_inputs['location'] == 'High-Risk Area':
        feature_dict['MSA_32820'] = 1.0
    
    # Create ordered feature vector
    feature_vector = [feature_dict[feature] for feature in FEATURE_ORDER]
    return np.array(feature_vector).reshape(1, -1)

def calculate_feature_contributions(user_inputs, default_prob):
    """
    Simulate SHAP-style feature contributions
    Positive values push toward default, negative values push away from default
    """
    contributions = {}
    
    # Base rate (average default probability)
    base_rate = 0.15
    
    # Calculate contributions based on feature values
    # Debt-to-income contribution
    dti_risk = (user_inputs['debt_to_income'] - 35) / 60.0 * 0.08
    contributions['Debt-to-Income Ratio'] = dti_risk
    
    # Credit score contribution (inverse relationship)
    credit_risk = (680 - user_inputs['credit_score']) / 550.0 * 0.10
    contributions['Credit Score'] = credit_risk
    
    # Loan-to-value contribution
    ltv_risk = (user_inputs['loan_to_value'] - 80) / 50.0 * 0.06
    contributions['Loan-to-Value Ratio'] = ltv_risk
    
    # Loan term contribution
    term_risk = 0.03 if user_inputs['loan_term'] == 30 else -0.02
    contributions['Loan Term'] = term_risk
    
    # Servicer contribution
    if user_inputs['servicer'] == 'High-Risk Servicer':
        contributions['Loan Servicer'] = 0.12
    elif user_inputs['servicer'] == 'Alternative Servicer':
        contributions['Loan Servicer'] = 0.05
    else:
        contributions['Loan Servicer'] = -0.02
    
    # Location contribution
    if user_inputs['location'] == 'High-Risk Area':
        contributions['Property Location'] = 0.08
    else:
        contributions['Property Location'] = -0.01
    
    # Mortgage insurance contribution
    contributions['Mortgage Insurance'] = -0.05 if user_inputs['mortgage_insurance'] else 0.02
    
    return contributions

def plot_feature_contributions(contributions):
    """Create a horizontal bar chart showing feature contributions"""
    # Sort by absolute value for better visualization
    sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    # Color coding: red for positive (increase risk), green for negative (decrease risk)
    colors = ['#e74c3c' if v > 0 else '#27ae60' for v in values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=features,
        x=values,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Feature Contributions to Default Risk',
        xaxis_title='Impact on Default Probability',
        yaxis_title='',
        height=400,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    return fig

def main():
    # Page configuration
    st.set_page_config(
        page_title="Mortgage Credit Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS styling
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stButton>button {
            width: 100%;
            background-color: #2e86de;
            color: white;
            font-weight: bold;
            padding: 0.75rem;
            border-radius: 8px;
            border: none;
            font-size: 1.1rem;
        }
        .stButton>button:hover {
            background-color: #1e5799;
        }
        .decision-box {
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            border: 3px solid;
        }
        .approved {
            background-color: #d4edda;
            border-color: #28a745;
            color: #155724;
        }
        .denied {
            background-color: #f8d7da;
            border-color: #dc3545;
            color: #721c24;
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2e86de;
            margin: 0.5rem 0;
        }
        h1 {
            color: #2c3e50;
            padding-bottom: 1rem;
            border-bottom: 3px solid #3498db;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🏠 Mortgage Credit Predictor: Loan Approval Simulator")
    
    st.markdown("""
    This application uses a simulated machine learning model to predict mortgage default risk 
    and determine loan approval decisions based on applicant characteristics.
    """)
    
    # Load model
    model, is_real_model = load_model()
    
    # Sidebar for user inputs
    st.sidebar.header("📋 Applicant Information")
    
    # Collect user inputs
    debt_to_income = st.sidebar.slider(
        "Debt-to-Income Ratio (%)",
        min_value=0,
        max_value=60,
        value=35,
        step=1,
        help="Percentage of gross monthly income that goes toward paying debts"
    )
    
    credit_score = st.sidebar.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        value=680,
        step=5,
        help="FICO credit score (300-850 range)"
    )
    
    loan_to_value = st.sidebar.slider(
        "Loan-to-Value Ratio (%)",
        min_value=50,
        max_value=100,
        value=80,
        step=1,
        help="Loan amount divided by property value"
    )
    
    loan_term = st.sidebar.selectbox(
        "Loan Term (years)",
        options=[15, 30],
        index=1,
        help="Duration of the mortgage loan"
    )
    
    mortgage_insurance = st.sidebar.checkbox(
        "Mortgage Insurance",
        value=False,
        help="Whether the loan includes mortgage insurance"
    )
    
    servicer = st.sidebar.selectbox(
        "Loan Servicer",
        options=['Standard Servicer', 'High-Risk Servicer', 'Alternative Servicer'],
        index=0,
        help="The company servicing the mortgage loan"
    )
    
    location = st.sidebar.selectbox(
        "Property Location Risk",
        options=['Standard Area', 'High-Risk Area'],
        index=0,
        help="Risk profile of the property location"
    )
    
    # Prediction button
    predict_button = st.sidebar.button("🔍 Analyze Application")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    if predict_button:
        # Prepare user inputs
        user_inputs = {
            'debt_to_income': debt_to_income,
            'credit_score': credit_score,
            'loan_to_value': loan_to_value,
            'loan_term': loan_term,
            'mortgage_insurance': mortgage_insurance,
            'servicer': servicer,
            'location': location
        }
        
        # Create feature vector
        feature_vector = create_feature_vector(user_inputs)
        
        # Make prediction
        prediction_proba = model.predict_proba(feature_vector)
        default_prob = prediction_proba[0][1]  # P(Default = 1)
        
        # Decision threshold
        THRESHOLD = 0.20
        
        with col1:
            st.subheader("📊 Risk Assessment")
            
            # Display probability
            st.markdown(f"""
            <div class="metric-container">
                <h3 style="margin:0; color:#2e86de;">Default Probability</h3>
                <h2 style="margin:0.5rem 0; color:#e74c3c;">{default_prob*100:.2f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Display decision
            if default_prob > THRESHOLD:
                decision = "LOAN DENIED"
                decision_class = "denied"
                explanation = f"The predicted default risk ({default_prob*100:.2f}%) exceeds the acceptable threshold of {THRESHOLD*100:.0f}%."
            else:
                decision = "LOAN GRANTED"
                decision_class = "approved"
                explanation = f"The predicted default risk ({default_prob*100:.2f}%) is within the acceptable threshold of {THRESHOLD*100:.0f}%."
            
            st.markdown(f"""
            <div class="decision-box {decision_class}">
                {decision}
            </div>
            """, unsafe_allow_html=True)
            
            st.info(explanation)
            
            # Display input summary
            with st.expander("📝 Application Summary"):
                st.write(f"**Debt-to-Income Ratio:** {debt_to_income}%")
                st.write(f"**Credit Score:** {credit_score}")
                st.write(f"**Loan-to-Value Ratio:** {loan_to_value}%")
                st.write(f"**Loan Term:** {loan_term} years")
                st.write(f"**Mortgage Insurance:** {'Yes' if mortgage_insurance else 'No'}")
                st.write(f"**Loan Servicer:** {servicer}")
                st.write(f"**Property Location:** {location}")
        
        with col2:
            st.subheader("🎯 Feature Analysis")
            
            # Calculate and display feature contributions
            contributions = calculate_feature_contributions(user_inputs, default_prob)
            fig = plot_feature_contributions(contributions)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - 🔴 **Red bars** indicate features increasing default risk
            - 🟢 **Green bars** indicate features decreasing default risk
            - Longer bars have stronger impact on the decision
            """)
    
    else:
        # Initial state - show instructions
        st.info("👈 Please enter applicant information in the sidebar and click **'Analyze Application'** to see the prediction.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📌 How It Works")
            st.markdown("""
            1. **Enter applicant details** in the sidebar
            2. **Click 'Analyze Application'** to run the prediction
            3. **Review the decision** and risk assessment
            4. **Examine feature contributions** to understand the decision
            """)
        
        with col2:
            st.subheader("⚙️ Model Information")
            model_type = "Trained Random Forest" if is_real_model else "Simulated Random Forest"
            st.markdown(f"""
            - **Model Type:** {model_type}
            - **Decision Threshold:** 20% default probability
            - **Key Risk Factors:** Credit score, debt-to-income ratio, loan-to-value ratio
            - **Features Analysed:** 25 total features, including location and servicer data
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <small>⚠️ This is a simulated model for research and educational purposes only. 
        Not suitable for actual lending decisions.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
