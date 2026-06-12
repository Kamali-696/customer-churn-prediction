import streamlit as st # type: ignore
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException

# =========================================================
# CONFIG
# =========================================================
API_URL = "https://customer-churn-prediction-duh0.onrender.com/predict"
REQUEST_TIMEOUT = 30  # seconds — Render free tier cold starts can take 20-30s

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# STYLES
# =========================================================
st.markdown("""
    <style>
    .main { background-color: #f8f9fb; }
    .block-container { padding-top: 1.5rem; max-width: 1100px; }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.8rem;
    }
    .app-header h1 { color: white; margin: 0; font-size: 1.9rem; font-weight: 700; }
    .app-header p { color: #dbeafe; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Section headers */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1f2937;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e5e7eb;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.65rem 2rem;
        border: none;
        width: 100%;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover { background-color: #1d4ed8; color: white; }

    /* Result card */
    .result-card {
        background-color: white;
        border-radius: 14px;
        padding: 1.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-top: 1.5rem;
        border-left: 6px solid #2563eb;
    }
    .result-card.risk-low { border-left-color: #16a34a; }
    .result-card.risk-medium { border-left-color: #d97706; }
    .result-card.risk-high { border-left-color: #dc2626; }

    /* Metric boxes */
    .metric-box {
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 1.1rem;
        text-align: center;
        border: 1px solid #e5e7eb;
    }
    .metric-box h4 {
        margin: 0 0 0.4rem 0;
        font-size: 0.85rem;
        color: #6b7280 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-box .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        color: #1f2937 !important;
    }

    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.25rem 0.9rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .risk-low-badge { background-color: #dcfce7; color: #16a34a; }
    .risk-medium-badge { background-color: #fef3c7; color: #d97706; }
    .risk-high-badge { background-color: #fee2e2; color: #dc2626; }

    /* Customer summary */
    .summary-box {
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border: 1px solid #e5e7eb;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #374151 !important;
    }
    .summary-box b { color: #1f2937 !important; }

    /* Sidebar */
    .sidebar-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS — risk-level color mapping (single source of truth)
# =========================================================
RISK_STYLES = {
    "Low":    {"color": "#16a34a", "badge_class": "risk-low-badge",    "card_class": "risk-low"},
    "Medium": {"color": "#d97706", "badge_class": "risk-medium-badge", "card_class": "risk-medium"},
    "High":   {"color": "#dc2626", "badge_class": "risk-high-badge",   "card_class": "risk-high"},
}
DEFAULT_RISK_STYLE = {"color": "#374151", "badge_class": "", "card_class": ""}


# =========================================================
# HEADER
# =========================================================
st.markdown("""
    <div class="app-header">
        <h1>📊 Customer Churn Prediction Dashboard</h1>
        <p>Predict customer churn risk using machine learning — enter the customer profile below.</p>
    </div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT FORM
# =========================================================
with st.form("churn_form"):

    # ---- Demographics ----
    st.markdown('<div class="section-header">👤 Demographics</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
    with col2:
        senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    with col3:
        partner = st.selectbox("Partner", ["Yes", "No"])
    with col4:
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    # ---- Services ----
    st.markdown('<div class="section-header">📞 Phone & Internet Services</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    with col3:
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    # ---- Add-on Services ----
    st.markdown('<div class="section-header">🎬 Add-on Services</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    with col2:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    with col3:
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    # ---- Account & Billing ----
    st.markdown('<div class="section-header">💳 Account & Billing</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col2:
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, step=1, value=0)
    with col3:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, step=0.5, value=0.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, step=0.5, value=0.0)

    # CLTV on its own row to avoid a cramped layout
    cltv = st.number_input("CLTV (Customer Lifetime Value)", min_value=0.0, max_value=10000.0, step=10.0, value=0.0)

    st.write("")
    submitted = st.form_submit_button("🔍 Predict Churn")


# =========================================================
# INPUT VALIDATION
# =========================================================
def validate_inputs(tenure, monthly_charges, total_charges):
    """Return a list of validation warning messages, empty if all good."""
    issues = []
    if tenure == 0 and total_charges > 0:
        issues.append("Tenure is 0 but Total Charges is greater than 0 — please double-check these values.")
    if total_charges < monthly_charges and tenure > 1:
        issues.append("Total Charges is less than Monthly Charges despite tenure > 1 month — please verify.")
    return issues


# =========================================================
# PREDICTION LOGIC
# =========================================================
if submitted:
    # Light validation — warn but still allow prediction (beginner-friendly)
    for warning_msg in validate_inputs(tenure, monthly_charges, total_charges):
        st.warning(f"⚠️ {warning_msg}")

    payload = {
        "Gender": gender,
        "Senior_Citizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure_Months": tenure,
        "Phone_Service": phone_service,
        "Paperless_Billing": paperless,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": total_charges,
        "CLTV": cltv,
        "Multiple_Lines": multiple_lines,
        "Internet_Service": internet_service,
        "Online_Security": online_security,
        "Online_Backup": online_backup,
        "Device_Protection": device_protection,
        "Tech_Support": tech_support,
        "Streaming_TV": streaming_tv,
        "Streaming_Movies": streaming_movies,
        "Contract": contract,
        "Payment_Method": payment_method
    }

    try:
        with st.spinner("⏳ Contacting prediction service... (this may take up to 30s on first request)"):
            response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)

        # ---- Handle HTTP status ----
        if response.status_code == 200:
            try:
                result = response.json()
            except ValueError:
                st.error("❌ The server returned an invalid response (not JSON). Please try again.")
                st.stop()

            # ---- Safely extract fields with defaults ----
            prediction = result.get("prediction")
            probability = result.get("churn_probability")
            risk_level = result.get("risk_level", "Unknown")

            if prediction is None or probability is None:
                st.error("❌ Prediction response is missing expected fields. Please check the API.")
                st.stop()

            # Ensure probability is a float in [0, 1]
            try:
                probability = float(probability)
            except (TypeError, ValueError):
                probability = 0.0
            probability = min(max(probability, 0.0), 1.0)

            # ---- Resolve risk styling ----
            style = RISK_STYLES.get(risk_level, DEFAULT_RISK_STYLE)

            # ---- Prediction Card ----
            card_class = f"result-card {style['card_class']}".strip()
            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.subheader("🎯 Prediction Result")

            col1, col2, col3 = st.columns(3)

            with col1:
                pred_label = "Will Churn" if prediction == 1 else "Will Stay"
                pred_icon = "⚠️" if prediction == 1 else "✅"
                st.markdown(
                    f'''<div class="metric-box">
                        <h4>Prediction</h4>
                        <p class="metric-value">{pred_icon} {pred_label}</p>
                    </div>''',
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f'''<div class="metric-box">
                        <h4>Churn Probability</h4>
                        <p class="metric-value">{probability:.1%}</p>
                    </div>''',
                    unsafe_allow_html=True
                )

            with col3:
                st.markdown(
                    f'''<div class="metric-box">
                        <h4>Risk Level</h4>
                        <p class="metric-value">
                            <span class="risk-badge {style["badge_class"]}">{risk_level}</span>
                        </p>
                    </div>''',
                    unsafe_allow_html=True
                )

            # ---- Probability Gauge ----
            st.write("")
            st.markdown("**Churn Probability Gauge**")
            st.progress(probability)
            st.caption(f"{probability:.1%} probability of churn — Risk Level: {risk_level}")

            st.markdown('</div>', unsafe_allow_html=True)

            # ---- Customer Summary ----
            st.markdown(
                f"""
                <div class="summary-box">
                <b>📋 Customer Summary</b><br><br>
                <b>Gender:</b> {gender} &nbsp;|&nbsp;
                <b>Senior Citizen:</b> {senior} &nbsp;|&nbsp;
                <b>Partner:</b> {partner} &nbsp;|&nbsp;
                <b>Dependents:</b> {dependents}<br>
                <b>Tenure:</b> {tenure} months &nbsp;|&nbsp;
                <b>Contract:</b> {contract} &nbsp;|&nbsp;
                <b>Payment Method:</b> {payment_method}<br>
                <b>Internet Service:</b> {internet_service} &nbsp;|&nbsp;
                <b>Monthly Charges:</b> ${monthly_charges:,.2f} &nbsp;|&nbsp;
                <b>Total Charges:</b> ${total_charges:,.2f}<br>
                <b>CLTV:</b> {cltv:,.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

        elif response.status_code == 422:
            st.error("❌ The API rejected the input data (422 Unprocessable Entity). "
                     "This usually means a field name or value type mismatch with the backend schema.")
            with st.expander("Show error details"):
                st.code(response.text)

        elif response.status_code >= 500:
            st.error("❌ The prediction server encountered an internal error (5xx). "
                     "Please try again in a moment.")
            with st.expander("Show error details"):
                st.code(response.text)

        else:
            st.error(f"❌ Prediction failed — server returned status code {response.status_code}.")
            with st.expander("Show error details"):
                st.code(response.text)

    except Timeout:
        st.error(
            "⏱️ Request timed out. The Render backend may be waking up from sleep "
            "(cold start can take 20-30s). Please try again in a few seconds."
        )

    except ConnectionError:
        st.error(
            "🔌 Could not connect to the prediction API. "
            "Please check that the backend URL is correct and the service is running."
        )

    except RequestException as e:
        st.error(f"❌ An unexpected error occurred while contacting the API: {e}")


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Built with Streamlit • Powered by FastAPI & Machine Learning • Telco Customer Churn Dataset")