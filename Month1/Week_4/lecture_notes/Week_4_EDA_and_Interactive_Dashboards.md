# Month 1, Week 4: Exploratory Data Analysis & Interactive Dashboards

## 1. Executive Summary & Industry Context
Before building a neural network or training an XGBoost model, an AI engineer must understand the statistical distributions, correlations, class imbalances, and data corruptions hiding within the dataset.

Exploratory Data Analysis (EDA) is not merely drawing random charts; it is **a rigorous diagnostic process**. Furthermore, modern AI engineers present their findings and build model prototypes not through static PDF reports, but through **interactive web dashboards (Plotly & Streamlit)** that stakeholders can query in real time.

---

## 2. Canonical Textbook References
* **Storytelling with Data: A Data Visualization Guide for Business Professionals** by Cole Nussbaumer Knaflic.
* **Hands-On Machine Learning with Scikit-Learn (3rd Edition)** by Aurélien Géron, Chapter 2: *End-to-End Machine Learning Project — Discover and Visualize the Data*.

---

## 3. Theoretical Foundations: Statistical Profiling & Outlier Filtering

### Statistical Summaries & Distribution Shapes
* **Five-Number Summary:** Minimum, 25th percentile ($Q_1$), Median ($Q_2$), 75th percentile ($Q_3$), Maximum.
* **Interquartile Range (IQR):**
  $$\text{IQR} = Q_3 - Q_1$$
* **Tukey's Fences for Outlier Detection:**
  $$\text{Lower Bound} = Q_1 - 1.5 \times \text{IQR}, \quad \text{Upper Bound} = Q_3 + 1.5 \times \text{IQR}$$
  * Any point beyond these bounds is an outlier candidate requiring investigation.

### Correlation Analysis: Pearson vs. Spearman
* **Pearson Correlation ($r$):** Measures the strength of a **linear** relationship between two continuous variables ($[-1.0, +1.0]$). Highly sensitive to extreme outliers.
* **Spearman Rank Correlation ($\rho$):** Measures monotonic relationships using ranks rather than raw values. Robust to outliers and non-linear monotonic patterns.

---

## 4. Complete Step-by-Step Code Walkthrough

### 1. Interactive Visualization with Plotly Express
```python
import plotly.express as px
import polars as pl
import numpy as np

# Load transaction data
df = pl.read_parquet("transactions.parquet").sample(10000).to_pandas()

# Interactive Distribution Plot with Marginal Box Plot
fig = px.histogram(
    df, 
    x="transaction_amount", 
    color="category", 
    marginal="box",
    title="Transaction Distribution across Categories (with Outliers)",
    hover_data=["country_code"],
    opacity=0.7,
    nbins=50
)
fig.update_layout(template="plotly_dark")
fig.write_html("eda_distribution.html")
print("Interactive chart saved to eda_distribution.html")
```

### 2. Live Stakeholder Dashboard in Streamlit (`app.py`)
```python
import streamlit as st
import polars as pl
import plotly.express as px

st.set_page_config(page_title="AI Feature & Data Explorer", layout="wide")

st.title("📊 Enterprise AI Feature Profiler")
st.markdown("Inspect distributions, filter anomalies, and assess feature correlation in real time.")

# Sidebar Filters
st.sidebar.header("Filter Criteria")
uploaded_file = st.sidebar.file_uploader("Upload Parquet/CSV Dataset", type=["parquet", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".parquet"):
        df = pl.read_parquet(uploaded_file).to_pandas()
    else:
        df = pl.read_csv(uploaded_file).to_pandas()
        
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Features Count", len(df.columns))
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    # Feature selector
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select Feature for Distribution Analysis", numeric_cols)
        
        # Interactive Plot
        fig = px.box(df, y=selected_col, points="outliers", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Please upload a dataset or explore demo data.")
```

---

## 5. Common Student Pitfalls & Debugging Strategies

1. **Assuming Correlation Implies Causation:**
   * *The Mistake:* Finding high correlation between two features (e.g., ice cream sales and shark attacks) and assuming one causes the other (both driven by summer temperature).
   * *The Fix:* Always cross-validate domain plausibility before building predictive models on raw correlations.
2. **Blindly Deleting Outliers:**
   * *The Mistake:* Removing all data points beyond $1.5 \times \text{IQR}$. In fraud detection or cybersecurity, outliers are often the exact positive class you need to detect!
   * *The Fix:* Differentiate between data entry errors (e.g., `age = 999`) and genuine high-variance events.
3. **Streamlit State Resets:**
   * *The Fix:* Understand that Streamlit re-runs the entire Python script from top to bottom on every user click. Use `st.session_state` to store heavy computations and cached datasets.

---

## 6. Production Best Practices & Real-World Case Study
At AI-first startups and enterprise labs:
* Data scientists deploy internal **Streamlit / Dash** tools to enable domain experts (doctors, fraud investigators, underwriters) to visually inspect training data anomalies before model training begins.
* Catching bad labels or distribution drifts at the EDA stage saves weeks of failed model training cycles.

---

## 7. Hands-on Mini-Lab & Self-Assessment Checklist
- [ ] Install Plotly and Streamlit: `uv pip install plotly streamlit`
- [ ] Calculate the Five-Number Summary and IQR for a continuous feature.
- [ ] Plot an interactive distribution with Plotly Express showing marginal box plots.
- [ ] Run a local Streamlit dashboard: `streamlit run app.py`
- [ ] Add interactive slider widgets to filter the dataset dynamically.
