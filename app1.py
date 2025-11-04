import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🛡️ Fraud Detection System Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 18px;'>Machine Learning Based Fraud Analytics using Random Forest</p>", unsafe_allow_html=True)
st.markdown("---")

# Load and preprocess data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("fraud_data.csv")
        
        # Convert datetime
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        
        # Clean is_fraud
        df['is_fraud'] = df['is_fraud'].astype(str).str.extract(r'(\d)').astype(int)
        
        # Clean text columns
        if 'merchant' in df.columns:
            df['merchant'] = df['merchant'].str.replace('"', '', regex=False)
        if 'job' in df.columns:
            df['job'] = df['job'].str.replace('"', '', regex=False)
        
        df.drop_duplicates(inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Preprocessing function
def preprocess_for_model(df):
    df_model = df.copy()
    
    # Feature engineering
    df_model['age'] = (pd.to_datetime("2020-01-01") - df_model['dob']).dt.days // 365
    
    # Drop columns
    cols_to_drop = ['trans_num', 'dob', 'trans_date_trans_time', 'merchant', 'first', 
                    'last', 'street', 'city', 'state', 'zip', 'job']
    df_model = df_model.drop(columns=[col for col in cols_to_drop if col in df_model.columns])
    
    # Encode categorical
    for col in df_model.select_dtypes(include=['object']).columns:
        if col != 'is_fraud':
            df_model[col] = pd.Categorical(df_model[col]).codes
    
    # Split features and target
    X = df_model.drop(columns=['is_fraud'])
    y = df_model['is_fraud']
    
    # Convert to numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    
    return X, y

# Train Random Forest model
@st.cache_data
def train_random_forest(test_size, df_hash):
    # Preprocess data
    X, y = preprocess_for_model(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42, stratify=y
    )
    
    # Impute and scale
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    return model, y_test, y_pred, y_pred_proba, feature_importance, X_test_scaled

# Load data
df = load_data()

if df is not None:
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Model Configuration")
        st.markdown("---")
        
        st.markdown("### 🤖 Algorithm")
        st.info("**Random Forest Classifier**\n\nEnsemble method with 100 decision trees for robust fraud detection.")
        
        st.markdown("---")
        test_size = st.slider("📊 Test Size (%)", 10, 40, 30, 5)
        
        st.markdown("---")
        st.markdown("### 📝 Dataset Info")
        st.info(f"**Total Records:** {len(df):,}\n\n**Fraud Cases:** {df['is_fraud'].sum():,}\n\n**Fraud Rate:** {(df['is_fraud'].sum()/len(df)*100):.2f}%")
    
    # Dataset Overview Section
    st.markdown("## 📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trans = len(df)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h2 style='color: white; margin: 0; font-size: 40px;'>📊</h2>
            <h2 style='color: white; margin: 10px 0;'>{total_trans:,}</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;'>Total Transactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fraud_count = df['is_fraud'].sum()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h2 style='color: white; margin: 0; font-size: 40px;'>⚠️</h2>
            <h2 style='color: white; margin: 10px 0;'>{fraud_count:,}</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;'>Fraud Cases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        fraud_rate = (fraud_count / len(df) * 100)
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h2 style='color: white; margin: 0; font-size: 40px;'>📉</h2>
            <h2 style='color: white; margin: 10px 0;'>{fraud_rate:.2f}%</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;'>Fraud Rate</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        total_amount = df['amt'].sum()
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h2 style='color: white; margin: 0; font-size: 40px;'>💰</h2>
            <h2 style='color: white; margin: 10px 0;'>${total_amount:,.0f}</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;'>Total Amount</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data Visualizations
    st.markdown("## 📈 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Fraud vs Non-Fraud Distribution")
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        fraud_counts = df['is_fraud'].value_counts()
        colors = ['#10b981', '#ef4444']
        
        wedges, texts, autotexts = ax.pie(
            fraud_counts.values, 
            labels=['Non-Fraud', 'Fraud'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 14, 'weight': 'bold'},
            pctdistance=0.85
        )
        
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')
        ax.set_title('Transaction Distribution', fontsize=16, weight='bold', pad=20)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### 📊 Fraud Rate by Category")
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        fraud_by_cat = df.groupby("category")["is_fraud"].mean().sort_values(ascending=False)
        colors_palette = sns.color_palette("viridis", len(fraud_by_cat))
        
        bars = ax.barh(fraud_by_cat.index, fraud_by_cat.values, color=colors_palette, edgecolor='black', linewidth=1)
        ax.set_xlabel("Fraud Rate", fontsize=12, weight='bold')
        ax.set_ylabel("Category", fontsize=12, weight='bold')
        ax.set_title("Fraud Rate by Transaction Category", fontsize=14, weight='bold', pad=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Fraud over time
    st.markdown("### 📅 Fraud Transactions Over Time")
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
    fraud_over_time = df.set_index("trans_date_trans_time").resample("D")["is_fraud"].sum()
    ax.plot(fraud_over_time.index, fraud_over_time.values, color='#ef4444', linewidth=2, marker='o', markersize=3)
    ax.fill_between(fraud_over_time.index, fraud_over_time.values, alpha=0.3, color='#ef4444')
    ax.set_xlabel("Date", fontsize=12, weight='bold')
    ax.set_ylabel("Number of Frauds", fontsize=12, weight='bold')
    ax.set_title("Daily Fraud Transaction Trend", fontsize=14, weight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Amount vs Population scatter
    st.markdown("### 🏙️ Transaction Amount vs City Population")
    fig, ax = plt.subplots(figsize=(14, 5), facecolor='white')
    fraud_data = df[df['is_fraud'] == 1]
    non_fraud_data = df[df['is_fraud'] == 0].sample(n=min(5000, len(df[df['is_fraud'] == 0])))
    
    ax.scatter(non_fraud_data['city_pop'], non_fraud_data['amt'], alpha=0.5, c='#10b981', label='Non-Fraud', s=30, edgecolors='black', linewidth=0.5)
    ax.scatter(fraud_data['city_pop'], fraud_data['amt'], alpha=0.7, c='#ef4444', label='Fraud', s=50, edgecolors='black', linewidth=0.5)
    ax.set_xlabel("City Population", fontsize=12, weight='bold')
    ax.set_ylabel("Transaction Amount", fontsize=12, weight='bold')
    ax.set_title("Transaction Amount vs City Population (Fraud Highlighted)", fontsize=14, weight='bold', pad=15)
    ax.set_ylim(0, 2000)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    
    # Model Training Section - Auto-runs when page loads
    st.markdown("## 🤖 Random Forest Model Training & Results")
    
    with st.spinner('🔄 Training Random Forest model... Please wait...'):
        # Create a hash of the dataframe for caching
        df_hash = hash(pd.util.hash_pandas_object(df).sum())
        
        # Train model
        model, y_test, y_pred, y_pred_proba, feature_importance, X_test_scaled = train_random_forest(test_size, df_hash)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate ROC AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    
    st.success('✅ Random Forest model trained successfully!')
    
    # Performance Metrics
    st.markdown("### 📊 Model Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h2 style='color: white; margin: 0; font-size: 40px;'>🎯</h2>
            <h2 style='color: white; margin: 10px 0;'>{accuracy*100:.2f}%</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;'>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h2 style='color: white; margin: 0; font-size: 40px;'>🔍</h2>
            <h2 style='color: white; margin: 10px 0;'>{precision*100:.2f}%</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;'>Precision</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h2 style='color: white; margin: 0; font-size: 40px;'>📈</h2>
            <h2 style='color: white; margin: 10px 0;'>{recall*100:.2f}%</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;'>Recall</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 25px; border-radius: 15px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.3);'>
            <h2 style='color: white; margin: 0; font-size: 40px;'>⚡</h2>
            <h2 style='color: white; margin: 10px 0;'>{f1*100:.2f}%</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 16px;'>F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Confusion Matrix and ROC Curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔥 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, 
                   xticklabels=["Non-Fraud", "Fraud"], 
                   yticklabels=["Non-Fraud", "Fraud"],
                   cbar_kws={'label': 'Count'},
                   linewidths=2, linecolor='white',
                   annot_kws={"size": 20, "weight": "bold"})
        ax.set_xlabel('Predicted Label', fontsize=12, weight='bold')
        ax.set_ylabel('True Label', fontsize=12, weight='bold')
        ax.set_title('Random Forest - Confusion Matrix', fontsize=14, weight='bold', pad=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### 📈 ROC Curve")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='#667eea', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#ef4444', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, weight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, weight='bold', pad=15)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    
    # Feature Importance and Classification Report
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌟 Top 10 Feature Importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        colors_bar = sns.color_palette("viridis", len(feature_importance))
        bars = ax.barh(feature_importance['feature'], feature_importance['importance'], color=colors_bar, edgecolor='black', linewidth=1)
        ax.set_xlabel('Importance Score', fontsize=12, weight='bold')
        ax.set_ylabel('Features', fontsize=12, weight='bold')
        ax.set_title('Most Important Features for Fraud Detection', fontsize=14, weight='bold', pad=15)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("### 📋 Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True, target_names=['Non-Fraud', 'Fraud'])
        report_df = pd.DataFrame(report).transpose()
        styled_df = report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']).format(precision=3)
        st.dataframe(styled_df, use_container_width=True, height=300)
    
    # Detailed Metrics Breakdown
    st.markdown("---")
    st.markdown("### 📊 Detailed Metrics Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
            <h4 style='color: #1e3a8a; margin-top: 0;'>True Positives (TP)</h4>
            <h2 style='color: #43e97b; margin: 10px 0;'>{tp:,}</h2>
            <p style='color: #475569; margin: 0;'>Correctly predicted fraud cases</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
            <h4 style='color: #1e3a8a; margin-top: 0;'>False Positives (FP)</h4>
            <h2 style='color: #f5576c; margin: 10px 0;'>{fp:,}</h2>
            <p style='color: #475569; margin: 0;'>Non-fraud predicted as fraud</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
            <h4 style='color: #1e3a8a; margin-top: 0;'>False Negatives (FN)</h4>
            <h2 style='color: #f093fb; margin: 10px 0;'>{fn:,}</h2>
            <p style='color: #475569; margin: 0;'>Fraud cases missed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);'>
            <h4 style='color: #1e3a8a; margin-top: 0;'>ROC AUC Score</h4>
            <h2 style='color: #4facfe; margin: 10px 0;'>{roc_auc:.3f}</h2>
            <p style='color: #475569; margin: 0;'>Model discrimination ability</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #475569; padding: 20px;'>
    <h3>🛡️ Fraud Detection Analytics Dashboard</h3>
    <p style='font-size: 14px;'>Powered by Random Forest Algorithm | Built with Streamlit & Scikit-learn</p>
    <p style='font-size: 12px; opacity: 0.7;'>© 2024 Advanced Fraud Detection System</p>
</div>
""", unsafe_allow_html=True)