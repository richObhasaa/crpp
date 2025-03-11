import streamlit as st

def apply_styles():
    """Apply custom CSS styles to the Streamlit app"""
    
    # Custom CSS for the entire app
    st.markdown("""
    <style>
        /* Global styles */
        .stApp {
            background-color: #0A192F;
            color: #F8F9FA;
        }
        
        /* Headers styling */
        h1, h2, h3, h4 {
            color: #F8F9FA !important;
            font-weight: 600 !important;
        }
        
        /* Custom styling for metrics */
        .metric-container {
            background-color: #172A46;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
        }
        
        .metric-label {
            font-size: 14px;
            color: #ADB5BD;
        }
        
        /* Positive and negative values styling */
        .positive-value {
            color: #00C853 !important;
        }
        
        .negative-value {
            color: #FF3D00 !important;
        }
        
        /* Dashboard card styling */
        .dashboard-card {
            background-color: #172A46;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 4px solid #0077B6;
        }
        
        /* Table styling */
        .dataframe {
            font-size: 14px !important;
        }
        
        /* Custom button styling */
        .stButton>button {
            background-color: #0077B6 !important;
            color: white !important;
            border-radius: 5px !important;
            border: none !important;
            padding: 8px 16px !important;
            transition: all 0.3s ease-in-out !important;
        }
        
        .stButton>button:hover {
            background-color: #005F8D !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
        }
        
        /* Custom selectbox styling */
        .stSelectbox label, .stMultiselect label {
            color: #ADB5BD !important;
        }
        
        /* Custom slider styling */
        .stSlider label {
            color: #ADB5BD !important;
        }
        
        /* Widget labels */
        label {
            color: #ADB5BD !important;
            font-weight: 500 !important;
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #172A46 !important;
        }
        
        /* Custom expander styling */
        .streamlit-expanderHeader {
            background-color: #172A46 !important;
            color: #F8F9FA !important;
            border-radius: 5px !important;
        }
        
        /* Link styling */
        a {
            color: #0077B6 !important;
            text-decoration: none !important;
        }
        
        a:hover {
            color: #005F8D !important;
            text-decoration: underline !important;
        }
        
        /* Card with hover effect */
        .hover-card {
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .hover-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
        }
        
        /* Custom tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 40px;
            white-space: pre-wrap;
            background-color: #172A46;
            border-radius: 4px 4px 0 0;
            color: #ADB5BD;
            padding: 8px 16px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0077B6 !important;
            color: white !important;
        }
    </style>
    """, unsafe_allow_html=True)

def apply_metric_container(label, value, delta=None, delta_color="normal"):
    """Create a custom styled metric container with label, value and optional delta"""
    
    delta_html = ""
    if delta is not None:
        delta_class = "positive-value" if delta_color == "positive" else "negative-value" if delta_color == "negative" else ""
        delta_prefix = "+" if delta_color == "positive" and not str(delta).startswith("+") else ""
        delta_html = f'<div class="metric-delta {delta_class}">{delta_prefix}{delta}</div>'
    
    metric_html = f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    
    return st.markdown(metric_html, unsafe_allow_html=True)

def apply_dashboard_card(title, content, key=None):
    """Create a custom styled dashboard card with title and content"""
    
    card_html = f"""
    <div class="dashboard-card" id="{key if key else ''}">
        <h3>{title}</h3>
        <div>{content}</div>
    </div>
    """
    
    return st.markdown(card_html, unsafe_allow_html=True)
