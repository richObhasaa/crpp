import streamlit as st
from utils.styles import apply_styles
from utils.constants import APP_TITLE, APP_DESCRIPTION

def main():
    # Set page configuration
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Apply custom styles
    apply_styles()
    
    # Sidebar
    st.sidebar.image("assets/logo.svg", width=80)
    st.sidebar.title("CryptoAnalytics Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Navigate through the pages using the sidebar menu to access different "
        "features of the platform."
    )
    
    # Main content
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    
    # Display main dashboard overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Market Overview")
        st.markdown(
            "Comprehensive view of the cryptocurrency market with real-time data, "
            "charts, and key metrics across multiple timeframes."
        )
        st.button("Go to Market Overview", on_click=lambda: st.switch_page("pages/1_Market_Overview.py"))
    
    with col2:
        st.subheader("🔍 Token Comparison")
        st.markdown(
            "Compare cryptocurrencies side by side with detailed metrics and "
            "visualizations to make informed investment decisions."
        )
        st.button("Go to Token Comparison", on_click=lambda: st.switch_page("pages/2_Token_Comparison.py"))
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("📊 Technical Analysis")
        st.markdown(
            "Advanced technical analysis tools for cryptocurrency price movements "
            "with multiple timeframes and performance calculator."
        )
        st.button("Go to Technical Analysis", on_click=lambda: st.switch_page("pages/3_Technical_Analysis.py"))
    
    with col4:
        st.subheader("📰 Crypto News")
        st.markdown(
            "Latest news and updates from the cryptocurrency world to stay "
            "informed about market trends and developments."
        )
        st.button("Go to News", on_click=lambda: st.switch_page("pages/4_News.py"))
    
    col5, _ = st.columns(2)
    
    with col5:
        st.subheader("📄 Whitepaper Analysis")
        st.markdown(
            "AI-powered analysis of cryptocurrency whitepapers to understand "
            "the technology and potential of various projects."
        )
        st.button("Go to Whitepaper Analysis", on_click=lambda: st.switch_page("pages/5_Whitepaper_Analysis.py"))
    
    # Display key market stats in expandable section
    with st.expander("📌 Current Market Highlights"):
        try:
            from utils.api import get_global_market_data
            market_data = get_global_market_data()
            
            if market_data:
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.metric(
                        "Total Market Cap (USD)", 
                        f"${market_data['total_market_cap']['usd']:,.0f}", 
                        f"{market_data['market_cap_change_percentage_24h_usd']:.2f}%"
                    )
                
                with metrics_col2:
                    st.metric(
                        "24h Trading Volume", 
                        f"${market_data['total_volume']['usd']:,.0f}"
                    )
                
                with metrics_col3:
                    st.metric(
                        "BTC Dominance", 
                        f"{market_data['market_cap_percentage']['btc']:.2f}%"
                    )
            else:
                st.warning("Unable to fetch current market data. Please try again later.")
        except Exception as e:
            st.error(f"Error loading market highlights: {str(e)}")
            st.warning("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()
