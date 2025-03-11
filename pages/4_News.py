import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import re
from urllib.parse import urlparse

from utils.api import get_crypto_news, get_top_cryptocurrencies
from utils.styles import apply_styles
from utils.constants import NEWS_CATEGORIES

# Set page config
st.set_page_config(
    page_title="Crypto News | CryptoAnalytics Dashboard",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styles
apply_styles()

# Helper function to extract domain from URL
def extract_domain(url):
    try:
        parsed_uri = urlparse(url)
        domain = parsed_uri.netloc
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "Unknown Source"

# Helper function to format date
def format_date(date_str):
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        # If it's today, just show the time
        if date_obj.date() == datetime.now().date():
            return f"Today {date_obj.strftime('%H:%M')}"
        # If it's yesterday, say "Yesterday"
        elif date_obj.date() == (datetime.now() - timedelta(days=1)).date():
            return f"Yesterday {date_obj.strftime('%H:%M')}"
        # Otherwise show the date
        else:
            return date_obj.strftime("%b %d, %Y")
    except:
        return date_str

def main():
    # Header
    st.title("📰 Crypto News")
    st.markdown(
        """
        Latest news and updates from the cryptocurrency world to stay informed 
        about market trends and developments.
        """
    )
    st.markdown("---")
    
    # Sidebar filters
    st.sidebar.header("News Filters")
    
    # Category selection
    selected_category = st.sidebar.selectbox(
        "News Category",
        options=NEWS_CATEGORIES,
        index=0
    )
    
    # Number of news items
    num_news = st.sidebar.slider(
        "Number of News Items",
        min_value=5,
        max_value=50,
        value=20,
        step=5
    )
    
    # Display format selection
    display_format = st.sidebar.radio(
        "Display Format",
        options=["Card View", "Compact List", "Table View"],
        index=0
    )
    
    # Check if the Crypto News API key is set
    api_key_missing = False
    
    # Main content
    with st.spinner("Loading crypto news..."):
        # Convert "All News" to an empty string for the API
        category_param = "" if selected_category == "All News" else selected_category
        
        # Fetch news
        news_data = get_crypto_news(category=category_param, items=num_news)
        
        if isinstance(news_data, dict) and "error" in news_data:
            error_message = news_data.get("error", "Unknown error")
            if "API key not found" in error_message:
                api_key_missing = True
                st.error("Crypto News API key not found. This feature requires an API key to function.")
                st.info("Please set the CRYPTO_NEWS_API_KEY environment variable to enable this feature.")
            else:
                st.error(f"Error fetching news: {error_message}")
        
        # If API key is missing, show a demo disclaimer
        if api_key_missing:
            st.warning("The following news items are provided as a demonstration only. For real-time news, please set up the API key.")
            # Create sample news data
            news_data = {
                "articles": [
                    {
                        "title": f"Sample Crypto News Article {i}",
                        "description": "This is a sample news article description. In a real implementation, this would be fetched from a crypto news API.",
                        "url": "https://example.com/crypto-news",
                        "source": "Sample News Source",
                        "date": (datetime.now() - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
                        "sentiment": "Neutral",
                        "tickers": ["BTC", "ETH"]
                    }
                    for i in range(1, num_news + 1)
                ]
            }
    
    # Display news based on the selected format
    if news_data and "articles" in news_data:
        articles = news_data["articles"]
        
        # Add sentiments if they don't exist (for sample data)
        for article in articles:
            if "sentiment" not in article:
                article["sentiment"] = "Neutral"
        
        if display_format == "Card View":
            # Create a grid layout for cards
            st.subheader(f"{selected_category} News")
            
            # Process articles in batches of 3 for the grid
            for i in range(0, len(articles), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i + j < len(articles):
                        article = articles[i + j]
                        
                        # Extract domain from URL
                        source = article.get("source", extract_domain(article.get("url", "")))
                        
                        # Format date
                        date = format_date(article.get("date", ""))
                        
                        # Get sentiment color
                        sentiment = article.get("sentiment", "Neutral")
                        sentiment_color = "#00C853" if sentiment == "Positive" else "#FF3D00" if sentiment == "Negative" else "#FFB300"
                        
                        # Create card HTML
                        card_html = f"""
                        <div style="border: 1px solid #293B5F; border-radius: 8px; padding: 15px; margin-bottom: 15px; background-color: #172A46; height: 230px; overflow: hidden; position: relative;" class="hover-card">
                            <div style="margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #ADB5BD; font-size: 12px;">{source}</span>
                                <span style="color: #ADB5BD; font-size: 12px;">{date}</span>
                            </div>
                            <h3 style="margin-top: 0; margin-bottom: 10px; font-size: 16px; line-height: 1.3;">{article.get("title", "")}</h3>
                            <p style="color: #ADB5BD; font-size: 14px; margin-bottom: 15px; line-height: 1.4; height: 80px; overflow: hidden; text-overflow: ellipsis;">{article.get("description", "")}</p>
                            <div style="position: absolute; bottom: 15px; left: 15px; right: 15px; display: flex; justify-content: space-between; align-items: center;">
                                <span style="background-color: {sentiment_color}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 12px;">{sentiment}</span>
                                <a href="{article.get("url", "#")}" target="_blank" style="color: #0077B6; text-decoration: none; font-size: 14px;">Read More →</a>
                            </div>
                        </div>
                        """
                        
                        with cols[j]:
                            st.markdown(card_html, unsafe_allow_html=True)
        
        elif display_format == "Compact List":
            st.subheader(f"{selected_category} News")
            
            for article in articles:
                # Extract domain from URL
                source = article.get("source", extract_domain(article.get("url", "")))
                
                # Format date
                date = format_date(article.get("date", ""))
                
                # Get sentiment color
                sentiment = article.get("sentiment", "Neutral")
                sentiment_color = "#00C853" if sentiment == "Positive" else "#FF3D00" if sentiment == "Negative" else "#FFB300"
                
                # Create compact list item HTML
                list_item_html = f"""
                <div style="border-bottom: 1px solid #293B5F; padding: 12px 0; display: flex; justify-content: space-between; align-items: flex-start;">
                    <div style="flex: 1;">
                        <h3 style="margin: 0; font-size: 16px; line-height: 1.3; margin-bottom: 5px;">
                            <a href="{article.get("url", "#")}" target="_blank" style="color: #F8F9FA; text-decoration: none; hover: #0077B6;">{article.get("title", "")}</a>
                        </h3>
                        <div style="display: flex; align-items: center; margin-top: 5px;">
                            <span style="color: #ADB5BD; font-size: 12px; margin-right: 10px;">{source}</span>
                            <span style="color: #ADB5BD; font-size: 12px; margin-right: 10px;">•</span>
                            <span style="color: #ADB5BD; font-size: 12px; margin-right: 10px;">{date}</span>
                            <span style="background-color: {sentiment_color}; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px;">{sentiment}</span>
                        </div>
                    </div>
                </div>
                """
                
                st.markdown(list_item_html, unsafe_allow_html=True)
        
        elif display_format == "Table View":
            st.subheader(f"{selected_category} News")
            
            # Create a dataframe for the table view
            news_df = pd.DataFrame([
                {
                    "Title": article.get("title", ""),
                    "Source": article.get("source", extract_domain(article.get("url", ""))),
                    "Date": format_date(article.get("date", "")),
                    "Sentiment": article.get("sentiment", "Neutral"),
                    "URL": article.get("url", "#")
                }
                for article in articles
            ])
            
            # Add clickable links
            def make_clickable(url, text):
                return f'<a href="{url}" target="_blank">{text}</a>'
            
            news_df["Title"] = news_df.apply(lambda row: make_clickable(row["URL"], row["Title"]), axis=1)
            
            # Drop the URL column as it's now embedded in the title
            display_df = news_df.drop(columns=["URL"])
            
            # Display the table with clickable links
            st.markdown(
                display_df.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
        
        # Display news source attribution
        st.markdown("""
        <div style="margin-top: 30px; text-align: center; color: #ADB5BD; font-size: 12px;">
            News data provided by Crypto News API. Sentiment analysis is performed automatically and may not reflect the actual tone of the articles.
        </div>
        """, unsafe_allow_html=True)
        
        # News analysis section
        if len(articles) > 5 and not api_key_missing:
            st.markdown("---")
            st.subheader("News Analysis")
            
            # Create columns for analysis widgets
            col1, col2 = st.columns(2)
            
            with col1:
                # Count sentiment distribution
                sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
                
                for article in articles:
                    sentiment = article.get("sentiment", "Neutral")
                    sentiment_counts[sentiment] += 1
                
                # Create sentiment distribution chart
                sentiment_df = pd.DataFrame({
                    "Sentiment": list(sentiment_counts.keys()),
                    "Count": list(sentiment_counts.values())
                })
                
                # Define sentiment colors
                sentiment_colors = {"Positive": "#00C853", "Neutral": "#FFB300", "Negative": "#FF3D00"}
                
                fig = px.bar(
                    sentiment_df,
                    x="Sentiment",
                    y="Count",
                    title="News Sentiment Distribution",
                    color="Sentiment",
                    color_discrete_map=sentiment_colors
                )
                
                fig.update_layout(
                    height=300,
                    template="plotly_dark",
                    paper_bgcolor="#0A192F",
                    plot_bgcolor="#172A46",
                    font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA")
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Extract all tickers mentioned
                all_tickers = []
                for article in articles:
                    tickers = article.get("tickers", [])
                    if isinstance(tickers, list):
                        all_tickers.extend(tickers)
                
                # Count ticker mentions
                ticker_counts = {}
                for ticker in all_tickers:
                    if ticker in ticker_counts:
                        ticker_counts[ticker] += 1
                    else:
                        ticker_counts[ticker] = 1
                
                # Sort by count
                sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
                
                # Take the top 10
                top_tickers = sorted_tickers[:10]
                
                if top_tickers:
                    # Create ticker mentions chart
                    ticker_df = pd.DataFrame({
                        "Ticker": [t[0] for t in top_tickers],
                        "Mentions": [t[1] for t in top_tickers]
                    })
                    
                    fig = px.bar(
                        ticker_df,
                        x="Ticker",
                        y="Mentions",
                        title="Most Mentioned Cryptocurrencies",
                        color="Mentions",
                        color_continuous_scale="Viridis"
                    )
                    
                    fig.update_layout(
                        height=300,
                        template="plotly_dark",
                        paper_bgcolor="#0A192F",
                        plot_bgcolor="#172A46",
                        font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                        coloraxis_showscale=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No ticker data available for analysis.")
            
            # Additional analysis - news sources
            sources = [article.get("source", extract_domain(article.get("url", ""))) for article in articles]
            
            # Count sources
            source_counts = {}
            for source in sources:
                if source in source_counts:
                    source_counts[source] += 1
                else:
                    source_counts[source] = 1
            
            # Sort by count
            sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Take the top 5
            top_sources = sorted_sources[:5]
            
            if top_sources:
                # Create source distribution chart
                source_df = pd.DataFrame({
                    "Source": [s[0] for s in top_sources],
                    "Articles": [s[1] for s in top_sources]
                })
                
                fig = px.pie(
                    source_df,
                    values="Articles",
                    names="Source",
                    title="News Sources Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                
                fig.update_layout(
                    height=350,
                    template="plotly_dark",
                    paper_bgcolor="#0A192F",
                    font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                )
                
                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    else:
        if not api_key_missing:
            st.warning("No news articles available. Try selecting a different category or check your connection.")

if __name__ == "__main__":
    main()
