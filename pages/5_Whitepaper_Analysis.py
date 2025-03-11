import streamlit as st
import pandas as pd
import json
import re
from io import StringIO

from utils.openai_helper import (
    analyze_whitepaper,
    extract_key_metrics_from_whitepaper,
    compare_whitepapers
)
from utils.api import get_top_cryptocurrencies
from utils.styles import apply_styles

# Set page config
st.set_page_config(
    page_title="Whitepaper Analysis | CryptoAnalytics Dashboard",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom styles
apply_styles()

def main():
    # Header
    st.title("📄 Whitepaper Analysis")
    st.markdown(
        """
        AI-powered analysis of cryptocurrency whitepapers to understand the technology 
        and potential of various projects.
        """
    )
    st.markdown("---")
    
    # Check if OpenAI API key is available
    import os
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not openai_api_key:
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable to enable this feature.")
        st.info(
            """
            This feature requires an OpenAI API key to function. Once you've set up your API key, 
            you'll be able to:
            
            - Analyze cryptocurrency whitepapers for technical feasibility
            - Extract key metrics and tokenomics information
            - Compare multiple whitepapers to assess relative strengths
            - Get investment outlook and risk assessments
            """
        )
        return
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs([
        "📊 Single Whitepaper Analysis", 
        "🔍 Extract Metrics", 
        "⚖️ Compare Whitepapers"
    ])
    
    with tab1:
        st.header("Analyze a Cryptocurrency Whitepaper")
        
        # Input method selection
        input_method = st.radio(
            "Select input method",
            options=["Upload PDF", "Paste Text"],
            horizontal=True
        )
        
        whitepaper_text = ""
        
        if input_method == "Upload PDF":
            st.warning("PDF parsing may not capture all formatting. For best results, copy and paste the text directly.")
            
            uploaded_file = st.file_uploader("Upload whitepaper PDF", type=["pdf", "txt"])
            
            if uploaded_file is not None:
                try:
                    # Check if it's a PDF
                    if uploaded_file.name.endswith('.pdf'):
                        try:
                            import pdfplumber
                            
                            with pdfplumber.open(uploaded_file) as pdf:
                                for page in pdf.pages:
                                    whitepaper_text += page.extract_text() + "\n\n"
                        except ImportError:
                            st.error("PDF processing library not available. Please install pdfplumber or use text paste option.")
                    else:
                        # Assume it's a text file
                        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                        whitepaper_text = stringio.read()
                    
                    # Show a preview of the text
                    with st.expander("Preview Extracted Text"):
                        st.text(whitepaper_text[:1000] + "..." if len(whitepaper_text) > 1000 else whitepaper_text)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        else:
            whitepaper_text = st.text_area(
                "Paste whitepaper text here",
                height=300,
                placeholder="Paste the content of the cryptocurrency whitepaper here for analysis..."
            )
        
        # Project name input
        project_name = st.text_input("Project Name", placeholder="Enter the name of the cryptocurrency project")
        
        # Analysis button
        if st.button("Analyze Whitepaper", key="analyze_single") and whitepaper_text:
            if len(whitepaper_text) < 100:
                st.error("The provided text is too short for a meaningful analysis. Please provide more content.")
            else:
                with st.spinner("Analyzing whitepaper... This may take a minute or two."):
                    # Call the OpenAI API for analysis
                    analysis_result = analyze_whitepaper(whitepaper_text)
                    
                    if "error" in analysis_result:
                        st.error(f"Error during analysis: {analysis_result['error']}")
                    else:
                        # Display the analysis results
                        st.success(f"Analysis complete for {project_name if project_name else 'the provided whitepaper'}")
                        
                        # Create columns for the analysis display
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Executive Summary
                            st.subheader("Executive Summary")
                            st.markdown(analysis_result.get("Executive Summary", "No summary available"))
                            
                            # Technology Analysis
                            with st.expander("Technology Analysis", expanded=True):
                                st.markdown(analysis_result.get("Technology Analysis", "No technology analysis available"))
                            
                            # Use Case & Value Proposition
                            with st.expander("Use Case & Value Proposition", expanded=True):
                                st.markdown(analysis_result.get("Use Case & Value Proposition", "No use case analysis available"))
                            
                            # Tokenomics Analysis
                            with st.expander("Tokenomics Analysis", expanded=True):
                                st.markdown(analysis_result.get("Tokenomics Analysis", "No tokenomics analysis available"))
                            
                            # Team & Development Assessment
                            with st.expander("Team & Development Assessment"):
                                st.markdown(analysis_result.get("Team & Development Assessment", "No team assessment available"))
                            
                            # Risk Assessment
                            with st.expander("Risk Assessment", expanded=True):
                                st.markdown(analysis_result.get("Risk Assessment", "No risk assessment available"))
                            
                            # Market Potential
                            with st.expander("Market Potential"):
                                st.markdown(analysis_result.get("Market Potential", "No market potential analysis available"))
                        
                        with col2:
                            # Technical Feasibility Rating card
                            tech_rating = analysis_result.get("Technical Feasibility Rating", "N/A")
                            
                            # Determine color based on rating
                            if isinstance(tech_rating, (int, float)) or (isinstance(tech_rating, str) and tech_rating.isdigit()):
                                rating_value = float(tech_rating)
                                if rating_value >= 7:
                                    rating_color = "#00C853"  # Green for high ratings
                                elif rating_value >= 5:
                                    rating_color = "#FFB300"  # Amber for medium ratings
                                else:
                                    rating_color = "#FF3D00"  # Red for low ratings
                            else:
                                rating_color = "#0077B6"  # Default blue
                            
                            st.markdown(
                                f"""
                                <div style="background-color: #172A46; border-radius: 8px; padding: 15px; margin-bottom: 20px; border-left: 4px solid {rating_color};">
                                    <h3 style="margin-top: 0;">Technical Feasibility Rating</h3>
                                    <div style="font-size: 42px; font-weight: bold; text-align: center; color: {rating_color};">{tech_rating}/10</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Investment Outlook
                            st.markdown(
                                """
                                <div style="background-color: #172A46; border-radius: 8px; padding: 15px; margin-bottom: 20px;">
                                    <h3 style="margin-top: 0;">Investment Outlook</h3>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            investment_outlook = analysis_result.get("Investment Outlook", {})
                            if isinstance(investment_outlook, dict):
                                st.markdown("**Short-term:**")
                                st.markdown(investment_outlook.get("short", "No short-term outlook available"))
                                
                                st.markdown("**Medium-term:**")
                                st.markdown(investment_outlook.get("medium", "No medium-term outlook available"))
                                
                                st.markdown("**Long-term:**")
                                st.markdown(investment_outlook.get("long", "No long-term outlook available"))
                            else:
                                st.markdown(investment_outlook)
                            
                            st.markdown(
                                """
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Key Strengths & Weaknesses
                            st.markdown(
                                """
                                <div style="background-color: #172A46; border-radius: 8px; padding: 15px;">
                                    <h3 style="margin-top: 0;">Key Strengths & Weaknesses</h3>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            strengths_weaknesses = analysis_result.get("Key Strengths & Weaknesses", {})
                            
                            if isinstance(strengths_weaknesses, dict) and "strengths" in strengths_weaknesses and "weaknesses" in strengths_weaknesses:
                                # Display strengths
                                st.markdown("**Strengths:**")
                                strengths = strengths_weaknesses.get("strengths", [])
                                if isinstance(strengths, list):
                                    for strength in strengths:
                                        st.markdown(f"- {strength}")
                                else:
                                    st.markdown(strengths)
                                
                                # Display weaknesses
                                st.markdown("**Weaknesses:**")
                                weaknesses = strengths_weaknesses.get("weaknesses", [])
                                if isinstance(weaknesses, list):
                                    for weakness in weaknesses:
                                        st.markdown(f"- {weakness}")
                                else:
                                    st.markdown(weaknesses)
                            else:
                                st.markdown(strengths_weaknesses)
                            
                            st.markdown(
                                """
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Analysis metadata
                            st.caption(f"Analysis performed using {analysis_result.get('model_used', 'AI')} at {analysis_result.get('analysis_timestamp', 'N/A')}")
    
    with tab2:
        st.header("Extract Key Metrics from Whitepaper")
        
        # Input method for metrics extraction
        metrics_input_method = st.radio(
            "Select input method for metrics extraction",
            options=["Upload PDF", "Paste Text"],
            horizontal=True,
            key="metrics_input_method"
        )
        
        metrics_whitepaper_text = ""
        
        if metrics_input_method == "Upload PDF":
            st.warning("PDF parsing may not capture all formatting. For best results, copy and paste the text directly.")
            
            metrics_uploaded_file = st.file_uploader("Upload whitepaper PDF for metrics extraction", type=["pdf", "txt"], key="metrics_uploader")
            
            if metrics_uploaded_file is not None:
                try:
                    # Check if it's a PDF
                    if metrics_uploaded_file.name.endswith('.pdf'):
                        try:
                            import pdfplumber
                            
                            with pdfplumber.open(metrics_uploaded_file) as pdf:
                                for page in pdf.pages:
                                    metrics_whitepaper_text += page.extract_text() + "\n\n"
                        except ImportError:
                            st.error("PDF processing library not available. Please install pdfplumber or use text paste option.")
                    else:
                        # Assume it's a text file
                        stringio = StringIO(metrics_uploaded_file.getvalue().decode("utf-8"))
                        metrics_whitepaper_text = stringio.read()
                    
                    # Show a preview of the text
                    with st.expander("Preview Extracted Text"):
                        st.text(metrics_whitepaper_text[:1000] + "..." if len(metrics_whitepaper_text) > 1000 else metrics_whitepaper_text)
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        else:
            metrics_whitepaper_text = st.text_area(
                "Paste whitepaper text here for metrics extraction",
                height=300,
                placeholder="Paste the content of the cryptocurrency whitepaper here to extract key metrics...",
                key="metrics_text_area"
            )
        
        # Project name input for metrics
        metrics_project_name = st.text_input("Project Name", placeholder="Enter the name of the cryptocurrency project", key="metrics_project_name")
        
        # Extract metrics button
        if st.button("Extract Key Metrics", key="extract_metrics") and metrics_whitepaper_text:
            if len(metrics_whitepaper_text) < 100:
                st.error("The provided text is too short for meaningful metrics extraction. Please provide more content.")
            else:
                with st.spinner("Extracting metrics from whitepaper... This may take a moment."):
                    # Call the OpenAI API for metrics extraction
                    metrics_result = extract_key_metrics_from_whitepaper(metrics_whitepaper_text)
                    
                    if "error" in metrics_result:
                        st.error(f"Error during metrics extraction: {metrics_result['error']}")
                    else:
                        # Display the extracted metrics
                        st.success(f"Metrics extracted for {metrics_project_name if metrics_project_name else 'the provided whitepaper'}")
                        
                        # Create a table for the metrics
                        metrics_data = []
                        for key, value in metrics_result.items():
                            # Skip metadata keys
                            if key not in ["extraction_timestamp", "model_used"]:
                                metrics_data.append({"Metric": key, "Value": value})
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        # Display the metrics table
                        st.table(metrics_df)
                        
                        # Create a more visual tokenomics display if available
                        token_supply = metrics_result.get("Total Token Supply", "Not specified")
                        token_distribution = metrics_result.get("Token Distribution Percentages", "Not specified")
                        
                        if token_distribution != "Not specified" and isinstance(token_distribution, str):
                            # Try to parse the distribution text into a dictionary or list
                            try:
                                # Check if it looks like a JSON object
                                if "{" in token_distribution and "}" in token_distribution:
                                    # Try to extract a JSON object
                                    json_match = re.search(r'({[\s\S]*})', token_distribution)
                                    if json_match:
                                        distribution_data = json.loads(json_match.group(1))
                                        
                                        # Create a pie chart for token distribution
                                        st.subheader("Token Distribution Visualization")
                                        
                                        # Convert to dataframe for charting
                                        dist_data = []
                                        for category, percentage in distribution_data.items():
                                            # Convert percentage string to number if needed
                                            if isinstance(percentage, str):
                                                percentage = float(percentage.strip("%"))
                                            dist_data.append({"Category": category, "Percentage": percentage})
                                        
                                        dist_df = pd.DataFrame(dist_data)
                                        
                                        # Create pie chart using Plotly
                                        import plotly.express as px
                                        
                                        fig = px.pie(
                                            dist_df, 
                                            values='Percentage', 
                                            names='Category',
                                            title=f"Token Distribution for {metrics_project_name if metrics_project_name else 'Project'}",
                                            color_discrete_sequence=px.colors.sequential.Plasma
                                        )
                                        
                                        fig.update_layout(
                                            height=400,
                                            template="plotly_dark",
                                            paper_bgcolor="#0A192F",
                                            font=dict(family="Arial, sans-serif", size=12, color="#F8F9FA")
                                        )
                                        
                                        fig.update_traces(
                                            textposition='inside',
                                            textinfo='percent+label'
                                        )
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.info(f"Could not visualize token distribution: {str(e)}")
                        
                        # Analysis metadata
                        st.caption(f"Metrics extracted using {metrics_result.get('model_used', 'AI')} at {metrics_result.get('extraction_timestamp', 'N/A')}")
    
    with tab3:
        st.header("Compare Two Whitepapers")
        
        # Create two columns for whitepaper inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("First Whitepaper")
            
            # Input method for first whitepaper
            first_input_method = st.radio(
                "Select input method",
                options=["Upload PDF", "Paste Text"],
                horizontal=True,
                key="first_input_method"
            )
            
            first_whitepaper_text = ""
            
            if first_input_method == "Upload PDF":
                st.warning("PDF parsing may not capture all formatting. For best results, copy and paste the text directly.")
                
                first_uploaded_file = st.file_uploader("Upload first whitepaper PDF", type=["pdf", "txt"], key="first_uploader")
                
                if first_uploaded_file is not None:
                    try:
                        # Check if it's a PDF
                        if first_uploaded_file.name.endswith('.pdf'):
                            try:
                                import pdfplumber
                                
                                with pdfplumber.open(first_uploaded_file) as pdf:
                                    for page in pdf.pages:
                                        first_whitepaper_text += page.extract_text() + "\n\n"
                            except ImportError:
                                st.error("PDF processing library not available. Please install pdfplumber or use text paste option.")
                        else:
                            # Assume it's a text file
                            stringio = StringIO(first_uploaded_file.getvalue().decode("utf-8"))
                            first_whitepaper_text = stringio.read()
                        
                        # Show a preview of the text
                        with st.expander("Preview Extracted Text"):
                            st.text(first_whitepaper_text[:500] + "..." if len(first_whitepaper_text) > 500 else first_whitepaper_text)
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            else:
                first_whitepaper_text = st.text_area(
                    "Paste first whitepaper text here",
                    height=250,
                    placeholder="Paste the content of the first cryptocurrency whitepaper here for comparison...",
                    key="first_text_area"
                )
            
            # First project name input
            first_project_name = st.text_input("First Project Name", placeholder="Enter the name of the first cryptocurrency project", key="first_project_name")
        
        with col2:
            st.subheader("Second Whitepaper")
            
            # Input method for second whitepaper
            second_input_method = st.radio(
                "Select input method",
                options=["Upload PDF", "Paste Text"],
                horizontal=True,
                key="second_input_method"
            )
            
            second_whitepaper_text = ""
            
            if second_input_method == "Upload PDF":
                st.warning("PDF parsing may not capture all formatting. For best results, copy and paste the text directly.")
                
                second_uploaded_file = st.file_uploader("Upload second whitepaper PDF", type=["pdf", "txt"], key="second_uploader")
                
                if second_uploaded_file is not None:
                    try:
                        # Check if it's a PDF
                        if second_uploaded_file.name.endswith('.pdf'):
                            try:
                                import pdfplumber
                                
                                with pdfplumber.open(second_uploaded_file) as pdf:
                                    for page in pdf.pages:
                                        second_whitepaper_text += page.extract_text() + "\n\n"
                            except ImportError:
                                st.error("PDF processing library not available. Please install pdfplumber or use text paste option.")
                        else:
                            # Assume it's a text file
                            stringio = StringIO(second_uploaded_file.getvalue().decode("utf-8"))
                            second_whitepaper_text = stringio.read()
                        
                        # Show a preview of the text
                        with st.expander("Preview Extracted Text"):
                            st.text(second_whitepaper_text[:500] + "..." if len(second_whitepaper_text) > 500 else second_whitepaper_text)
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            else:
                second_whitepaper_text = st.text_area(
                    "Paste second whitepaper text here",
                    height=250,
                    placeholder="Paste the content of the second cryptocurrency whitepaper here for comparison...",
                    key="second_text_area"
                )
            
            # Second project name input
            second_project_name = st.text_input("Second Project Name", placeholder="Enter the name of the second cryptocurrency project", key="second_project_name")
        
        # Compare button
        if st.button("Compare Whitepapers", key="compare_whitepapers") and first_whitepaper_text and second_whitepaper_text:
            if len(first_whitepaper_text) < 100 or len(second_whitepaper_text) < 100:
                st.error("One or both of the provided texts are too short for a meaningful comparison. Please provide more content.")
            else:
                with st.spinner("Comparing whitepapers... This may take a few minutes."):
                    # Call the OpenAI API for comparison
                    comparison_result = compare_whitepapers(
                        first_whitepaper_text, 
                        second_whitepaper_text,
                        first_project_name if first_project_name else "Project 1",
                        second_project_name if second_project_name else "Project 2"
                    )
                    
                    if "error" in comparison_result:
                        st.error(f"Error during comparison: {comparison_result['error']}")
                    else:
                        # Display the comparison results
                        project1 = first_project_name if first_project_name else "Project 1"
                        project2 = second_project_name if second_project_name else "Project 2"
                        
                        st.success(f"Comparison complete between {project1} and {project2}")
                        
                        # Create headings for each comparison section
                        comparison_sections = [
                            "Technology Approach",
                            "Use Cases",
                            "Tokenomics",
                            "Consensus Mechanisms",
                            "Scalability Solutions",
                            "Governance Models",
                            "Security Approaches",
                            "Relative Strengths",
                            "Market Positioning",
                            "Overall Assessment"
                        ]
                        
                        # Display each comparison section
                        for section in comparison_sections:
                            if section in comparison_result:
                                with st.expander(section, expanded=section == "Overall Assessment"):
                                    st.markdown(comparison_result[section])
                        
                        # Create a visual comparison chart for the overall evaluation
                        if "Overall Assessment" in comparison_result:
                            overall = comparison_result["Overall Assessment"]
                            
                            # Try to determine which project is rated better
                            try:
                                # Look for patterns that might indicate a preference
                                project1_preferred = False
                                project2_preferred = False
                                
                                # Check if there are explicit mentions of one being better
                                project1_patterns = [
                                    f"{project1} presents a more robust",
                                    f"{project1} is more feasible",
                                    f"{project1} demonstrates greater potential",
                                    f"{project1} has a stronger",
                                    f"{project1} appears more promising"
                                ]
                                
                                project2_patterns = [
                                    f"{project2} presents a more robust",
                                    f"{project2} is more feasible",
                                    f"{project2} demonstrates greater potential",
                                    f"{project2} has a stronger",
                                    f"{project2} appears more promising"
                                ]
                                
                                for pattern in project1_patterns:
                                    if pattern.lower() in overall.lower():
                                        project1_preferred = True
                                        break
                                        
                                for pattern in project2_patterns:
                                    if pattern.lower() in overall.lower():
                                        project2_preferred = True
                                        break
                                
                                # Display a visual comparison based on the analysis
                                st.subheader("Visual Comparison")
                                
                                if project1_preferred and not project2_preferred:
                                    # Project 1 is preferred
                                    preference = 0.7  # Leaning towards project 1
                                elif project2_preferred and not project1_preferred:
                                    # Project 2 is preferred
                                    preference = 0.3  # Leaning towards project 2
                                else:
                                    # No clear preference or both preferred
                                    preference = 0.5  # Neutral
                                
                                # Create a comparison meter
                                st.markdown(
                                    f"""
                                    <div style="position: relative; height: 60px; background-color: #172A46; border-radius: 30px; margin: 20px 0; overflow: hidden;">
                                        <div style="position: absolute; left: 5%; top: 50%; transform: translateY(-50%); color: #F8F9FA; z-index: 2;">{project1}</div>
                                        <div style="position: absolute; right: 5%; top: 50%; transform: translateY(-50%); color: #F8F9FA; z-index: 2;">{project2}</div>
                                        <div style="position: absolute; top: 0; left: 0; height: 100%; width: {preference * 100}%; background-color: #0077B6; border-radius: 30px 0 0 30px;"></div>
                                        <div style="position: absolute; top: 0; right: 0; height: 100%; width: {(1 - preference) * 100}%; background-color: #FB8C00; border-radius: 0 30px 30px 0;"></div>
                                        <div style="position: absolute; left: {preference * 100}%; top: 0; height: 100%; width: 4px; background-color: white; transform: translateX(-2px);"></div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                
                                # Add a note about the visual comparison
                                if project1_preferred and not project2_preferred:
                                    st.caption(f"Based on the analysis, {project1} appears to present a more robust and feasible project.")
                                elif project2_preferred and not project1_preferred:
                                    st.caption(f"Based on the analysis, {project2} appears to present a more robust and feasible project.")
                                else:
                                    st.caption("Both projects have their strengths and weaknesses. The analysis doesn't indicate a clear preference.")
                                
                            except Exception as e:
                                # If any error occurs during the visual comparison creation, skip it
                                pass
                        
                        # Analysis metadata
                        st.caption(f"Comparison performed using {comparison_result.get('model_used', 'AI')} at {comparison_result.get('comparison_timestamp', 'N/A')}")

# Run the main function
if __name__ == "__main__":
    main()
