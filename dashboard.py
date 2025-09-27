import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import re

# --- CSS and Utility Functions ---

def load_css():
    """Load professional CSS styling."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* This targets the main content area */
    .stApp {
        background: transparent;
    }

    /* Professional header */
    .professional-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        text-align: center;
    }

    .professional-header h1 {
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .professional-header .subtitle {
        color: #e8f4fd;
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    /* Professional metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px rgba(31, 38, 135, 0.5);
    }

    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #2d3748;
        margin: 0;
    }

    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }

    /* Alert cards (used for insights here) */
    .alert-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        border-left: 4px solid;
        transition: transform 0.3s ease;
    }

    .alert-card:hover {
        transform: translateX(5px);
    }

    .alert-success { border-left-color: #48bb78; }
    .alert-info { border-left-color: #4299e1; }
    .alert-warning { border-left-color: #ed8936; }
    .alert-critical { border-left-color: #e53e3e; }
    
    /* Highlight */
    .profit-highlight {
        background: linear-gradient(135deg, #48bb78, #38a169);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(72, 187, 120, 0.3);
        font-family: 'Inter', sans-serif;
    }
    
    </style>
    """, unsafe_allow_html=True)

def create_professional_header(title="📊 ContractIQ Pro", subtitle="AI-Powered Distributor Contract Intelligence"):
    """Create the professional header."""
    st.markdown(f"""
    <div class="professional-header">
        <h1>{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title, value, change, icon="📊", card_class="metric-card"):
    """Create a professional metric card."""
    return f"""
    <div class="{card_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{icon} {title}</div>
        <div style="color: #48bb78; font-size: 0.9rem; margin-top: 0.5rem; font-weight: 600;">{change}</div>
    </div>
    """

def create_info_card(data):
    """Create an AI insight card."""
    return f"""
    <div class="alert-card alert-{data['type']}">
        <div style="display: flex; align-items: flex-start; gap: 1rem;">
            <div style="font-size: 1.5rem;">{data['icon']}</div>
            <div>
                <h4 style="margin: 0; color: #2d3748; font-family: 'Inter', sans-serif;">{data['title']}</h4>
                <p style="margin: 0.5rem 0 0 0; color: #4a5568; line-height: 1.6;">{data['message']}</p>
            </div>
        </div>
    </div>
    """

# --- Dashboard Content Functions ---

def show_executive_dashboard(df_contracts):
    """Executive dashboard with key metrics (adapted to RAG data)"""
    st.markdown("## 📊 Executive Dashboard")
    st.markdown("*Real-time insights into your processed contract portfolio*")

    if df_contracts.empty:
        st.info("👆 Process documents to see your analytics dashboard.")
        return

    # Mock data generation based on RAG structure (for demo purposes)
    df_contracts['commission_rate'] = df_contracts['avg_commission_rate']
    df_contracts['iphone_model'] = df_contracts['iphone_model'].fillna('Generic Model')
    df_contracts['contract_value'] = df_contracts['annual_revenue'] / (df_contracts['commission_rate']/100)
    df_contracts['annual_volume'] = df_contracts['annual_volume'].fillna(1000)

    # Calculate key metrics
    total_contracts = len(df_contracts)
    avg_commission = df_contracts['commission_rate'].mean()
    total_revenue = df_contracts['annual_revenue'].sum()
    
    # Mock Profit Opportunities Calculation (Simplified)
    profit_opportunities = df_contracts[df_contracts['commission_rate'] < 12].copy()
    profit_opportunities['impact'] = profit_opportunities['contract_value'] * ((12 - profit_opportunities['commission_rate'])/100)
    total_opportunity = profit_opportunities['impact'].sum()

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(create_metric_card(
            "Active Contracts", 
            str(total_contracts), 
            f"From {len(df_contracts['iphone_model'].unique())} unique models", 
            "📄"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_metric_card(
            "Avg Commission", 
            f"{avg_commission:.1f}%", 
            f"+{avg_commission-8.5:.1f}% vs baseline", 
            "💰"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(create_metric_card(
            "Annual Revenue", 
            f"${total_revenue/1000000:.1f}M", 
            "+15.2% projected growth", 
            "📈"
        ), unsafe_allow_html=True)

    with col4:
        st.markdown(create_metric_card(
            "Profit Opportunities", 
            str(len(profit_opportunities)), 
            f"${total_opportunity/1000:.0f}K potential", 
            "💡"
        ), unsafe_allow_html=True)

    # Charts section
    col1, col2 = st.columns(2)

    with col1:
        show_commission_chart(df_contracts)

    with col2:
        show_revenue_breakdown(df_contracts) # Using revenue breakdown instead of territory

    # AI Insights
    st.markdown("### 🧠 AI-Generated Executive Insights")
    show_executive_insights(df_contracts)

def show_commission_chart(df):
    """Show commission rate comparison chart"""
    fig = px.bar(df, x='Contract Name', y='commission_rate',
                 color='iphone_model', 
                 title='📈 Commission Rate Comparison',
                 hover_data=['Contract Name'])

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_revenue_breakdown(df):
    """Show revenue breakdown by contract"""
    fig = px.pie(df, values='annual_revenue', names='Contract Name',
                title='💰 Annual Revenue Distribution')

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)


def show_executive_insights(df):
    """Show AI-generated executive insights"""
    insights = []
    avg_commission = df['commission_rate'].mean()
    
    # Commission insights
    if avg_commission >= 11:
        insights.append({
            'type': 'success',
            'icon': '🏆',
            'title': 'Outstanding Commission Performance',
            'message': f'Your portfolio average of {avg_commission:.1f}% significantly exceeds industry benchmarks of 8-10%. This strong negotiating position should be leveraged in future contract discussions.'
        })
    else:
        insights.append({
            'type': 'warning',
            'icon': '⚠️',
            'title': 'Commission Optimization Required',
            'message': f'Portfolio average of {avg_commission:.1f}% is below market standards. Priority focus should be on renegotiating terms to achieve 10-12% benchmark rates.'
        })

    # Territory insights (MOCK data)
    territory_count = 3 # Hardcode for demo simplicity
    insights.append({
        'type': 'info',
        'icon': '🗺️',
        'title': 'Moderate Geographic Coverage',
        'message': f'Current {territory_count}-territory coverage offers a good foundation for expansion. Consider adjacent markets for 25-40% revenue growth potential.'
    })


    # Display insights
    for insight in insights:
        st.markdown(create_info_card(insight), unsafe_allow_html=True)

def show_profit_optimizer(df_contracts):
    """Profit optimization dashboard (Placeholder)"""
    st.markdown("## 💰 Profit Optimizer")
    st.markdown("*AI-powered revenue optimization for maximum profitability*")
    
    if df_contracts.empty:
        st.info("Process documents to analyze profit opportunities.")
        return
        
    st.warning("This section contains mock data as detailed RAG extraction is not available in the main file.")
    
    # Mock profit opportunity calculation (reusing logic from Executive Dashboard)
    df_contracts['commission_rate'] = df_contracts['avg_commission_rate']
    df_contracts['contract_value'] = df_contracts['annual_revenue'] / (df_contracts['commission_rate']/100)
    
    profit_opportunities = df_contracts[df_contracts['commission_rate'] < 12].copy()
    profit_opportunities['impact'] = profit_opportunities['contract_value'] * ((12 - profit_opportunities['commission_rate'])/100)
    
    total_opportunity = profit_opportunities['impact'].sum()
    
    # Profit highlight
    st.markdown(f"""
    <div class="profit-highlight">
        <h3>💎 Annual Profit Optimization Potential</h3>
        <div class="amount">${total_opportunity:,.0f}</div>
        <p>Based on analysis of {len(df_contracts)} contracts with {len(profit_opportunities)} identified optimization opportunities</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🎯 Optimization Recommendations")
    
    if not profit_opportunities.empty:
        for i, row in profit_opportunities.head(5).iterrows():
            st.markdown(f"**Contract:** {row['Contract Name']}")
            st.markdown(f"**Recommendation:** Increase commission from {row['commission_rate']:.1f}% to 12.0%.")
            st.markdown(f"**Impact:** +${row['impact']:,.0f} annually.")
            st.markdown("---")
    else:
        st.success("All processed contracts meet or exceed the 12% commission benchmark!")

# --------------------------------------------------------------------------------

def prepare_dashboard_data(available_docs, get_summary_func):
    """
    Prepares a clean DataFrame from RAG summary data for the dashboard.
    This function mocks non-critical data points for the visual appeal.
    """
    data = []
    
    # Mock data helper
    mock_volume = {'54': 15000, 'DISTRIBUTOR A': 20000, 'default': 10000}
    mock_revenue = {'54': 150000, 'DISTRIBUTOR A': 300000, 'default': 100000}
    mock_commission = {'54': 11.0, 'DISTRIBUTOR A': 12.5, 'default': 9.5}
    mock_model = {'54': 'iPhone 16', 'DISTRIBUTOR A': 'iPhone 17', 'default': 'Generic Model'}

    for doc_id in available_docs:
        # 1. Get Summary Content
        summary = get_summary_func(doc_id)
        
        # 2. Extract Commission Rate
        # We need a robust way to extract the commission rate for the dashboard metrics
        comm_match = re.search(r"## Margins\n.*?(\d+\.?\d*)%", summary)
        
        # Fallback to mock rate if RAG extraction fails
        commission_rate = float(comm_match.group(1)) if comm_match else mock_commission.get(doc_id, mock_commission['default'])
        
        # MOCK REVENUE/VALUE
        annual_revenue = mock_revenue.get(doc_id, mock_revenue['default'])
        annual_volume = mock_volume.get(doc_id, mock_volume['default'])
        
        # Calculate derived value
        contract_value = annual_revenue / (commission_rate/100)
        
        data.append({
            'Contract Name': doc_id,
            'avg_commission_rate': commission_rate,
            'annual_revenue': annual_revenue,
            'annual_volume': annual_volume,
            'contract_value': contract_value,
            'iphone_model': mock_model.get(doc_id, mock_model['default'])
        })

    return pd.DataFrame(data)


# --------------------------------------------------------------------------------
# Placeholder functions for original RAG content (Defined in your main app)
# --------------------------------------------------------------------------------

# Note: The functions below are placeholders. In the main app, we will import 
# the actual RAG functions and call them.

# def display_alert_page(available_docs): pass
# def get_summary(doc_id): pass
# def chat_logic(): pass
# def compare_logic(): pass