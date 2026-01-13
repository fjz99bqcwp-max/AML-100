#!/usr/bin/env python3
"""
AML-100 Streamlit Live Dashboard
Real-time monitoring for autonomous HFT system
Displays wallet PnL, trade density, live alerts
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

# Wallet configuration
WALLET = "0x12045C1Cc410461B24e4293Dd05e2a6c47ebb584"

st.set_page_config(
    page_title="AML-100 Live Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
    .profit { color: #00FF00; }
    .loss { color: #FF0000; }
    .warning { color: #FFA500; }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🚀 AML-100 Monitor")
    st.metric("Wallet", WALLET[:10] + "...")
    
    refresh_rate = st.selectbox("Refresh Rate", [5, 10, 30, 60], index=1)
    st.caption(f"Auto-refresh every {refresh_rate}s")
    
    show_alerts = st.checkbox("Show Alerts", value=True)
    show_charts = st.checkbox("Show Charts", value=True)

# Main content
st.title("🎯 AML-100 Live Dashboard")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S CET')}")

# Fetch wallet data
@st.cache_data(ttl=refresh_rate)
def fetch_wallet_data():
    try:
        from hyperliquid.info import Info
        from hyperliquid.utils import constants
        
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        user_state = info.user_state(WALLET)
        fills = info.user_fills(WALLET)
        
        return user_state, fills
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return None, None

user_state, fills = fetch_wallet_data()

if user_state and fills:
    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    account_value = float(user_state['marginSummary']['accountValue'])
    total_margin_used = float(user_state['marginSummary']['totalMarginUsed'])
    total_pnl = sum([float(p.get('unrealizedPnl', 0)) for p in user_state.get('assetPositions', [])])
    pnl_pct = (total_pnl / account_value * 100) if account_value > 0 else 0
    
    with col1:
        st.metric(
            "💰 Account Value",
            f"${account_value:,.2f}",
            delta=f"{pnl_pct:+.2f}%"
        )
    
    with col2:
        st.metric(
            "📊 Total PnL",
            f"${total_pnl:,.2f}",
            delta=f"{pnl_pct:+.2f}%",
            delta_color="normal" if total_pnl >= 0 else "inverse"
        )
    
    with col3:
        positions = user_state.get('assetPositions', [])
        st.metric("📍 Open Positions", len(positions))
    
    with col4:
        leverage_used = (total_margin_used / account_value) if account_value > 0 else 0
        st.metric("⚡ Leverage", f"{leverage_used:.2f}x")
    
    # Alerts
    if show_alerts:
        st.subheader("🚨 Alerts")
        
        # Trade density check (24h)
        recent_fills = [f for f in fills if f['time'] > (time.time() - 86400) * 1000]
        if len(recent_fills) < 10:
            st.warning(f"⚠️ LOW TRADE DENSITY: {len(recent_fills)} trades in 24h (target: >50)")
        
        # High leverage warning
        if leverage_used > 10:
            st.warning(f"⚠️ HIGH LEVERAGE: {leverage_used:.1f}x (consider reducing)")
        
        # Funding check (if negative LONG position)
        for pos in positions:
            if float(pos['position']['szi']) > 0:  # LONG position
                # Would need to fetch funding rate here
                st.info(f"💸 LONG {pos['position']['coin']}: Check funding rate")
    
    # Charts
    if show_charts and len(recent_fills) > 0:
        st.subheader("📈 Performance Charts")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Trade density heatmap
            df_fills = pd.DataFrame(recent_fills)
            df_fills['datetime'] = pd.to_datetime(df_fills['time'], unit='ms')
            df_fills['hour'] = df_fills['datetime'].dt.hour
            
            hourly_counts = df_fills.groupby('hour').size().reindex(range(24), fill_value=0)
            
            fig_density = go.Figure(data=[
                go.Bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    marker_color='lightblue'
                )
            ])
            fig_density.update_layout(
                title="Trade Density (Last 24h)",
                xaxis_title="Hour (CET)",
                yaxis_title="Trades",
                height=300
            )
            st.plotly_chart(fig_density, use_container_width=True)
        
        with col2:
            # PnL distribution
            df_fills['pnl'] = df_fills.apply(
                lambda row: float(row.get('closedPnl', 0)) if row.get('closedPnl') else 0,
                axis=1
            )
            
            fig_pnl = go.Figure(data=[
                go.Histogram(
                    x=df_fills['pnl'],
                    nbinsx=30,
                    marker_color='green',
                    opacity=0.7
                )
            ])
            fig_pnl.update_layout(
                title="PnL Distribution (Last 24h)",
                xaxis_title="PnL ($)",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        # Recent fills table
        st.subheader("📜 Recent Fills")
        df_display = df_fills[['datetime', 'coin', 'side', 'px', 'sz', 'pnl']].head(20)
        df_display.columns = ['Time', 'Symbol', 'Side', 'Price', 'Size', 'PnL']
        st.dataframe(df_display, use_container_width=True)
    
    # Position details
    if positions:
        st.subheader("📍 Open Positions")
        
        pos_data = []
        for pos in positions:
            p = pos['position']
            pos_data.append({
                'Symbol': p['coin'],
                'Size': float(p['szi']),
                'Entry': float(p['entryPx']),
                'Current': float(pos.get('unrealizedPnl', 0)),
                'PnL': f"${float(pos.get('unrealizedPnl', 0)):,.2f}",
                'Leverage': float(p.get('leverage', {}).get('value', 1))
            })
        
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True)

else:
    st.error("❌ Failed to fetch wallet data. Check API connection.")

# Auto-refresh
if st.button("🔄 Refresh Now"):
    st.rerun()

# Auto-refresh every N seconds
time.sleep(refresh_rate)
st.rerun()
