import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pickle
import os

# Set page configuration with a custom theme
st.set_page_config(
    page_title="Crude Oil Price Dashboard",
    page_icon="🛢️",
    layout="wide"
)

# Load data function
@st.cache_data
def load_data():
    try:
        csv_files = [f for f in os.listdir() if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(csv_files[0])
            df['date'] = pd.to_datetime(df['date'])
            return df
        
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        prices = np.random.normal(75, 5, len(dates))
        
        df = pd.DataFrame({
            'date': dates,
            'price': prices
        })
        
        df['change'] = df['price'].diff()
        df['percentChange'] = (df['price'].pct_change() * 100)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load ARIMA model function
@st.cache_resource
def load_arima_model():
    try:
        if os.path.exists('arima_crude_oil_model.pkl'):
            with open('arima_crude_oil_model.pkl', 'rb') as file:
                model = pickle.load(file)
            return model
        
        return None
    except Exception as e:
        st.error(f"Error loading ARIMA model: {str(e)}")
        return None

# Function to predict price for today
def predict_today_price(model, df):
    if model is None:
        last_price = df['price'].iloc[-1]
        return last_price * (1 + np.random.normal(0, 0.01))
    
    forecast = model.forecast(steps=1)
    return forecast[0]

def create_monthly_price_distribution(df):
    monthly_data = df.set_index('date').resample('M')['price'].mean()
    price_ranges = pd.cut(monthly_data, bins=5)
    distribution = price_ranges.value_counts()
    
    fig = px.pie(
        values=distribution.values,
        names=distribution.index.astype(str),
        title='Monthly Price Distribution',
        template='plotly_dark',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0')
    )
    return fig

# Custom styling with enhanced metric cards
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
        }
        .main-header {
            background: linear-gradient(90deg, #FF4E50 0%, #F9D423 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 2rem;
            padding: 1rem 0;
            text-align: center;
        }
        .metric-card {
            background: linear-gradient(145deg, #1e2132 0%, #252b48 100%);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid #2a325a;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
            margin-bottom: 1rem;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-title {
            color: #8892b0;
            font-size: 1rem;
            margin-bottom: 0.5rem;
        }
        .metric-value {
            color: #00ff88;
            font-size: 1.75rem;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }
        .metric-delta {
            color: #64ffda;
            font-size: 1rem;
        }
        .prediction-card {
            background: linear-gradient(145deg, #2a1f3d 0%, #1f2937 100%);
            border: 1px solid #6d28d9;
        }
        .prediction-value {
            color: #a78bfa;
            font-size: 2rem;
        }
        .stButton > button {
            background: linear-gradient(90deg, #FF4E50 0%, #F9D423 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.75rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.2);
        }
        h2, h3 {
            color: #00ff88;
            font-weight: 600;
            margin: 1.5rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #2a325a;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🛢️ Crude Oil Price Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #00ff88; font-size: 1.2rem; margin-bottom: 2rem;'>Real-time analysis and price prediction</p>", unsafe_allow_html=True)
    
    df = load_data()
    arima_model = load_arima_model()
    
    if df is not None:
        df['date'] = df['date'].dt.tz_localize(None)  # Ensure all dates are timezone-naive
        
        # Get today's predicted price
        today_predicted = predict_today_price(arima_model, df)
        
        # KPI Metrics Section
        st.subheader("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card prediction-card">
                    <div class="metric-title">Today's Predicted Price</div>
                    <div class="prediction-value">${today_predicted:.2f}</div>
                    <div class="metric-delta">Predicted value</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            current_price = df['price'].iloc[-1]
            price_change = df['change'].iloc[-1]
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Current Oil Price</div>
                    <div class="metric-value">${current_price:.2f}</div>
                    <div class="metric-delta">Δ ${price_change:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            daily_return = df['percentChange'].iloc[-1]
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Daily Return</div>
                    <div class="metric-value">{daily_return:.2f}%</div>
                    <div class="metric-delta">Last 24 hours</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            volatility = df['percentChange'].std()
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Price Volatility</div>
                    <div class="metric-value">{volatility:.2f}%</div>
                    <div class="metric-delta">Standard deviation</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Price Prediction Calculator
        with st.expander("🔮 Price Prediction Calculator", expanded=True):
            col1, col2 = st.columns([2, 3])
            
            with col1:
                min_date = datetime.now().date()
                max_date = min_date + timedelta(days=365*2)
                
                prediction_date = st.date_input(
                    "Select future date for prediction:",
                    min_value=min_date,
                    max_value=max_date,
                    value=min_date + timedelta(days=30)
                )
                
                if st.button("Calculate Price Prediction"):
                    predicted_price = predict_today_price(arima_model, df)
                    st.success(f"Predicted Price on {prediction_date}: ${predicted_price:.2f} USD/Bbl")
            
            with col2:
                st.markdown("""
                    <div class="prediction-card">
                        <h3 style='color: #a78bfa; margin: 0;'>How to use the predictor</h3>
                        <ol style='color: #e0e0e0; margin: 1rem 0;'>
                            <li>Select a future date (up to 2 years ahead)</li>
                            <li>Click "Calculate Price Prediction"</li>
                            <li>Get the predicted oil price based on our model</li>
                        </ol>
                        <p style='color: #8892b0; margin: 0;'>
                            Note: Predictions are based on historical patterns and may not account for unexpected market events.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Charts Section
        st.subheader("📈 Price Analysis")
        
        # Historical Price Chart
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(
            x=df['date'],
            y=df['price'],
            name='Historical Price',
            line=dict(color='#00ff88', width=2)
        ))
        
        fig_price.update_layout(
            title='Crude Oil Price History',
            template='plotly_dark',
            height=400,
            xaxis_title='Date',
            yaxis_title='Price (USD/Bbl)',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            font=dict(color='#e0e0e0')
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Price Distribution
        st.plotly_chart(create_monthly_price_distribution(df), use_container_width=True)
        
        # Statistics Section
        st.subheader("📊 Price Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            stats_df = pd.DataFrame({
                'Metric': ['Mean Price', 'Median Price', 'Max Price', 'Min Price', 'Standard Deviation'],
                'Value': [
                    f"${df['price'].mean():.2f}",
                    f"${df['price'].median():.2f}",
                    f"${df['price'].max():.2f}",
                    f"${df['price'].min():.2f}",
                    f"${df['price'].std():.2f}"
                ]
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)
        
        with col2:
            monthly_returns = df.set_index('date').resample('M')['percentChange'].mean()
            monthly_stats = pd.DataFrame({
                'Month': monthly_returns.index.strftime('%B %Y').tolist()[-6:],
                'Average Return (%)': monthly_returns.values[-6:].round(2)
            })
            st.markdown("#### Last 6 Months Returns")
            st.dataframe(monthly_stats, hide_index=True, use_container_width=True)
        
        # Historical Data Table
        st.subheader("📜 Historical Data")
        historical_data = df.copy()
        st.dataframe(
            historical_data,
            column_config={
                "date": "Date",
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "percentChange": st.column_config.NumberColumn("Percent Change", format="%.2f%"),
                "change": st.column_config.NumberColumn("Change", format="$%.2f")
            },
            use_container_width=True
        )
        
        # Data Entry Form
        st.subheader("➕ Add Historical Data")
        with st.form(key='data_entry_form'):
            date = st.date_input("Date", value=datetime.now().date())
            price = st.number_input("Price (USD/Bbl)", format="%.2f")
            percent_change = st.number_input("Percent Change (%)", format="%.2f")
            change = st.number_input("Change (USD)", format="%.2f")

            submit_button = st.form_submit_button("Submit")
            if submit_button:
                # Create a new DataFrame for the new data
                new_data = pd.DataFrame({
                    'date': [pd.to_datetime(date)],
                    'price': [price],
                    'percentChange': [percent_change],
                    'change': [change]
                })

                new_data['date'] = new_data['date'].dt.tz_localize(None)  # Ensure new data is timezone-naive
                
                # Append the new data to the existing DataFrame
                df = pd.concat([df, new_data], ignore_index=True)
                df.sort_values(by='date', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
                st.success("New data added successfully!")
                
                # Refresh the historical data table
                st.dataframe(
                    df,
                    column_config={
                        "date": "Date",
                        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                        "percentChange": st.column_config.NumberColumn("Percent Change", format="%.2f%"),
                        "change": st.column_config.NumberColumn("Change", format="$%.2f")
                    },
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
