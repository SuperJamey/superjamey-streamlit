import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Data Visualization", page_icon="📊", layout="wide")

st.title("📊 Interactive Data Visualization")
st.markdown("Explore various data visualization techniques using Plotly and Streamlit")

# Sidebar controls
st.sidebar.header("Visualization Controls")
viz_type = st.sidebar.selectbox(
    "Choose Visualization Type",
    ["Line Chart", "Bar Chart", "Scatter Plot", "Pie Chart", "Heatmap"]
)

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100)
    df = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(100, 1000, 100),
        'Profit': np.random.randint(50, 500, 100),
        'Orders': np.random.randint(10, 100, 100),
        'Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100)
    })
    return df

df = generate_sample_data()

# Display data info
with st.expander("📋 View Sample Data"):
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Dataset contains {len(df)} rows")

st.markdown("---")

# Visualization based on selection
if viz_type == "Line Chart":
    st.subheader("📈 Sales & Profit Trends Over Time")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_profit = st.checkbox("Show Profit", value=True)
        show_sales = st.checkbox("Show Sales", value=True)
    
    fig = go.Figure()
    
    if show_sales:
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Sales'],
            mode='lines',
            name='Sales',
            line=dict(color='#667eea', width=2)
        ))
    
    if show_profit:
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Profit'],
            mode='lines',
            name='Profit',
            line=dict(color='#764ba2', width=2)
        ))
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Bar Chart":
    st.subheader("📊 Sales by Category")
    
    category_sales = df.groupby('Category')['Sales'].sum().reset_index()
    
    fig = px.bar(
        category_sales,
        x='Category',
        y='Sales',
        color='Sales',
        color_continuous_scale='Viridis',
        title='Total Sales by Category'
    )
    
    fig.update_layout(height=500, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Show stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sales", f"${category_sales['Sales'].sum():,.0f}")
    with col2:
        st.metric("Average Sale", f"${category_sales['Sales'].mean():,.0f}")
    with col3:
        st.metric("Top Category", category_sales.loc[category_sales['Sales'].idxmax(), 'Category'])

elif viz_type == "Scatter Plot":
    st.subheader("🎯 Sales vs Profit Correlation")
    
    color_by = st.sidebar.radio("Color by:", ["Category", "Orders"])
    
    fig = px.scatter(
        df,
        x='Sales',
        y='Profit',
        color=color_by,
        size='Orders',
        hover_data=['Date'],
        title='Sales vs Profit Analysis',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(height=500, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate correlation
    correlation = df['Sales'].corr(df['Profit'])
    st.info(f"📊 Correlation coefficient: {correlation:.3f}")

elif viz_type == "Pie Chart":
    st.subheader("🥧 Category Distribution")
    
    category_counts = df['Category'].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title='Orders by Category',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

elif viz_type == "Heatmap":
    st.subheader("🔥 Correlation Heatmap")
    
    # Calculate correlation matrix
    corr_matrix = df[['Sales', 'Profit', 'Orders']].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='RdBu',
        aspect='auto',
        title='Feature Correlation Matrix'
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("💡 Values closer to 1 or -1 indicate stronger correlation")

# Interactive filters
st.markdown("---")
st.subheader("🔍 Interactive Filtering")

col1, col2 = st.columns(2)

with col1:
    date_range = st.date_input(
        "Select Date Range",
        value=(df['Date'].min(), df['Date'].max()),
        min_value=df['Date'].min(),
        max_value=df['Date'].max()
    )

with col2:
    selected_categories = st.multiselect(
        "Select Categories",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )

# Filter data
if len(date_range) == 2:
    mask = (df['Date'] >= pd.Timestamp(date_range[0])) & (df['Date'] <= pd.Timestamp(date_range[1]))
    filtered_df = df[mask]
    filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)]
    
    # Show filtered stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.0f}")
    with col2:
        st.metric("Total Profit", f"${filtered_df['Profit'].sum():,.0f}")
    with col3:
        st.metric("Total Orders", f"{filtered_df['Orders'].sum():,}")
    with col4:
        st.metric("Records", len(filtered_df))