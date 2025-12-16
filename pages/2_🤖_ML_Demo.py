import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="ML Demo", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Demonstrations")
st.markdown("Interactive machine learning examples with real-time training")

# Sidebar
st.sidebar.header("Model Configuration")
model_type = st.sidebar.selectbox(
    "Choose Model Type",
    ["Classification", "Regression"]
)

st.markdown("---")

if model_type == "Classification":
    st.subheader("📊 Binary Classification Demo")
    st.markdown("Predict binary outcomes using Logistic Regression")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Parameters")
        n_samples = st.slider("Number of samples", 100, 1000, 500)
        n_features = st.slider("Number of features", 2, 10, 2)
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random seed", 0, 100, 42)
        
        train_button = st.button("🚀 Train Model", type="primary")
    
    with col2:
        if train_button:
            with st.spinner("Training model..."):
                # Generate data
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=n_features,
                    n_redundant=0,
                    n_clusters_per_class=1,
                    random_state=random_state
                )
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = LogisticRegression(random_state=random_state)
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)
                
                # Metrics
                train_accuracy = accuracy_score(y_train, y_pred_train)
                test_accuracy = accuracy_score(y_test, y_pred_test)
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Train Accuracy", f"{train_accuracy:.2%}")
                with metric_col2:
                    st.metric("Test Accuracy", f"{test_accuracy:.2%}")
                with metric_col3:
                    st.metric("Samples Used", n_samples)
                
                # Visualization for 2D data
                if n_features == 2:
                    st.markdown("#### Decision Boundary Visualization")
                    
                    # Create mesh
                    h = 0.02
                    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
                    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
                    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                         np.arange(y_min, y_max, h))
                    
                    # Predict on mesh
                    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
                    Z = Z.reshape(xx.shape)
                    
                    # Plot
                    fig = go.Figure()
                    
                    # Decision boundary
                    fig.add_trace(go.Contour(
                        x=xx[0],
                        y=yy[:, 0],
                        z=Z,
                        colorscale='RdBu',
                        opacity=0.3,
                        showscale=False
                    ))
                    
                    # Test points
                    fig.add_trace(go.Scatter(
                        x=X_test_scaled[:, 0],
                        y=X_test_scaled[:, 1],
                        mode='markers',
                        marker=dict(
                            color=y_test,
                            colorscale='RdBu',
                            size=8,
                            line=dict(width=1, color='white')
                        ),
                        name='Test Data'
                    ))
                    
                    fig.update_layout(
                        title='Decision Boundary with Test Points',
                        xaxis_title='Feature 1',
                        yaxis_title='Feature 2',
                        height=500,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance (coefficients)
                st.markdown("#### Feature Importance")
                coef_df = pd.DataFrame({
                    'Feature': [f'Feature {i+1}' for i in range(n_features)],
                    'Coefficient': model.coef_[0]
                })
                coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
                
                fig = go.Figure(go.Bar(
                    x=coef_df['Coefficient'],
                    y=coef_df['Feature'],
                    orientation='h',
                    marker_color='#667eea'
                ))
                fig.update_layout(
                    title='Model Coefficients',
                    xaxis_title='Coefficient Value',
                    height=300,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)

else:  # Regression
    st.subheader("📈 Linear Regression Demo")
    st.markdown("Predict continuous values using Linear Regression")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Parameters")
        n_samples = st.slider("Number of samples", 100, 1000, 300)
        n_features = st.slider("Number of features", 1, 10, 1)
        noise = st.slider("Noise level", 0, 50, 20)
        test_size = st.slider("Test size (%)", 10, 40, 20) / 100
        random_state = st.number_input("Random seed", 0, 100, 42)
        
        train_button = st.button("🚀 Train Model", type="primary")
    
    with col2:
        if train_button:
            with st.spinner("Training model..."):
                # Generate data
                X, y = make_regression(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_informative=n_features,
                    noise=noise,
                    random_state=random_state
                )
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Train model
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Metrics
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                st.success("✅ Model trained successfully!")
                
                # Display metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Train R²", f"{train_r2:.3f}")
                with metric_col2:
                    st.metric("Test R²", f"{test_r2:.3f}")
                with metric_col3:
                    st.metric("Train RMSE", f"{train_rmse:.2f}")
                with metric_col4:
                    st.metric("Test RMSE", f"{test_rmse:.2f}")
                
                # Visualization for 1D data
                if n_features == 1:
                    st.markdown("#### Predictions vs Actual Values")
                    
                    fig = go.Figure()
                    
                    # Training points
                    fig.add_trace(go.Scatter(
                        x=X_train.flatten(),
                        y=y_train,
                        mode='markers',
                        name='Train Data',
                        marker=dict(color='#667eea', size=6)
                    ))
                    
                    # Test points
                    fig.add_trace(go.Scatter(
                        x=X_test.flatten(),
                        y=y_test,
                        mode='markers',
                        name='Test Data',
                        marker=dict(color='#764ba2', size=8)
                    ))
                    
                    # Regression line
                    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                    y_line = model.predict(X_line)
                    
                    fig.add_trace(go.Scatter(
                        x=X_line.flatten(),
                        y=y_line,
                        mode='lines',
                        name='Regression Line',
                        line=dict(color='red', width=2)
                    ))
                    
                    fig.update_layout(
                        title='Linear Regression Fit',
                        xaxis_title='Feature Value',
                        yaxis_title='Target Value',
                        height=500,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Residuals plot
                st.markdown("#### Residual Analysis")
                residuals = y_test - y_pred_test
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_pred_test,
                    y=residuals,
                    mode='markers',
                    marker=dict(color='#667eea', size=8),
                    name='Residuals'
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                
                fig.update_layout(
                    title='Residuals vs Predicted Values',
                    xaxis_title='Predicted Values',
                    yaxis_title='Residuals',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Information section
st.markdown("---")
st.info("""
💡 **About this demo:**
- Models are trained on synthetic data generated using scikit-learn
- Adjust parameters in the sidebar to see how they affect model performance
- Classification uses Logistic Regression for binary outcomes
- Regression uses Linear Regression for continuous predictions
""")