import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Black-Scholes pricing & Greeks
def black_and_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return price, delta, gamma

# Monte Carlo pricing with CI and price samples
def monte_carlo_price(S, K, T, r, sigma, option_type="call", num_simulations=100000):
    z = np.random.normal(0, 1, num_simulations)
    St = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    if option_type == "call":
        payoffs = np.maximum(St - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - St, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    discounted_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs) / np.sqrt(num_simulations)
    ci_lower = price - 1.96 * std_error
    ci_upper = price + 1.96 * std_error
    return price, ci_lower, ci_upper, St

# Dark/light mode toggle (simple CSS hack)
def set_theme(dark_mode):
    if dark_mode:
        st.markdown(
            """
            <style>
            .main {
                background-color: #0E1117;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .main {
                background-color: white;
                color: black;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Streamlit App
st.set_page_config(page_title="Option Pricing", layout="wide")
st.title("📈 Option Pricing Calculator — Black-Scholes & Monte Carlo")

# Sidebar inputs
with st.sidebar:
    st.header("Input Parameters")
    S = st.number_input("Current Stock Price (S)", value=100.0, min_value=0.01, step=1.0, help="Current price of the underlying asset")
    K = st.number_input("Strike Price (K)", value=100.0, min_value=0.01, step=1.0, help="Price at which option can be exercised")
    T = st.number_input("Time to Maturity (Years)", value=1.0, min_value=0.01, step=0.01, help="Time until option expiry")
    r = st.number_input("Risk-Free Rate (r)", value=0.05, min_value=0.0, max_value=1.0, format="%.4f", help="Annualized risk-free interest rate")
    sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=5.0, format="%.4f", help="Annualized volatility of underlying asset")
    option_type = st.selectbox("Option Type", ("call", "put"))
    num_simulations = st.slider("Monte Carlo Simulations", 1000, 500000, 100000, step=1000)
    dark_mode = st.checkbox("Dark Mode")

set_theme(dark_mode)

if st.button("Calculate Prices"):
    bs_price, delta, gamma = black_and_scholes(S, K, T, r, sigma, option_type)
    mc_price, ci_lower, ci_upper, St = monte_carlo_price(S, K, T, r, sigma, option_type, num_simulations)

    # Results in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Black-Scholes Model")
        st.write(f"Price: **{bs_price:.4f}**")
        st.write(f"Delta: **{delta:.4f}**")
        st.write(f"Gamma: **{gamma:.6f}**")

    with col2:
        st.subheader("Monte Carlo Simulation")
        st.write(f"Price: **{mc_price:.4f}**")
        st.write(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
        st.write(f"Simulations: {num_simulations}")

        # Plot histogram of simulated terminal prices
        fig, ax = plt.subplots()
        ax.hist(St, bins=50, color='#4A90E2', alpha=0.7)
        ax.axvline(K, color='red', linestyle='dashed', label='Strike Price')
        ax.set_title("Distribution of Simulated Terminal Stock Prices")
        ax.set_xlabel("Stock Price at Maturity")
        ax.set_ylabel("Frequency")
        ax.legend()
        st.pyplot(fig)
