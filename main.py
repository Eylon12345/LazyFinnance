import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from phi.assistant import Assistant
from phi.llm.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
import os
from dotenv import load_dotenv

st.set_page_config(page_title="Rent vs. Buy Calculator Tool")

# Load environment variables
#load_dotenv()
openai_api_key = st.secrets["openai_api_key"]

# Title of the application
st.title('Rent vs. Buy Calculator')

# Organize inputs into two lines
col_a, col_b, col_c = st.columns(3)
with col_a:
    current_savings = st.number_input('Current Savings', min_value=0.0, step=10000.0, value=100000.0)
with col_b:
    monthly_income = st.number_input('Monthly Savings', min_value=0.0, step=1000.0, value=10000.0)
with col_c:
    apartment_price = st.number_input('Apartment Price', min_value=100000, step=10000, value=1000000)

col_d, col_e = st.columns(2)
with col_d:
    annual_return_percent = st.selectbox('Expected Annual Return on Investment when Renting', options=[10, 15, 20], index=0)
with col_e:
    years_options = st.selectbox('Select Mortgage Terms (years)', options=[20, 30, 40], index=0)
    if years_options is None:
        years_options = 20

# Define calculation functions
def calculate_mortgage(principal, years, interest_rate=0.03):
    """Calculate monthly mortgage payment."""
    monthly_rate = interest_rate / 12
    payments = years * 12
    return principal * (monthly_rate * (1 + monthly_rate)**payments) / ((1 + monthly_rate)**payments - 1)

def calculate_investment_growth(initial_amount, monthly_income, rental_cost, years, annual_return):
    """Calculate future value of a series of investments, considering rent payments from the initial amount."""
    monthly_savings = monthly_income - rental_cost
    monthly_growth_rate = (1 + annual_return / 100) ** (1/12) - 1
    future_value = initial_amount
    for month in range(years * 12):
        future_value = future_value * (1 + monthly_growth_rate) + monthly_savings
    return future_value

def calculate_future_house_worth(initial_price, years, annual_growth_rate=0.04):
    """Calculate future worth of the house given an annual growth rate."""
    return initial_price * ((1 + annual_growth_rate) ** years)

# Calculation and display block
if st.button('Calculate'):
    interest_rates = {20: 0.07, 30: 0.05, 40: 0.04}
    rental_cost = 0.0023 * apartment_price

    mortgage_payment = calculate_mortgage(apartment_price, years_options, interest_rates[years_options])
    total_paid_mortgage = mortgage_payment * 12 * years_options
    investment_growth = calculate_investment_growth(current_savings, monthly_income, rental_cost, years_options, annual_return_percent)
    rent_paid = rental_cost * 12 * years_options
    house_future_worth = calculate_future_house_worth(apartment_price, years_options)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Mortgage Details")
        st.metric(label=f"{years_options} Years - Total Paid on Mortgage", value=f"${int(total_paid_mortgage):,}", delta="Total Cost")
        st.metric(label=f"{years_options} Years - Future House Worth 🏠", value=f"${int(house_future_worth):,}")
    with col2:
        st.subheader("Rent and Invest Details")
        st.metric(label=f"{years_options} Years - Total Paid on Rent", value=f"${int(rent_paid):,}", delta="Total Cost")
        st.metric(label=f"{years_options} Years - Total Investment Growth", value=f"${int(investment_growth):,}")

    fig, ax = plt.subplots()
    years_range = np.linspace(0, years_options, years_options + 1)
    investments_by_year = [calculate_investment_growth(current_savings, monthly_income, rental_cost, year, annual_return_percent) for year in range(years_options + 1)]
    house_worth_by_year = [calculate_future_house_worth(apartment_price, year) for year in range(years_options + 1)]
    ax.plot(years_range, [investment / 1e6 for investment in investments_by_year], label='Investment Growth from Renting (in millions)', marker='o')
    ax.plot(years_range, [house_worth / 1e6 for house_worth in house_worth_by_year], label='Future House Worth Growth', marker='x')
    ax.set_xlabel('Years')
    ax.set_ylabel('Value (Millions $)')
    ax.set_title('Investment Growth Over Years')
    ax.grid(True)
    ax.legend()
    ax.tick_params(axis='y', which='major', labelsize=10)
    st.pyplot(fig)

st.success("""
| **Factor**     | **Top 3 Pros**                                     | **Top 3 Cons**                                             |
|----------------|-----------------------------------------------------|-----------------------------------------------------------|
| **Renting**    | **1. Flexibility:**   | **1. No Equity:**   |
|                | **2. Lower Upfront Costs:** | **2. Rising Rent:**      |
|                | **3. No Maintenance Responsibilities:** | **3. Limited Customization:** |
| **Buying**     | **1. Stability:**      | **1. High Upfront Costs:** |
|                | **2. Equity Building:**     | **2. Maintenance Costs:** |
|                | **3. Customization:** | **3. Market Risk:**       |
[Streamlit](https://streamlit.io).""", icon="ℹ️")

# Create an instance of the Assistant
assistant = Assistant(
    llm=OpenAIChat(model="gpt-4o", api_key=openai_api_key),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=False, company_info=True, company_news=True)],
    show_tool_calls=True,
)

# Input fields for the stocks to compare
stock1 = st.text_input("Enter the first stock symbol")

if stock1:
    query1 = f"Generate a detailed report on {stock1} using get_company_info and get_current_stock_price."
    query11 = f"Generate a concise news report on {stock1} using company news."
    response1 = assistant.run(query1, stream=False)
    response11 = assistant.run(query11, stream=False)

    st.subheader(f"Report for {stock1}")
    st.write(response1)
    st.write(response11)
