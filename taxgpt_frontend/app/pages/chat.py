import requests
import streamlit as st

# Function to fetch response from the backend
def get_response(input_text):
    try:
        response = requests.post(
            "http://52.66.236.117:8000/response",
            json={"question": input_text}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"generation": "Error: Unable to fetch the response. Please try again later."}

# Set up Streamlit UI
st.set_page_config(
    page_title="Tax Assistance",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        padding: 20px;
    }
    [data-testid="stSidebar"] h1 {
        color: #10b981;
        font-size: 1.5em;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title(" Your Personal Tax Assistance")

# User input
query = st.text_input(
    "Write your query below:",
    placeholder="E.g., What are the tax slabs for FY 2023-24?"
)

# Submit button
if st.button("Get Answer") and query:
    with st.spinner("Fetching your answer..."):
        response = get_response(query)
        answer = response.get("generation", "Sorry, no answer available.")
    st.markdown(f"### Answer:\n{answer}")
