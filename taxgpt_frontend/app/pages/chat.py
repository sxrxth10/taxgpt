import requests
import streamlit as st


def get_response(input_text):
    try:
        response = requests.post(
            "http://3.109.157.165:8000/response",

            json={"question": input_text}
        )
        response.raise_for_status()
        response_data = response.json()
        print("API response:", response_data)
        return {"generation": response_data.get
                ("generation", "No answer received.")}
    except requests.exceptions.HTTPError as e:
        return {"generation": f"HTTP Error: {str(e)}"}
    except requests.exceptions.RequestException as e:
        return {"generation": f"Error: Unable to fetch the response. {str(e)}"}
    except ValueError as e:
        return {"generation": f"Error: Invalid response format. {str(e)}"}

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

# Submit buttonz
if st.button("Get Answer") and query:
    with st.spinner("Fetching your answer..."):
        response = get_response(query)
        answer = response.get("generation", "Sorry, no answer available.")
    st.markdown(f"### Answer:\n{answer}")
