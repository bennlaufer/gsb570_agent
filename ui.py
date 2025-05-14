import streamlit as st
from agent.agent_setup import setup_agent
import json
from pathlib import Path

# Load schema
base_path = Path(__file__).resolve().parent
schema_path = base_path / 'data' / 'salesforce_schema.json'
with schema_path.open('r', encoding='utf-8') as f:
    salesforce_schema = json.load(f)

# Setup agent
agent_executor, memory = setup_agent(salesforce_schema)

def run_ui():
    st.set_page_config(page_title="PRIZM Agent Chat", layout="centered")
    st.title("Chat Agent - Ben Laufer")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask the agent something...")

    if user_input:
        response = agent_executor.invoke({"input": user_input})
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Agent", response["output"]))

    for speaker, msg in st.session_state.chat_history:
        with st.chat_message(speaker):
            st.markdown(msg)
