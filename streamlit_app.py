import streamlit as st
from main import ResearchCrew  # Import the ResearchCrew class from main.py
import os

st.title('Your Technology Research Assistant')
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["SERPER_API_KEY"] = st.secrets["SERPER_API_KEY"]

with st.sidebar:
    st.header('Enter Technology Research Details')
    topic = st.text_input("Technology topic of your research:")
    st.write(f"topic is: {topic}")

if st.button('Run Research'):
    if not topic:
        st.error("Please fill all the fields.")
    else:
        topic = topic
        research_crew = ResearchCrew(topic)
        result = research_crew.run()
        st.subheader("Results of your research project:")
        st.write(result)
