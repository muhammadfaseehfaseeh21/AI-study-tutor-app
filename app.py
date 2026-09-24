import os
import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM

# -------------------------------------------------------------------
# Page Configuration & UI Layout
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AI Study Tutor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 AI Study Tutor (CrewAI)")
st.caption("Powered by CrewAI & `openai/gpt-oss-120b`")

# -------------------------------------------------------------------
# Sidebar API Configuration
# -------------------------------------------------------------------
with st.sidebar:
    st.header("🔑 Model & API Settings")
    
    # Fetch default key/url from Streamlit secrets or OS environment
    env_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    env_base_url = os.getenv("OPENAI_API_BASE") or st.secrets.get("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
    
    api_key = st.text_input("API Key", value=env_api_key, type="password")
    base_url = st.text_input("API Base URL", value=env_base_url)
    model_name = st.text_input("Model Identifier", value="openrouter/openai/gpt-oss-120b")
    
    st.divider()
    tutor_mode = st.radio(
        "Select Tutor Mode",
        ["Socratic Explanation", "Practice Quiz Generator", "Concept Summary & Flashcards"],
        index=0
    )

# -------------------------------------------------------------------
# Helper: Initialize CrewAI Agent & Task Execution
# -------------------------------------------------------------------
def run_tutor_crew(user_topic: str, mode: str, api_key: str, base_url: str, model: str):
    # Set environment variables directly so underlying libraries (LiteLLM/CrewAI) pick them up
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENROUTER_API_KEY"] = api_key

    # Initialize CrewAI LLM with explicit API key and base URL
    llm = LLM(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3
    )

    # 1. Define the Single Agent
    study_tutor_agent = Agent(
        role="Expert Socratic Study Tutor",
        goal="Help students deeply understand academic topics through interactive, structured, and pedagogical assistance.",
        backstory=(
            "You are a patient, world-class academic tutor. You avoid directly giving final answers "
            "when explaining concepts, preferring step-by-step guidance. You write mathematical expressions "
            "using clean LaTeX ($...$) and format outputs clearly using standard Markdown."
        ),
        verbose=False,
        allow_delegation=False,
        llm=llm
    )

    # 2. Dynamic Task Configuration based on Selected Mode
    if mode == "Socratic Explanation":
        description = (
            f"Explain the topic '{user_topic}' using a step-by-step Socratic approach. "
            "Break down core principles using everyday analogies, provide key definitions, and finish "
            "with 2 guided follow-up questions to test the student's understanding."
        )
        expected_output = "A well-structured explanation with analogies, clear headers, latex math where applicable, and 2 probing questions."

    elif mode == "Practice Quiz Generator":
        description = (
            f"Generate a 3-question practice quiz based on the topic '{user_topic}'. "
            "Include Multiple Choice Questions (MCQs) with options A, B, C, D. Provide an answer key "
            "and detailed explanations at the end."
        )
        expected_output = "A formatted Markdown quiz with questions, multiple-choice options, and an answer key with explanations."

    else:  # Concept Summary & Flashcards
        description = (
            f"Create a high-yield study sheet for '{user_topic}'. "
            "Include: 1) A bulleted summary of key takeaways, 2) Important formulas/terms, and "
            "3) 5 Flashcards formatted as Q&A pairs."
        )
        expected_output = "A study guide containing bulleted key takeaways, term definitions, and 5 distinct flashcard Q&A blocks."

    # 3. Create Task
    tutor_task = Task(
        description=description,
        expected_output=expected_output,
        agent=study_tutor_agent
    )

    # 4. Form Crew and Kickoff
    crew = Crew(
        agents=[study_tutor_agent],
        tasks=[tutor_task],
        process=Process.sequential
    )

    result = crew.kickoff()
    return result

# -------------------------------------------------------------------
# Main UI Logic
# -------------------------------------------------------------------
user_topic = st.text_area(
    "Enter a topic, homework question, or concept you want to study:",
    placeholder="e.g., Photosynthesis light reactions, Bayes' Theorem, or Python Recursion",
    height=100
)

if st.button("Get Guidance", type="primary", use_container_width=True):
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
    elif not user_topic.strip():
        st.warning("Please enter a study topic first.")
    else:
        with st.spinner("Your AI Tutor is processing..."):
            try:
                response = run_tutor_crew(
                    user_topic=user_topic,
                    mode=tutor_mode,
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name
                )
                st.markdown("### 📚 Tutor Response")
                st.markdown(response.raw if hasattr(response, 'raw') else str(response))
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
