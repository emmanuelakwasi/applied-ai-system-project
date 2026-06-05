import os
import streamlit as st
from rag_pipeline import RAGAssistant

# Paths (reuse from rag_cli.py)
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
CSV_PATH = os.path.join(_ROOT_DIR, "data", "songs.csv")
DOCS_PATHS = [
    os.path.join(_ROOT_DIR, "data", "genre_profiles.json"),
    os.path.join(_ROOT_DIR, "data", "activity_contexts.json"),
    os.path.join(_ROOT_DIR, "data", "artist_notes.json"),
]

st.set_page_config(page_title="VibeMatch RAG Music Assistant", layout="wide")
st.title("🎵 VibeMatch RAG Music Assistant")
st.markdown("""
Type a natural language query (e.g. _recommend upbeat music for a morning workout_, _explain why 'Midnight Coding' is good for studying_, _classify the mood of 'Night Drive Loop'_, _debug my playlist: Sunrise City → Storm Runner → Library Rain_)
""")

api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    st.warning("ANTHROPIC_API_KEY environment variable is not set. Claude-powered answers will not work.")

# Initialize assistant (cache to avoid reloading)
@st.cache_resource(show_spinner=False)
def get_assistant():
    return RAGAssistant(csv_path=CSV_PATH, docs_paths=DOCS_PATHS, api_key=api_key)

assistant = get_assistant()

with st.form("query_form"):
    query = st.text_input("Ask anything about the music catalog:", "recommend chill music for late-night studying")
    submitted = st.form_submit_button("Get Recommendation")

if submitted and query.strip():
    with st.spinner("Retrieving recommendations and generating answer..."):
        # Retrieve top-3 songs (TF-IDF)
        retrieved = assistant.kb.retrieve(query, k=3)
        st.subheader("🎶 Top Retrieved Songs (TF-IDF Similarity)")
        for s in retrieved:
            st.markdown(f"- **{s['title']}** by *{s['artist']}*  [Genre: {s['genre']}, Mood: {s['mood']}, Energy: {s['energy']:.2f}]")

        # Get AI answer (streamed)
        st.subheader("🤖 VibeMatch AI Recommendation")
        # Stream output to Streamlit
        import sys
        from io import StringIO
        output_buffer = StringIO()
        sys_stdout = sys.stdout
        sys.stdout = output_buffer
        try:
            assistant.ask(query, k=3, stream=True)
        except Exception as e:
            st.error(f"Error calling Claude API: {e}")
        finally:
            sys.stdout = sys_stdout
        st.markdown(output_buffer.getvalue())
