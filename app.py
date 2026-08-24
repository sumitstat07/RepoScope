import streamlit as st

from src.retriever import RAGRetriever
from src.generator import generate_answer
from src.vector_store import VectorStoreManager
from src.config import TOP_K, GROQ_API_KEY

st.set_page_config(page_title="RepoScope — RBI Policy Q&A", page_icon="📘", layout="wide")


@st.cache_resource
def get_retriever():
    return RAGRetriever()


@st.cache_resource
def get_doc_count():
    return VectorStoreManager().count()


st.title("📘 RepoScope")
st.caption("Ask questions about RBI monetary policy (repo rate decisions). Answers are generated only from the indexed documents, with sources shown.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 8, TOP_K)
    st.divider()
    st.metric("Chunks indexed", get_doc_count())
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Documents split into ~500-char chunks\n"
        "2. Embedded with all-MiniLM-L6-v2\n"
        "3. Stored in ChromaDB\n"
        "4. Question embedded, nearest chunks retrieved\n"
        "5. Groq LLM answers from those chunks only"
    )

EXAMPLES = [
    "What did the RBI decide on the repo rate in June 2025?",
    "What did the RBI say about liquidity conditions in August 2022?",
    "What was the RBI's inflation projection for FY2017-18?",
]

st.write("**Try one:**")
cols = st.columns(len(EXAMPLES))
for i, ex in enumerate(EXAMPLES):
    if cols[i].button(ex, use_container_width=True, key=f"ex{i}"):
        st.session_state["question"] = ex

question = st.text_input(
    "Your question",
    key="question",
    placeholder="e.g. What did RBI say about inflation projections?",
)

if st.button("Get answer", type="primary"):
    q = st.session_state.get("question", "").strip()
    if not q:
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Retrieving context and generating answer..."):
                retriever = get_retriever()
                answer, hits = generate_answer(q, retriever, top_k=top_k)

            st.subheader("Answer")
            st.write(answer)

            st.subheader(f"Sources ({len(hits)} chunks retrieved)")
            for i, h in enumerate(hits, start=1):
                label = f"[Source {i}] {h['source_file']} — page {h['page']} — similarity {h['similarity']}"
                with st.expander(label):
                    st.write(h["text"])
        except Exception as e:
            st.error(f"Something went wrong: {e}")