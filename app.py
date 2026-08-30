import os
import io
import hashlib

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from google import genai
import google.genai.types as types
load_dotenv()

# -----------------------------
# Setup
# -----------------------------


# Directly set your API key here to test
# Load your Gemini API key from the .env file (make sure GEMINI_API_KEY is in your .env)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Please set GEMINI_API_KEY in your .env file.")
    st.stop()

# Initialize the official Google GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

# Use Gemini's fast, free-tier model for text generation
MODEL_NAME = "gemini-3.6-flash"  # or "gemini-2.5-flash" / "gemini-1.5-flash" depending on your SDK version

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_documents")

# -----------------------------
# Helpers
# -----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    file_bytes = uploaded_file.getvalue()
    pdf = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


def get_collection_name(file_bytes: bytes) -> str:
    doc_hash = hashlib.md5(file_bytes).hexdigest()[:12]
    return f"study_{doc_hash}"


def build_vector_store(text: str, collection_name: str):
    chunks = chunk_text(text)

    # Re-create collection if it already exists
    try:
        chroma_client.delete_collection(collection_name)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    return collection


def retrieve_context(collection_name: str, query: str, k: int = 4) -> str:
    collection = chroma_client.get_or_create_collection(
        name=collection_name
    )
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    docs = results["documents"][0] if results and results.get("documents") else []
    if not docs:
        return ""

    return "\n\n---\n\n".join(docs)

def generate_llm_response(system_prompt: str, user_prompt: str) -> str:
    # Combine system prompt and user prompt for Gemini
    full_prompt = f"System Instructions:\n{system_prompt}\n\nUser Question:\n{user_prompt}"
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=full_prompt,
    )
    return response.text



def answer_question_with_context(question: str, context: str) -> str:
    system_prompt = (
        "You are a helpful college study assistant. "
        "Use only the provided context when answering. "
        "If the answer is not present in the context, say you cannot find it in the document. "
        "Be clear, concise, and accurate."
    )

    user_prompt = f"""
Context:
{context}

Question:
{question}

Answer in a helpful student-friendly way.
"""
    return generate_llm_response(system_prompt, user_prompt)


def summarize_text(text: str) -> str:
    system_prompt = (
        "You are a college note summarizer. "
        "Turn the text into simple, concise revision notes."
    )

    user_prompt = f"""
Summarize the following text for a college student.

Include:
- main ideas
- key terms
- short revision points

Text:
{text[:15000]}
"""
    return generate_llm_response(system_prompt, user_prompt)


def generate_flashcards(text: str) -> str:
    system_prompt = (
        "You are a flashcard generator for college students. "
        "Create useful study flashcards from the content."
    )

    user_prompt = f"""
Create 10 flashcards from the following text.

Format:
Q: ...
A: ...

Text:
{text[:15000]}
"""
    return generate_llm_response(system_prompt, user_prompt)


def generate_quiz(text: str) -> str:
    system_prompt = (
        "You are a quiz creator for college students. "
        "Create practice questions that help with revision."
    )

    user_prompt = f"""
Create 10 quiz questions from the following text.

Mix:
- 4 easy
- 4 medium
- 2 hard

Provide answers below each question.

Text:
{text[:15000]}
"""
    return generate_llm_response(system_prompt, user_prompt)


def help_with_assignment(topic: str) -> str:
    system_prompt = (
        "You are a college assignment assistant. "
        "Help the student understand the topic, structure an answer, and suggest what to study. "
        "Do not invent fake citations."
    )

    user_prompt = f"""
Topic / assignment question:
{topic}

Please provide:
1. simple explanation
2. suggested outline
3. key points to include
4. what to research next
5. if possible, suggest types of references to look for
"""
    return generate_llm_response(system_prompt, user_prompt)


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="College AI Study Agent", page_icon="🎓", layout="wide")
st.title("🎓 College AI Study Agent")

st.sidebar.header("How to use")
st.sidebar.write(
    """
- Upload a PDF lecture note, chapter, or research paper
- Ask questions from the document
- Generate summaries, flashcards, and quizzes
- Use assignment help for topic planning

**Note:** This tool is for studying and learning, not cheating in exams.
"""
)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if "text" not in st.session_state:
    st.session_state.text = ""
if "collection_name" not in st.session_state:
    st.session_state.collection_name = ""
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    collection_name = get_collection_name(file_bytes)

    if not st.session_state.doc_loaded or st.session_state.collection_name != collection_name:
        with st.spinner("Processing PDF..."):
            text = extract_text_from_pdf(uploaded_file)

            if not text:
                st.error("No text found in the PDF. It may be scanned or image-based.")
                st.stop()

            build_vector_store(text, collection_name)

            st.session_state.text = text
            st.session_state.collection_name = collection_name
            st.session_state.doc_loaded = True

        st.success("PDF processed successfully!")

    text = st.session_state.text
    collection_name = st.session_state.collection_name

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["💬 Ask PDF", "📝 Summarize", "🧠 Flashcards", "🧪 Quiz", "📚 Assignment Help"]
    )

    with tab1:
        st.subheader("Ask questions from the uploaded PDF")
        question = st.text_input("Your question", placeholder="What is the main idea of this chapter?")
        if st.button("Get Answer"):
            if question.strip():
                with st.spinner("Thinking..."):
                    context = retrieve_context(collection_name, question, k=4)
                    answer = answer_question_with_context(question, context)
                    st.write(answer)
            else:
                st.warning("Please enter a question.")

    with tab2:
        st.subheader("Generate a summary")
        if st.button("Summarize Document"):
            with st.spinner("Summarizing..."):
                summary = summarize_text(text)
                st.write(summary)

    with tab3:
        st.subheader("Generate flashcards")
        if st.button("Create Flashcards"):
            with st.spinner("Creating flashcards..."):
                flashcards = generate_flashcards(text)
                st.write(flashcards)

    with tab4:
        st.subheader("Generate quiz questions")
        if st.button("Create Quiz"):
            with st.spinner("Creating quiz..."):
                quiz = generate_quiz(text)
                st.write(quiz)

    with tab5:
        st.subheader("Assignment / topic help")
        topic = st.text_area("Enter your assignment topic or question")
        if st.button("Help Me"):
            if topic.strip():
                with st.spinner("Preparing help..."):
                    result = help_with_assignment(topic)
                    st.write(result)
            else:
                st.warning("Please enter a topic.")
else:
    st.info("Upload a PDF to start using the agent.")