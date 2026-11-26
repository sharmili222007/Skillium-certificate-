
import streamlit as st
import faiss
import numpy as np

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ==================================
# 1. PAGE CONFIGURATION
# ==================================

st.set_page_config(
    page_title="Mini RAG Document AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Mini RAG - Document AI")
st.write("Upload a document and ask questions about it.")


# ==================================
# 2. LOAD EMBEDDING MODEL
# ==================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ==================================
# 3. LOAD LANGUAGE MODEL
# ==================================

@st.cache_resource
def load_llm():

    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )


# ==================================
# 4. READ DOCUMENT
# ==================================

def read_document(uploaded_file):

    pages = []

    if uploaded_file.name.lower().endswith(".pdf"):

        pdf = PdfReader(uploaded_file)

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text() or ""

            if text.strip():

                pages.append({
                    "page": page_number,
                    "text": text
                })

    elif uploaded_file.name.lower().endswith(".txt"):

        text = uploaded_file.getvalue().decode(
            "utf-8",
            errors="ignore"
        )

        pages.append({
            "page": 1,
            "text": text
        })

    return pages


# ==================================
# 5. CREATE DOCUMENT CHUNKS
# ==================================

def create_chunks(pages, max_words=180):

    chunks = []

    for page in pages:

        words = page["text"].split()

        for i in range(0, len(words), max_words):

            chunk_text = " ".join(
                words[i:i + max_words]
            )

            if chunk_text.strip():

                chunks.append({
                    "text": chunk_text,
                    "page": page["page"]
                })

    return chunks


# ==================================
# 6. CREATE FAISS VECTOR DATABASE
# ==================================

def create_vector_database(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ==================================
# 7. RETRIEVE RELEVANT PARAGRAPHS
# ==================================

def retrieve_chunks(question, index, chunks, top_k=3):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(question_embedding)

    k = min(top_k, len(chunks))

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx != -1:

            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score)
            })

    return results


# ==================================
# 8. GENERATE ANSWER USING FLAN-T5
# ==================================

def generate_answer(question, retrieved_chunks, model):

    context = "\n\n".join(
        chunk["text"]
        for chunk in retrieved_chunks
    )

    prompt = f"""
Answer the question using only the context below.

If the answer is not available in the context,
say that it is not available in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    response = model(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )

    return response[0]["generated_text"]


# ==================================
# 9. DOCUMENT UPLOAD
# ==================================

uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"]
)

if uploaded_file:

    st.success("Document uploaded: " + uploaded_file.name)

    if st.button("Process Document"):

        with st.spinner("Reading and processing document..."):

            pages = read_document(uploaded_file)

            chunks = create_chunks(pages)

            if not chunks:

                st.error("No readable text found.")

                st.stop()

            index = create_vector_database(chunks)

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index

            st.success("Document processed successfully!")

            st.write("Total pages:", len(pages))
            st.write("Total chunks:", len(chunks))


# ==================================
# 10. QUESTION AND ANSWER
# ==================================

if "index" in st.session_state:

    st.divider()

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Generate Answer"):

        if question.strip():

            try:

                with st.spinner("Loading AI model and generating answer..."):

                    retrieved_chunks = retrieve_chunks(
                        question,
                        st.session_state["index"],
                        st.session_state["chunks"]
                    )

                    model = load_llm()

                    answer = generate_answer(
                        question,
                        retrieved_chunks,
                        model
                    )

                st.subheader("🤖 AI Answer")

                st.write(answer)

                st.subheader("📖 Retrieved Source Paragraphs")

                for i, chunk in enumerate(
                    retrieved_chunks,
                    start=1
                ):

                    with st.expander(
                        f"Source {i} | Page {chunk['page']}"
                    ):

                        st.write(chunk["text"])

                        st.write(
                            "Similarity:",
                            round(chunk["score"], 3)
                        )

            except Exception as e:

                st.error("Error: " + str(e))

else:

    st.info(
        "Upload a PDF or TXT file and click Process Document."
    )


# ==================================
# END OF MINI RAG APPLICATION
# ==================================
