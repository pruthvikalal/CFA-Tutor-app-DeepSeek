import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📘 CFA Tutor App (DeepSeek R1)")

# Upload PDF
uploaded_file = st.file_uploader("📄 Upload your CFA PDF (e.g., Ethics)", type="pdf")

# User question
query = st.text_input("💬 Ask your CFA question:")

# Load DeepSeek Model (only once)
@st.cache_resource
def load_deepseek_model():
    model_name = "deepseek-reasoner"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_deepseek_model()

# On button click
if st.button("✅ Get Answer"):
    if not uploaded_file:
        st.error("❌ Please upload a PDF file.")
    elif not query:
        st.error("❌ Please enter a CFA question.")
    else:
        try:
            os.makedirs("cache", exist_ok=True)
            os.makedirs("faiss_store", exist_ok=True)
            pdf_path = os.path.join("cache", uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            index_folder = os.path.join("faiss_store", uploaded_file.name.split(".")[0])
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            if os.path.exists(index_folder):
                vectorstore = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
            else:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = splitter.split_documents(pages)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                vectorstore.save_local(index_folder)

            results = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in results])

            prompt = f"""You are a CFA tutor who explains concepts clearly and simply.
Use the following context to answer the student's question. Use analogies, exam traps, examples, and a TL;DR at the end. Avoid fluff. Be clear, direct, and structured.

Context:
{context}

Question:
{query}

Answer:"""

            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.4,
                do_sample=True,
                top_p=0.95
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("Answer:")[-1].strip()

            st.markdown("### 🤖 Answer")
            st.markdown(answer)

        except Exception as e:
            st.error(f"⚠️ Unexpected error: {str(e)}")
