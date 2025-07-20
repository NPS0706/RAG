# 🧠 Retrieval-Augmented Generation (RAG) for ICD-10 Medical Document QA

## 📝 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to answer natural language queries from the **ICD-10 medical classification PDF**. It combines semantic search using FAISS and contextual answering using a hosted LLM (`llama3-8b-8192` via Groq) to enable intelligent question answering over unstructured medical documentation.

---

## ✅ Features Implemented

- 📄 Automatic document chunking with token overlap
- 🔍 Semantic vector embeddings using `all-MiniLM-L6-v2`
- ⚡ Vector similarity search using FAISS
- 🤖 Answer generation using Groq-hosted `llama3-8b-8192`
- 🔄 End-to-end pipeline: PDF → Embeddings → FAISS → LLM → Answer

---

## ⚙️ Tech Stack

| Component          | Tool/Model                                |
|--------------------|-------------------------------------------|
| Document Parser    | PyMuPDF                                   |
| Text Splitter      | LangChain `RecursiveCharacterTextSplitter`|
| Embeddings         | `sentence-transformers/all-MiniLM-L6-v2`  |
| Vector Search      | FAISS                                     |
| LLM                | `llama3-8b-8192` via Groq API             |
| Notebook Interface | Google Colab                              |
| Language           | Python 3.11                               |

---

## 💡 Design Decisions & Assumptions

- Opted for `all-MiniLM-L6-v2` due to its balance of performance and speed.
- Used token-based chunking (300 tokens with 50-token overlap) to preserve semantic boundaries.
- Hosted inference via Groq for fast and reliable LLM access.
- Adopted a simple prompt structure:
- Assumed input document is semantically rich and well-formatted.
- Focused purely on backend retrieval and QA—front-end or deployment was not prioritized.

---

## 🤖 Use of ChatGPT

ChatGPT was used for:

- Designing the RAG architecture and deciding on appropriate tools
- Writing the modular code for:
- PDF parsing
- Chunking and embeddings
- FAISS indexing and querying
- Groq LLM-based answer generation
- Drafting prompt templates
- Writing this professional README
- Debugging model migration (`mixtral` → `llama3`)
- Benchmark testing with sample queries

All core code was reviewed and tested manually before finalizing.

---

## ❓ Sample Queries & Responses

### 🔹 Query 1:
**Question:**  
What are the diagnostic criteria for schizophrenia?

**Answer:**
According to the provided context from the ICD-10, the diagnostic criteria for schizophrenia are not explicitly stated. However, it is mentioned that the general criteria for a diagnosis of schizophrenia must be satisfied, and the introduction to F20 provides more information on the diagnostic criteria for schizophrenia.

However, if we look at the diagnostic guidelines for other types of schizophrenia, such as hebephrenia and simple schizophrenia, we can infer some of the diagnostic criteria for schizophrenia. For example, the guidelines for simple schizophrenia state that the individual must have a history of at least one clear-cut psychotic episode meeting the diagnostic criteria for schizophrenia, and the intensity and frequency of florid symptoms such as delusions and hallucinations must have been minimal or substantially reduced for a period of at least 1 year.

It's also mentioned that the disorder runs a chronic course with fluctuations of intensity, and occasionally it evolves into overt schizophrenia. This suggests that schizophrenia is a chronic and persistent disorder that can have varying levels of severity over time.

In summary, while the diagnostic criteria for schizophrenia are not explicitly stated in the provided context, we can infer that it involves a history of at least one clear-cut psychotic episode, a period of reduced symptoms, and a chronic course with fluctuations of intensity.


---

### 🔹 Query 2:
**Code Example:**
```python
response = rag_qa("Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission")
print("🔹Answer:\n", response)


Based on the provided context from the ICD-10, the correct coded classification for the diagnosis "Recurrent depressive disorder, currently in remission" is:

F33.4

This code is specified for "Recurrent depressive disorder, currently in remission", which indicates that the individual has had previous episodes of depression, but is currently not experiencing any symptoms.


