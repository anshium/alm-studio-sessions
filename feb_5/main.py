import time
import os
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from google import genai

try:
    with open("key.txt", "r") as f:
        GEMINI_API_KEY = f.read().strip()
except FileNotFoundError:
    print("Error: key.txt not found.")
    exit()

client = genai.Client(api_key=GEMINI_API_KEY)
LLM_MODEL = "gemini-2.5-flash"

bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def load_and_chunk_pdf(pdf_path, chunk_size=500):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

def run_rag_pipeline(query, chunks):
    print(f"\n--- Processing Query: {query} ---")
    start_time_baseline = time.time()
    
    corpus_embeddings = bi_encoder.encode(chunks, convert_to_tensor=True)
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=10)[0]
    baseline_indices = [hit['corpus_id'] for hit in hits]
    
    latency_baseline = time.time() - start_time_baseline
    print(f"Baseline Retrieval Latency: {latency_baseline:.4f}s")

    start_time_advanced = time.time()
    
    cross_inp = [[query, chunks[i]] for i in baseline_indices]
    cross_scores = cross_encoder.predict(cross_inp)
    
    reranked_indices = np.argsort(cross_scores)[::-1]
    
    top_3_indices = [baseline_indices[i] for i in reranked_indices[:3]]
    top_3_chunks = [chunks[i] for i in top_3_indices]
    
    latency_advanced = (time.time() - start_time_advanced) + latency_baseline
    print(f"Advanced Retrieval Latency (including Re-ranking): {latency_advanced:.4f}s")

    context = "\n\n".join(top_3_chunks)
    prompt = (
        f"Context provided below:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Instructions: Answer the question precisely using ONLY the context provided."
    )
    
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    
    return {
        "answer": response.text,
        "baseline_top_5": [chunks[i][:100] for i in baseline_indices[:5]],
        "reranked_top_3": [chunks[i][:100] for i in top_3_indices],
        "latencies": {"baseline": latency_baseline, "advanced": latency_advanced}
    }

if __name__ == "__main__":
    pdf_file = "data/writing-best-practices-rag.pdf"
    
    if os.path.exists(pdf_file):
        document_chunks = load_and_chunk_pdf(pdf_file)
        user_query = "What are the best practices for handling tables and graphical information in source documents to improve RAG performance?"
        
        try:
            results = run_rag_pipeline(user_query, document_chunks)
            print("\n--- FINAL ANSWER ---")
            print(results["answer"])
        except Exception as e:
            print(f"An error occurred during pipeline execution: {e}")
    else:
        print(f"Error: {pdf_file} not found. Check your data/ directory.")