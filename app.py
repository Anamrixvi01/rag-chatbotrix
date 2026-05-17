import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import json

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

# Use Managed Identity in production, API key locally
if AZURE_OPENAI_API_KEY:
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version="2024-02-01"
    )
    search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
else:
    azure_credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        azure_credential,
        "https://cognitiveservices.azure.com/.default"
    )
    openai_client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_ad_token_provider=token_provider,
        api_version="2024-02-01"
    )
    search_credential = azure_credential

search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name="rag-chatbotrix-index",
    credential=search_credential
)

app = Flask(__name__)

def embed_question(question):
    response = openai_client.embeddings.create(
        input=question,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding

def retrieve_chunks(question, top_k=3):
    question_vector = embed_question(question)
    vector_query = VectorizedQuery(
        vector=question_vector,
        k_nearest_neighbors=top_k,
        fields="embedding"
    )
    results = search_client.search(
        search_text=question,
        vector_queries=[vector_query],
        select=["content"],
        top=top_k
    )
    chunks = [result["content"] for result in results]
    return chunks

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400

    chunks = retrieve_chunks(question)
    context = "\n\n".join(chunks)

    def generate():
        stream = openai_client.chat.completions.create(
            model=AZURE_OPENAI_CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant for Contoso Electronics. "
                        "Answer questions using ONLY the context provided. "
                        "If the answer is not in the context, say: 'I don't have that information in my knowledge base.' "
                        "Be concise, friendly, and professional."
                    )
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ],
            temperature=0.2,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield f"data: {json.dumps({'content': delta.content})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")

@app.route("/sources", methods=["POST"])
def sources():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"sources": []})
    chunks = retrieve_chunks(question)
    previews = [c[:120] + "..." if len(c) > 120 else c for c in chunks]
    return jsonify({"sources": previews})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8000)
