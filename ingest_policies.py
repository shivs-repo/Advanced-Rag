"""
ingest_policies.py — Load Corporate_Policies_Dataset_Final.pdf,
                     chunk it, and upsert into Pinecone 'policy-rag' index.
"""
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import fitz  # pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

PINECONE_API_KEY   = os.environ["PINECONE_API_KEY"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
POLICY_INDEX       = os.getenv("PINECONE_POLICY_INDEX", "policy-rag")
NAMESPACE          = "policies"
PDF_FILE           = "Corporate_Policies_Dataset_Final.pdf"
EMBED_MODEL        = "openai/text-embedding-3-small"
VISION_MODEL       = "openai/gpt-4o-mini"
BATCH_SIZE         = 20
CHUNK_SIZE         = 800
CHUNK_OVERLAP      = 100

pc            = Pinecone(api_key=PINECONE_API_KEY)
openai_client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

def extract_text_from_page_image(page, page_num: int) -> tuple[str, list[str]]:
    """Render a PDF page as image, extract text via vision LLM, return (text, image_labels)."""
    pix = page.get_pixmap(dpi=150)
    b64 = base64.b64encode(pix.tobytes("png")).decode()

    # collect embedded image labels for this page
    image_labels = [
        f"Page {page_num + 1} Image {j + 1}"
        for j, _ in enumerate(page.get_images(full=True))
    ]

    resp = openai_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all text from this document page exactly as it appears, including text in tables, diagrams, and images."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
    )
    extracted = resp.choices[0].message.content or ""
    print(f"[ingest_policies] Page {page_num + 1} images: {image_labels or 'none'}")
    print(f"[ingest_policies] Page {page_num + 1} LLM output:\n{extracted}\n{'-'*60}")
    return extracted, image_labels


def load_chunks() -> list[dict]:
    doc      = fitz.open(PDF_FILE)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks   = []
    chunk_id = 0

    for page_num, page in enumerate(doc):
        text, image_labels = extract_text_from_page_image(page, page_num)
        for t in splitter.split_text(text):
            chunks.append({
                "id": f"policy-{chunk_id}",
                "text": t,
                "metadata": {
                    "source": PDF_FILE,
                    "text": t,
                    "page": page_num + 1,
                    "images": ", ".join(image_labels) if image_labels else "",
                },
            })
            chunk_id += 1

    print(f"[ingest_policies] {len(chunks)} chunks from '{PDF_FILE}'")
    return chunks

def upsert_pinecone(chunks: list[dict]):
    index = pc.Index(POLICY_INDEX)
    total = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        resp  = openai_client.embeddings.create(model=EMBED_MODEL, input=[c["text"] for c in batch])
        index.upsert(
            vectors=[
                {"id": c["id"], "values": d.embedding, "metadata": c["metadata"]}
                for c, d in zip(batch, resp.data)
            ],
            namespace=NAMESPACE,
        )
        total += len(batch)
        print(f"[ingest_policies] Upserted {total}/{len(chunks)} chunks...")
    print(f"[ingest_policies] Done — {total} chunks in index='{POLICY_INDEX}' namespace='{NAMESPACE}'")

if __name__ == "__main__":
    print(f"[ingest_policies] Connecting to Pinecone index: '{POLICY_INDEX}'")
    upsert_pinecone(load_chunks())
