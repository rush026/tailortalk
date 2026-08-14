# 🪡 TailorTalk — AI Saree Visual Similarity Search Agent

An AI-powered conversational agent that finds visually similar sarees from a fashion catalog using image embeddings, vector search, and natural-language chat.

> **Live Demo**: [🔗 Try TailorTalk](#) *(deploy URL here)*
>
> Upload any saree image and get the closest visual matches from the Byrappa Silks collection.

---

## 🎯 What It Does

TailorTalk is a chat-based AI assistant that:

1. **Accepts a saree image** — via file upload, URL paste, or drag-and-drop
2. **Analyzes visual features** — fabric texture, color palette, weave pattern, border work
3. **Searches a vector index** — finding the most visually similar sarees from 650+ catalogued products
4. **Presents results conversationally** — with match scores, pricing, availability, and direct product links

---

## 🏗 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│   Streamlit UI  │────▶│  LangGraph Agent │────▶│  Search Tool        │
│  (Chat + Upload)│     │  (GPT-4o-mini)   │     │  (find_similar)     │
└─────────────────┘     └──────────────────┘     └──────────┬──────────┘
                                                            │
                           ┌────────────────────────────────┤
                           ▼                                ▼
                    ┌──────────────┐              ┌─────────────────┐
                    │  Background  │              │  DINOv2 Embed   │
                    │  Removal     │              │  + HSV Histogram │
                    │  (rembg/u2net)│             └────────┬────────┘
                    └──────────────┘                       │
                                                  ┌───────┴────────┐
                                                  ▼                ▼
                                           ┌───────────┐   ┌────────────┐
                                           │ FAISS IP  │   │ Color Hist │
                                           │ Search    │   │ Re-ranking │
                                           └───────────┘   └────────────┘
```

---

## 🔧 Technical Stack

| Component | Choice | Rationale |
|---|---|---|
| **Embedding Model** | [DINOv2-base](https://huggingface.co/facebook/dinov2-base) (768-d) | Self-supervised vision model; captures fine-grained texture/pattern/weave features far better than CLIP (which is text-aligned and loses texture detail) |
| **Color Signal** | HSV histogram (8×12×3 bins = 288-d) | Explicit color-family matching; DINOv2 can match patterns but confuse color families — this fixes that |
| **Background Removal** | [rembg](https://github.com/danielgatis/rembg) (U²-Net) | Isolates the saree from mannequins, backgrounds, and flat-lay surfaces — critical because background noise dominates embeddings otherwise |
| **Vector Database** | [FAISS](https://github.com/facebookresearch/faiss) (IndexFlatIP) | In-memory inner-product search on pre-normalized vectors = cosine similarity; <1ms for 650 vectors; zero infrastructure |
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) + [LangChain](https://python.langchain.com/) | ReAct agent with `create_react_agent`; GPT-4o-mini does function-calling to invoke the search tool |
| **LLM** | OpenAI GPT-4o-mini | Cost-effective, excellent at function calling, provides natural conversational responses about fashion |
| **Frontend** | [Streamlit](https://streamlit.io/) | Rich chat UI with file upload, URL input, styled result cards with scores/pricing/links |

---

## 🔬 What I Did to Improve Search Quality

The core challenge: every image is a saree — differences are **fine-grained** (fabric, weave, color, border detail). A naive CLIP embedding search produces loose, generic matches. Here's what I did:

### 1. DINOv2 over CLIP
CLIP embeddings are optimized for text↔image alignment ("a red silk saree"), which smooths over visual texture. DINOv2 is a **self-supervised** vision model trained purely on image structure — it captures weave density, border motif repetition, fabric sheen, and drape pattern at the pixel level.

### 2. Background Removal (rembg / U²-Net)
The dataset mixes model shots (saree on a person), flat-lay shots (saree on a surface), and studio shots (mannequin). Without background removal, the embedding conflates the **background** with the saree — two completely different sarees on the same white mannequin score higher than two identical sarees in different settings. U²-Net foreground segmentation isolates the garment first.

### 3. Fused Scoring (α·DINOv2 + β·Color)
Two sarees can be DINOv2-similar in weave pattern but belong to entirely different color families (e.g., same Banarasi brocade in red vs. green). The fused score adds an explicit **HSV color histogram intersection** signal:

```
fused_score = 0.7 × DINOv2_cosine + 0.3 × HSV_intersection
```

The histogram uses 8 Hue × 12 Saturation × 3 Value bins, weighted toward Hue (color family) and Saturation (vibrancy) over Value (lighting).

### 4. Foreground-Only Color Histogram
The color histogram is computed **only on foreground pixels** (using the alpha mask from rembg), not on the entire image. This prevents white backgrounds, skin tones, and studio lighting from polluting the color signal.

### 5. Over-Retrieve + Re-rank
FAISS retrieves `4× top_k` candidates using DINOv2 cosine alone, then the fused scorer re-ranks them. This gives the color signal a chance to promote matches that FAISS alone might rank lower.

---

## 📁 Project Structure

```
tailortalk/
├── app.py                      # Streamlit chat frontend
├── src/
│   ├── agent.py                # LangGraph ReAct agent (GPT-4o-mini)
│   ├── tool.py                 # LangChain @tool — fused similarity search
│   ├── embed.py                # DINOv2 + HSV embedding pipeline
│   ├── build_index.py          # FAISS index builder (run once)
│   ├── build_metadata.py       # Product catalog JSON builder (run once)
│   └── download_images.py      # Image downloader from CSV URLs
├── data/
│   ├── byrappa_tejas_31july.csv # Product catalog CSV
│   ├── images/                  # Downloaded saree images (650+)
│   └── index/
│       ├── dino.index           # FAISS index (650 vectors, dim=768)
│       ├── color_hists.npy      # Color histograms (650×288)
│       ├── metadata.json        # Index ID → filename mapping
│       └── product_catalog.json # SKU → product details
├── .streamlit/config.toml       # Streamlit theme/server config
├── Dockerfile                   # HF Spaces / Docker deployment
├── requirements.txt
├── .env                         # OPENAI_API_KEY (not committed)
└── README.md
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- OpenAI API key

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/tailortalk.git
cd tailortalk

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# 5. Download images (if not already present)
python src/download_images.py data/byrappa_tejas_31july.csv

# 6. Build the index (if not already present)
python src/build_index.py
python src/build_metadata.py

# 7. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t tailortalk .

# Run
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... tailortalk
```

### Hugging Face Spaces
1. Create a new Space (Docker SDK)
2. Push this repo
3. Set `OPENAI_API_KEY` as a Space secret
4. The Dockerfile handles everything else

---

## ⚖️ Assumptions & Trade-offs

| Decision | Trade-off |
|---|---|
| **FAISS over Pinecone/Qdrant** | No infrastructure needed; 650 images fit easily in memory. For 100K+ images, switch to Qdrant with HNSW. |
| **DINOv2-base (86M params)** | Smaller than DINOv2-large/giant but good enough for 768-d fine-grained texture. Keeps inference fast on CPU. |
| **rembg adds ~2s per query** | Background removal is the bottleneck but critical for quality. Pre-computed for indexed images; only query images pay this cost at search time. |
| **GPT-4o-mini over GPT-4o** | Much cheaper for function-calling; the LLM only routes to the tool and formats results — it doesn't need frontier-level reasoning. |
| **Fused weights (0.7/0.3)** | Tuned by visual inspection on this dataset. For a different product category, these would need re-tuning. |
| **No text-based search** | This is a purely visual search tool. Text queries like "red Banarasi silk" aren't handled — the agent recognizes image inputs only. |

---

## 📜 License

MIT
