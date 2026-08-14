"""
TailorTalk — AI Saree Visual Similarity Search
Streamlit chat interface with premium dark-themed UI.
"""

import os
import sys
import uuid
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent import get_agent
from src.tool import search_similar, _PROJECT_ROOT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TailorTalk — Saree Style Assistant",
    page_icon="🪡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark theme with gradient accents
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Global overrides */
.stApp {
    font-family: 'Outfit', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e8d5b7 !important;
    font-family: 'Outfit', sans-serif;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #b8b8cc !important;
    font-family: 'Outfit', sans-serif;
}

/* Chat message styling */
.stChatMessage {
    border-radius: 16px !important;
    margin-bottom: 12px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Result card styling */
.saree-card {
    background: linear-gradient(135deg, rgba(30,30,50,0.9) 0%, rgba(25,25,45,0.95) 100%);
    border: 1px solid rgba(232,213,183,0.15);
    border-radius: 16px;
    padding: 16px;
    margin: 8px 0;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
}

.saree-card:hover {
    border-color: rgba(232,213,183,0.4);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(232,213,183,0.1);
}

.saree-card .card-title {
    color: #e8d5b7;
    font-size: 1.05em;
    font-weight: 600;
    margin-bottom: 6px;
    font-family: 'Outfit', sans-serif;
    line-height: 1.3;
}

.saree-card .card-sku {
    color: #8888aa;
    font-size: 0.82em;
    font-family: 'Outfit', sans-serif;
}

.saree-card .card-price {
    color: #7cdb8a;
    font-size: 1.1em;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
}

.saree-card .card-mrp {
    color: #888;
    text-decoration: line-through;
    font-size: 0.88em;
    margin-left: 6px;
    font-family: 'Outfit', sans-serif;
}

.saree-card .card-stock {
    font-size: 0.82em;
    padding: 2px 10px;
    border-radius: 12px;
    display: inline-block;
    font-family: 'Outfit', sans-serif;
    font-weight: 500;
}

.saree-card .in-stock {
    background: rgba(124,219,138,0.15);
    color: #7cdb8a;
}

.saree-card .out-of-stock {
    background: rgba(219,124,124,0.15);
    color: #db7c7c;
}

/* Score bar */
.score-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 8px;
    margin: 8px 0;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #c9a96e 0%, #e8d5b7 100%);
    transition: width 0.6s ease;
}

.score-label {
    color: #c9a96e;
    font-size: 0.88em;
    font-weight: 500;
    font-family: 'Outfit', sans-serif;
}

/* Header banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    border: 1px solid rgba(232,213,183,0.12);
    border-radius: 20px;
    padding: 32px 28px;
    margin-bottom: 24px;
    text-align: center;
}

.hero-banner h1 {
    background: linear-gradient(135deg, #e8d5b7, #c9a96e, #e8d5b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2em;
    font-weight: 700;
    margin-bottom: 6px;
    font-family: 'Outfit', sans-serif;
}

.hero-banner p {
    color: #8888aa;
    font-size: 1.05em;
    font-family: 'Outfit', sans-serif;
}

/* Upload area styling */
.upload-area {
    background: linear-gradient(135deg, rgba(30,30,50,0.6) 0%, rgba(25,25,45,0.7) 100%);
    border: 2px dashed rgba(232,213,183,0.25);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: rgba(232,213,183,0.5);
}

/* Link button */
.view-link {
    display: inline-block;
    background: linear-gradient(135deg, #c9a96e, #e8d5b7);
    color: #1a1a2e !important;
    padding: 6px 16px;
    border-radius: 8px;
    text-decoration: none !important;
    font-size: 0.85em;
    font-weight: 600;
    font-family: 'Outfit', sans-serif;
    transition: all 0.3s ease;
    margin-top: 8px;
}

.view-link:hover {
    opacity: 0.85;
    transform: scale(1.02);
}

/* Divider */
.styled-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232,213,183,0.3), transparent);
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "uploaded_image_path" not in st.session_state:
    st.session_state.uploaded_image_path = None
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 16px 0;">
        <h1 style="font-size: 1.8em; margin-bottom: 4px;">🪡 TailorTalk</h1>
        <p style="font-size: 0.95em; opacity: 0.7;">Saree Style Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Image upload
    st.markdown("### 📸 Upload a Saree Image")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg", "jpeg", "png", "webp"],
        key="file_uploader",
        label_visibility="collapsed",
    )

    st.markdown("##### — OR —")

    # URL input
    image_url = st.text_input(
        "🔗 Paste an image URL",
        placeholder="https://example.com/saree.jpg",
        key="url_input",
    )

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Search Settings")
    top_k = st.slider("Number of results", min_value=3, max_value=10, value=5, key="top_k")

    st.markdown("---")

    # New session button
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.uploaded_image_path = None
        st.session_state.search_results = []
        st.rerun()

    st.markdown("---")

    st.markdown("""
    <div style="text-align:center; opacity:0.5; font-size:0.8em; padding:12px;">
        Powered by DINOv2 + FAISS<br>
        Built for Byrappa Silks
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: render results as cards
# ---------------------------------------------------------------------------
def render_result_cards(results: list[dict]):
    """Render search results as premium styled cards."""
    cols_per_row = 3
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(results):
                break
            r = results[idx]
            with col:
                # Image
                image_path = r.get("local_image_path", "")
                if image_path and Path(image_path).exists():
                    st.image(str(image_path), use_container_width=True)
                elif r.get("image_url"):
                    st.image(r["image_url"], use_container_width=True)

                # Score bar
                score_pct = min(r["similarity_score"] * 100, 100)
                score_display = f"{score_pct:.1f}%"

                # Price
                price_html = f"₹{r['discounted_price']}"
                if r["retail_price"] != r["discounted_price"] and r["retail_price"] != "N/A":
                    price_html = f"₹{r['discounted_price']} <span class='card-mrp'>₹{r['retail_price']}</span>"

                # Stock
                stock_class = "in-stock" if r.get("in_stock") else "out-of-stock"
                stock_text = "✓ In Stock" if r.get("in_stock") else "✗ Out of Stock"

                # Link
                link_html = ""
                if r.get("website_link"):
                    link_html = f'<a href="{r["website_link"]}" target="_blank" class="view-link">View Product →</a>'

                st.markdown(f"""
                <div class="saree-card">
                    <div class="card-title">{r.get('name', 'Saree')}</div>
                    <div class="card-sku">SKU: {r.get('sku', 'N/A')} · #{r.get('rank', idx+1)}</div>
                    <hr class="styled-divider">
                    <div class="score-label">Match: {score_display}</div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width: {score_pct}%"></div>
                    </div>
                    <div style="margin-top: 10px;">
                        <span class="card-price">{price_html}</span>
                    </div>
                    <div style="margin-top: 6px;">
                        <span class="card-stock {stock_class}">{stock_text}</span>
                    </div>
                    <div style="margin-top: 10px;">
                        {link_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
# Hero banner
if not st.session_state.messages:
    st.markdown("""
    <div class="hero-banner">
        <h1>🪡 TailorTalk</h1>
        <p>Upload a saree image to discover visually similar designs from our collection</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick start tips
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="saree-card" style="text-align:center;">
            <div style="font-size:2em; margin-bottom:8px;">📸</div>
            <div class="card-title">Upload an Image</div>
            <div class="card-sku">JPG, PNG, or WEBP</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="saree-card" style="text-align:center;">
            <div style="font-size:2em; margin-bottom:8px;">🔍</div>
            <div class="card-title">AI-Powered Search</div>
            <div class="card-sku">DINOv2 visual matching</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="saree-card" style="text-align:center;">
            <div style="font-size:2em; margin-bottom:8px;">✨</div>
            <div class="card-title">Discover Matches</div>
            <div class="card-sku">Color, pattern & fabric</div>
        </div>
        """, unsafe_allow_html=True)


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # If this message had search results, re-render the cards
        if msg.get("results"):
            render_result_cards(msg["results"])


# ---------------------------------------------------------------------------
# Process image inputs
# ---------------------------------------------------------------------------
def process_image_search(image_source: str, display_image=None):
    """Run the search and display results through the agent."""
    # Add user message
    user_msg = "Find sarees similar to this image."
    st.session_state.messages.append({"role": "user", "content": user_msg})

    with st.chat_message("user"):
        if display_image is not None:
            st.image(display_image, width=250)
        st.markdown(user_msg)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing image and searching catalog..."):
            try:
                # Get the agent
                agent = get_agent()

                # Direct search for results to display as cards
                results = search_similar(image_source, top_k=top_k)
                st.session_state.search_results = results

                # Also get agent's conversational response
                agent_input = {
                    "messages": [
                        {"role": "user", "content": f"I've uploaded a saree image. Please find similar sarees. The image is at: {image_source}"}
                    ]
                }
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                response = agent.invoke(agent_input, config=config)

                # Extract AI response
                ai_messages = [m for m in response["messages"] if m.type == "ai" and m.content]
                ai_text = ai_messages[-1].content if ai_messages else "Here are the most similar sarees from our collection:"

                st.markdown(ai_text)

                # Render result cards
                if results:
                    render_result_cards(results)

                # Save to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_text,
                    "results": results,
                })

            except Exception as e:
                error_msg = f"I encountered an error while processing: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


# Handle file upload
if uploaded_file is not None:
    # Save to temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Check if we already processed this file
    if st.session_state.uploaded_image_path != tmp_path:
        st.session_state.uploaded_image_path = tmp_path
        display_img = Image.open(uploaded_file)
        process_image_search(tmp_path, display_image=display_img)

# Handle URL input
if image_url and image_url.startswith("http"):
    # Use a session key to track processed URLs
    url_key = f"processed_url_{hash(image_url)}"
    if url_key not in st.session_state:
        st.session_state[url_key] = True
        process_image_search(image_url)

# ---------------------------------------------------------------------------
# Chat input for text messages
# ---------------------------------------------------------------------------
if prompt := st.chat_input("Ask me about sarees or paste an image URL..."):
    # Check if it looks like a URL
    if prompt.strip().startswith(("http://", "https://")) and any(
        ext in prompt.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]
    ):
        process_image_search(prompt.strip())
    else:
        # Regular text chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    agent = get_agent()
                    agent_input = {
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    config = {"configurable": {"thread_id": st.session_state.thread_id}}
                    response = agent.invoke(agent_input, config=config)

                    ai_messages = [m for m in response["messages"] if m.type == "ai" and m.content]
                    ai_text = ai_messages[-1].content if ai_messages else "I'm not sure how to respond to that."

                    st.markdown(ai_text)
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
