import streamlit as st
import openai
import requests
import json
import re
import io
import time
from functools import wraps
from audio_recorder_streamlit import audio_recorder

# Page configuration
st.set_page_config(
    page_title="Style Scout ✨",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- UPDATED CUSTOM CSS ---
# Updated with better contrast and cream boxes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Palette Variables */
    :root {
        --cream: #F4F0EC;           /* Light cream for boxes */
        --caramel: #C3895D;         /* Medium brown for accents */
        --coffee: #956737;          /* Medium-dark brown */
        --brownie: #5E3D23;         /* Dark brown for text */
        --very-dark: #2D1B0F;       /* Very dark brown for text on cream */
        --dark-bg: #3D2C1D;         /* Deep brown for background */
    }

    /* Main App Background - Keep Brown */
    .stApp {
        background-color: var(--dark-bg) !important;
    }

    /* Logo container - top left */
    .logo-container {
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 1000;
        width: 60px;
        height: 60px;
    }
    
    .logo-container img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }

    /* General text - Dark brown on light backgrounds */
    h1, h2, h3, h4, h5, h6 {
        color: var(--very-dark) !important;
    }
    
    p, li, .st-emotion-cache-16idsys p {
        color: var(--very-dark) !important;
    }
    
    /* Main Header Block - Cream Background */
    .main-header {
        background: var(--cream) !important;
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
        border: 2px solid var(--caramel);
    }
    
    /* Main Title: "Style Scout" - Dark Brown */
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        color: var(--very-dark) !important;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle text - Dark Brown */
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: var(--brownie) !important;
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    /* Containers - Cream Background */
    .search-container, .voice-recording {
        background: var(--cream) !important;
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid var(--caramel);
    }
    
    /* Product Card - Cream Background */
    .product-card {
        background: var(--cream) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 1px solid var(--caramel);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.3);
    }
    
    .product-card h1, .product-card h2, .product-card h3, 
    .product-card h4, .product-card h5, .product-card h6,
    .product-card p, .product-card div {
        color: var(--very-dark) !important;
    }
    
    .product-image {
        border-radius: 10px;
        width: 100%;
        height: 200px;
        object-fit: cover;
        margin-bottom: 1rem;
    }
    
    /* Vintage Indicator Badge */
    .vintage-indicator {
        background: linear-gradient(45deg, var(--coffee), var(--caramel));
        color: var(--cream) !important;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
        display: inline-block;
        border: 1px solid var(--caramel);
    }
    
    /* Fashion Tip Box */
    .fashion-tip {
        background: transparent;
        border-left: 4px solid var(--caramel);
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        font-style: italic;
        color: var(--very-dark) !important;
    }

    /* Shop buttons */
    .shop-button, .vintage-button {
        display: inline-block;
        width: 100%;
        padding: 0.75rem 1rem;
        margin-top: 1rem;
        border-radius: 8px;
        background-color: var(--caramel) !important;
        color: var(--cream) !important;
        font-weight: 600;
        text-align: center;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    .shop-button:hover, .vintage-button:hover {
        background-color: var(--coffee) !important;
        color: var(--cream) !important;
        text-decoration: none;
        transform: translateY(-2px);
    }
    
    /* Main Streamlit buttons */
    .stButton > button {
        background-color: var(--caramel) !important;
        color: var(--cream) !important;
        border: 1px solid var(--caramel) !important;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem;
    }
    .stButton > button:hover {
        background-color: var(--coffee) !important;
        color: var(--cream) !important;
        border: 1px solid var(--coffee) !important;
    }
    .stButton > button:disabled {
        background-color: rgba(149, 103, 55, 0.5) !important;
        color: rgba(240, 230, 208, 0.5) !important;
        border-color: transparent !important;
    }

    /* Text input - FIXED for readability */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: var(--very-dark) !important;
        border: 2px solid var(--caramel) !important;
        border-radius: 8px;
        font-size: 1rem !important;
        padding: 0.75rem !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: var(--coffee) !important;
        box-shadow: 0 0 10px rgba(195, 137, 93, 0.3) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: rgba(45, 27, 15, 0.6) !important;
    }
    
    /* Progress bar styling */
    .stProgress .st-bo {
        background-color: var(--caramel) !important;
    }
    
    /* Success/error messages */
    .stSuccess {
        background-color: rgba(195, 137, 93, 0.1) !important;
        border-left: 4px solid var(--caramel) !important;
        color: var(--very-dark) !important;
    }
    
    .stError {
        background-color: rgba(139, 69, 19, 0.1) !important;
        border-left: 4px solid var(--brownie) !important;
        color: var(--very-dark) !important;
    }
</style>
""", unsafe_allow_html=True)

# Add logo in top left corner
st.markdown("""
<div class="logo-container">
    <img src="style.png" alt="Style Scout Logo">
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "text"
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Style Scout</h1>
    <p class="subtitle">Your AI-powered personal fashion curator ✨</p>
</div>
""", unsafe_allow_html=True)

# Load API keys with better error handling
try:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    PPLX_KEY = st.secrets["PPLX_API_KEY"]
    openai.api_key = OPENAI_KEY
except Exception as e:
    st.error("🔑 Please set up your API keys in Streamlit Cloud secrets.")
    st.info("Required: OPENAI_API_KEY and PPLX_API_KEY")
    st.stop()

# Utility functions
def retry_with_backoff(max_retries=3):
    """Decorator for API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise e
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

@st.cache_data(ttl=3600)
def refine_search_query(user_query, is_vintage):
    """Refine user query with AI - cached to save tokens"""
    if is_vintage:
        system_prompt = """You are an expert vintage fashion curator. Convert the user's request into specific product search terms for secondhand platforms.

Focus on:
- Specific item types (blazer, dress, jeans, etc.)
- Brand names if mentioned
- Era/decade references
- Condition keywords
- Price range if mentioned

Examples:
- "something like Kurt Cobain wore" → "vintage 90s grunge flannel shirt leather jacket"
- "designer bag but affordable" → "pre-owned luxury handbag Coach Kate Spade"

Return only the refined search terms, under 12 words."""
    else:
        system_prompt = """You are an expert fashion stylist. Convert the user's request into specific product search terms for online retailers.

Focus on:
- Specific item types and categories
- Style descriptors (casual, formal, trendy)
- Occasions (work, date, party)
- Colors and materials if mentioned
- Current fashion trends

Examples:
- "something trendy for dates" → "date night dress midi bodycon"
- "professional but not boring" → "modern blazer women business casual"

Return only the refined search terms, under 12 words."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Query refinement failed: {e}")
        return user_query

def is_vintage_search(query):
    """Detect vintage/secondhand searches"""
    vintage_keywords = [
        'vintage', 'used', 'secondhand', 'second hand', 'pre-owned', 'thrift', 
        'consignment', 'pre-loved', 'previously owned', 'retro', 'preloved', 
        'gently used', 'resale', 'estate sale'
    ]
    return any(keyword in query.lower() for keyword in vintage_keywords)

def looks_like_product(url: str) -> bool:
    """Quick check to filter out obvious non-product pages"""
    bad_parts = ["blog", "category", "collections", "search?", "help"]
    return all(bp not in url.lower() for bp in bad_parts)

def voice_recorder():
    """Record from mic, transcribe with Whisper, return text."""
    st.markdown("### 🎙️ Voice Fashion Search")
    
    # Voice recorder with updated colors
    audio_bytes = audio_recorder(
        text="🎤",
        recording_color="#956737",  # Coffee
        neutral_color="#C3895D",    # Caramel
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0,
        key="voice_recorder"
    )
    
    if audio_bytes:
        with st.spinner("🎧 Transcribing your voice..."):
            try:
                import tempfile
                import os
                
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp.write(audio_bytes)
                tmp.flush()
                
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=open(tmp.name, "rb"),
                    response_format="text"
                )
                
                tmp.close()
                os.unlink(tmp.name)
                
                st.session_state.transcribed_text = transcript.strip()
                st.success(f"✨ I heard: '{transcript.strip()}'")
                
            except Exception as e:
                st.error(f"Transcription error: {e}")
                st.session_state.transcribed_text = ""
    
    return st.session_state.transcribed_text

@retry_with_backoff(max_retries=3)
def search_fashion_items(refined_query, is_vintage):
    """Query Perplexity, return JSON search_results list."""
    headers = {
        "Authorization": f"Bearer {PPLX_KEY}",
        "Content-Type": "application/json",
    }

    domains = (
        ["depop.com", "poshmark.com", "therealreal.com"]
        if is_vintage
        else ["zara.com", "nordstrom.com", "asos.com"]
    )

    search_prompt = (
        f"""Find **buyable** vintage items for: {refined_query}

Return 8 specific product pages with:
- title
- direct product URL
- price (if available)
- single image URL
"""
        if is_vintage
        else f"""Find **buyable** new fashion items for: {refined_query}

Return 8 specific product pages with:
- title
- direct product URL
- price (if available)
- single image URL
"""
    )

    body = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [{"role": "user", "content": search_prompt}],
        "search_domain_filter": domains,
        "search_results": True,
        "return_images": True,
    }

    r = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=body,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()

# Main interface
st.markdown('<div class="search-container">', unsafe_allow_html=True)

# Mode selection
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("💬 Text Search", use_container_width=True):
        st.session_state.search_mode = "text"

with col2:
    if st.button("🎙️ Voice Search", use_container_width=True):
        st.session_state.search_mode = "voice"

# Input handling
user_query = ""

if st.session_state.search_mode == "text":
    st.markdown("### ✍️ Describe Your Perfect Look")
    user_query = st.text_input(
        "",
        placeholder="Try: 'vintage leather jacket' or 'black midi dress for work'",
        key="fashion_query",
        label_visibility="collapsed"
    )
    
    st.markdown('<div class="fashion-tip">💡 <strong>Tip:</strong> Be specific about style, color, or occasion for better results!</div>', unsafe_allow_html=True)

elif st.session_state.search_mode == "voice":
    st.markdown('<div class="voice-recording">', unsafe_allow_html=True)
    user_query = voice_recorder()
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Main search functionality
if st.button("🔍 Find My Perfect Style!", disabled=not user_query.strip(), use_container_width=True):
    user_query = user_query.strip()
    
    is_vintage = is_vintage_search(user_query)
    
    if is_vintage:
        st.markdown('<div class="vintage-indicator">🌿 Sustainable Fashion Search</div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🧠 Analyzing your style preferences...")
        progress_bar.progress(25)
        
        refined_query = refine_search_query(user_query, is_vintage)
        
        status_text.text("🔍 Searching fashion platforms...")
        progress_bar.progress(50)
        
        response = search_fashion_items(refined_query, is_vintage)
        
        status_text.text("✨ Curating recommendations...")
        progress_bar.progress(75)
        
        data = response
        
        progress_bar.progress(100)
        status_text.text("🎉 Your style curation is ready!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        search_type = "Vintage/Secondhand" if is_vintage else "New Fashion"
        st.markdown(f"""
        <div class="search-container">
            <h3>🎯 {search_type} Search Results</h3>
            <p style="font-size: 1.1rem;">Searched for: "{refined_query}"</p>
        </div>
        """, unsafe_allow_html=True)
        
        ai_response = data['choices'][0]['message']['content']
        st.markdown("## 💎 Your Personal Stylist Says:")
        st.markdown(f'<div class="search-container">{ai_response}</div>', unsafe_allow_html=True)
            
        if data.get("search_results"):
            shop_title = "🌿 Shop Sustainable Fashion" if is_vintage else "🛍️ Shop These Curated Picks"
            st.markdown(f"## {shop_title}")

            images = data.get("images", [])

            cols = st.columns(2)
            for i, prod in enumerate([p for p in data["search_results"] if looks_like_product(p.get("url",""))][:8]):
                if not prod.get("url") or not prod.get("title"):
                    continue

                with cols[i % 2]:
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)

                    image_url = None
                    if i < len(images) and images[i] is not None:
                        img_data = images[i]
                        if isinstance(img_data, dict) and 'image_url' in img_data:
                            image_url = img_data['image_url']
                        elif isinstance(img_data, str):
                            image_url = img_data
                    
                    if image_url:
                        try:
                            st.image(image_url, use_column_width=True)
                        except:
                            st.markdown('<div style="height: 200px; background: #f0f0f0; display: flex; align-items: center; justify-content: center; border-radius: 10px; color: #999;">📸 Image unavailable</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="height: 200px; background: #f0f0f0; display: flex; align-items: center; justify-content: center; border-radius: 10px; color: #999;">📸 No image</div>', unsafe_allow_html=True)

                    st.markdown(f"**{prod['title']}**")
                    if prod.get("price"):
                        st.markdown(f"💰 {prod['price']}")

                    button_class = "vintage-button" if is_vintage else "shop-button"
                    emoji = "♻️" if is_vintage else "✨"
                    st.markdown(
                        f'<a href="{prod["url"]}" target="_blank" class="{button_class}">{emoji} Shop Now</a>',
                        unsafe_allow_html=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)
                
        else:
            st.error("Search temporarily unavailable. Please try again!")
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Something went wrong: {str(e)}")
        st.info("Please try again or use different search terms.")

# Footer
st.markdown("""
---
<div style="text-align: center; padding: 2rem; color: var(--cream); opacity: 0.7;">
    <p style="font-family: 'Playfair Display', serif; font-size: 1.1rem;">
        ✨ <strong>Style Scout</strong> - Where AI meets fashion discovery ✨
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        Powered by OpenAI & Perplexity
    </p>
</div>
""", unsafe_allow_html=True)
