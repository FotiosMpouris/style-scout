import streamlit as st
import openai
import requests
import json
import re
import io
import time
from functools import wraps
from audio_recorder_streamlit import audio_recorder
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Style Scout ✨",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- HELPER FUNCTION TO DISPLAY LOGO ---
# This function reads a local image and encodes it to base64,
# which is the correct way to display it in st.markdown.
def get_image_as_base64(path_str):
    """Reads an image file and returns its base64 encoded version."""
    path = Path(path_str)
    if not path.is_file():
        # If the logo file is not found, return None
        return None
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- NEW & IMPROVED CUSTOM CSS ---
st.markdown("""
<style>
    /* Import new, more stylish fonts */
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;700&family=Lato:wght@300;400;700&display=swap');
    
    /* Define the color palette */
    :root {
        --cream: #F4F0EC;
        --caramel: #C3895D;
        --coffee: #956737;
        --brownie: #5E3D23;
        --dark-text: #3D2C1D; /* Dark brown for text on light backgrounds */
        --dark-bg-start: #3D2C1D;
        --dark-bg-end: #291a10;
    }

    /* Main App Background with a subtle gradient */
    .stApp {
        background-image: linear-gradient(170deg, var(--dark-bg-start) 0%, var(--dark-bg-end) 100%);
        color: var(--cream); /* Default text color for the dark background */
    }

    /* Logo container - bigger and positioned top left */
    .logo-container {
        position: fixed;
        top: 20px;
        left: 30px;
        z-index: 1000;
        width: 100px; /* Increased size */
        height: 100px; /* Increased size */
        transition: transform 0.3s ease;
    }
    
    .logo-container:hover {
        transform: scale(1.05);
    }
    
    .logo-container img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
    }

    /* Set default fonts for the entire app */
    body, .st-emotion-cache-16idsys p {
        font-family: 'Lato', sans-serif;
        color: var(--dark-text); /* Default to dark text for content boxes */
    }
    
    /* Main Content Containers - Cream background for high contrast */
    .main-header, .search-container, .voice-recording, .product-card {
        background: var(--cream);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem auto; /* Center the main blocks */
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        border: 1px solid var(--caramel);
        color: var(--dark-text);
        max-width: 950px; /* Constrain width for better readability */
    }

    /* Specific styles for the product grid to allow it to be wider */
    .st-emotion-cache-ocqkz7 {
        max-width: 1200px !important;
        margin: auto !important;
    }

    .product-card {
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 50px rgba(0,0,0,0.35);
    }

    /* Headings with the new serif font */
    h1, h2, h3 {
        font-family: 'Cormorant Garamond', serif;
        font-weight: 700;
        color: var(--dark-text);
    }
    
    /* Main Title - larger and more prominent */
    .main-title {
        font-size: 3.5rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'Lato', sans-serif;
        font-size: 1.2rem;
        color: var(--brownie);
        font-weight: 400;
    }

    .product-image {
        border-radius: 15px;
        width: 100%;
        height: 250px;
        object-fit: cover;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Vintage Indicator Badge - for dark text readability */
    .vintage-indicator {
        background: var(--caramel);
        color: var(--cream) !important;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-size: 1rem;
        font-weight: 600;
        margin: 1rem auto;
        display: inline-block;
        border: 1px solid var(--coffee);
    }
    
    /* Fashion Tip Box - more distinct */
    .fashion-tip {
        background: rgba(216, 203, 191, 0.3);
        border-left: 4px solid var(--caramel);
        padding: 1rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 1.5rem 0;
        color: var(--dark-text);
    }

    /* Buttons - unified and stylish */
    .shop-button, .vintage-button {
        display: inline-block;
        width: 100%;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        border-radius: 12px;
        background-color: var(--caramel);
        color: var(--cream) !important;
        font-weight: 700;
        font-family: 'Lato', sans-serif;
        text-align: center;
        text-decoration: none;
        transition: all 0.3s ease;
    }

    .shop-button:hover, .vintage-button:hover {
        background-color: var(--coffee);
        transform: translateY(-3px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        text-decoration: none;
    }
    
    .stButton > button {
        background-color: var(--caramel);
        color: var(--cream);
        border: 1px solid transparent;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.8rem 1.5rem;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: var(--coffee);
        transform: translateY(-2px);
    }
    .stButton > button:disabled {
        background-color: #958475;
        color: rgba(244, 240, 236, 0.6);
        border-color: transparent;
    }

    /* Text input - clean and clear */
    .stTextInput > div > div > input {
        background-color: #FFFFFF;
        color: var(--dark-text);
        border: 2px solid #D8CBBF;
        border-radius: 12px;
        font-size: 1.1rem;
        padding: 0.8rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: var(--caramel);
        box-shadow: 0 0 0 3px rgba(195, 137, 93, 0.3);
    }

    /* Footer text on dark background */
    .footer-text {
        color: var(--cream) !important;
        opacity: 0.7;
    }
</style>
""", unsafe_allow_html=True)


# --- RENDER THE LOGO ---
# Get the base64 string for the logo
logo_base64 = get_image_as_base64("style.png")
# Create the HTML for the logo, only if the logo file was found
if logo_base64:
    st.markdown(
        f'<div class="logo-container"><img src="data:image/png;base64,{logo_base64}" alt="Style Scout Logo"></div>',
        unsafe_allow_html=True
    )

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

# Utility functions (NO CHANGES HERE)
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
        recording_color="#956737",
        neutral_color="#C3895D",
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
        st.markdown("<h2>💎 Your Personal Stylist Says:</h2>", unsafe_allow_html=True)
        st.markdown(f'<div class="search-container">{ai_response}</div>', unsafe_allow_html=True)
            
        if data.get("search_results"):
            shop_title = "🌿 Shop Sustainable Fashion" if is_vintage else "🛍️ Shop These Curated Picks"
            st.markdown(f"<h2 style='text-align: center; color: var(--cream);'>{shop_title}</h2>", unsafe_allow_html=True)

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
                        st.markdown(f'<img src="{image_url}" class="product-image">', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="product-image" style="display: flex; align-items: center; justify-content: center; background: #e0e0e0; color: #999;">📸 No image</div>', unsafe_allow_html=True)

                    st.markdown(f"<h3>{prod['title']}</h3>", unsafe_allow_html=True)
                    if prod.get("price"):
                        st.markdown(f"<p><strong>💰 {prod['price']}</strong></p>", unsafe_allow_html=True)

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
<div style="text-align: center; padding: 3rem;">
    <p class="footer-text" style="font-family: 'Cormorant Garamond', serif; font-size: 1.2rem;">
        ✨ <strong>Style Scout</strong> - Where AI meets fashion discovery ✨
    </p>
    <p class="footer-text" style="font-size: 0.9rem; margin-top: 0.5rem;">
        Powered by OpenAI & Perplexity
    </p>
</div>
""", unsafe_allow_html=True)
