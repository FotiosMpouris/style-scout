import streamlit as st
import openai
import requests
import json
import re
import io
import time
from functools import wraps
from st_audiorec import st_audiorec

# Page configuration
st.set_page_config(
    page_title="Style Scout ✨",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Simplified and cleaner
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .main-title {
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #666;
        font-weight: 300;
        margin-bottom: 1rem;
    }
    
    .search-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .product-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        transition: transform 0.3s ease;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
    }
    
    .product-image {
        border-radius: 10px;
        width: 100%;
        height: 200px;
        object-fit: cover;
        margin-bottom: 1rem;
        background: #f0f0f0;
    }
    
    .voice-recording {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .vintage-indicator {
        background: linear-gradient(45deg, #8B4513, #D2691E);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem 0;
        display: inline-block;
    }
    
    .fashion-tip {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-left: 4px solid #ff6b6b;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
        font-style: italic;
        color: #555;
    }
</style>
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

def is_valid_product_url(url):
    """Check if URL is likely a product page"""
    if not url or url == "#":
        return False
    
    # Look for product indicators in URL
    product_indicators = [
        '/product/', '/item/', '/p/', '/products/', '/shop/',
        '/buy/', '/listing/', '/goods/', '/merchandise/'
    ]
    
    url_lower = url.lower()
    
    # Check for product indicators
    has_product_indicator = any(indicator in url_lower for indicator in product_indicators)
    
    # Avoid homepage and category pages
    avoid_patterns = [
        '/category/', '/collection/', '/search/', '/filter/',
        '/brand/', '/sale/', '/clearance/', '/about/', '/contact/'
    ]
    
    has_avoid_pattern = any(pattern in url_lower for pattern in avoid_patterns)
    
    # Check if URL ends with homepage patterns
    homepage_endings = ['/', '/home', '/index', '.com', '.net', '.org']
    is_homepage = any(url_lower.endswith(ending) for ending in homepage_endings)
    
    return has_product_indicator and not has_avoid_pattern and not is_homepage

def voice_recorder():
    """Fixed voice recording with st_audiorec"""
    st.markdown('<div class="voice-recording">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Voice Fashion Search")
    st.markdown("Click the record button and describe what you're looking for:")
    
    # Use st_audiorec for actual microphone recording
    audio_bytes = st_audiorec()
    
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("🎧 Transcribing your voice..."):
            try:
                # Create a BytesIO object for Whisper API
                audio_file = io.BytesIO(audio_bytes)
                audio_file.name = "recorded_audio.wav"
                
                # Transcribe with OpenAI Whisper
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
                
                st.session_state.transcribed_text = transcript.strip()
                st.success(f"✨ I heard: '{transcript.strip()}'")
                
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")
                st.info("Try speaking more clearly or use text search instead.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return st.session_state.transcribed_text

@retry_with_backoff(max_retries=3)
def search_fashion_items(refined_query, is_vintage):
    """Search with Perplexity using search_results format - FIXED"""
    headers = {
        "Authorization": f"Bearer {PPLX_KEY}",
        "Content-Type": "application/json"
    }
    
    # Improved search domains and prompts
    if is_vintage:
        search_domains = ["depop.com", "poshmark.com", "therealreal.com"]
        search_prompt = f"""Find specific vintage/secondhand fashion items for: {refined_query}

I need actual product listings with:
- Exact product titles and descriptions
- Current prices in USD
- Direct links to individual product pages (not category pages)
- Item condition and authenticity info
- Specific vintage/secondhand marketplace listings

Focus on real products currently for sale on secondhand platforms. Include product page URLs only."""
    else:
        search_domains = ["asos.com", "zara.com", "hm.com"]
        search_prompt = f"""Find specific fashion products for: {refined_query}

I need actual product listings with:
- Exact product names and descriptions  
- Current retail prices in USD
- Direct links to individual product pages (not category pages)
- Size availability and color options
- Specific items currently in stock

Focus on real products available for purchase. Include product page URLs only."""
    
    # FIXED: Use search_results instead of citations
    body = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [{"role": "user", "content": search_prompt}],
        "return_images": True,  # FIXED: Enable images
        "search_domain_filter": search_domains,
        "search_recency_filter": "month"  # Get recent listings
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=body,
        timeout=30
    )
    
    return response

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
        key="fashion_query"
    )
    
    # Show helpful tip
    st.markdown('<div class="fashion-tip">💡 <strong>Tip:</strong> Be specific about style, color, or occasion for better results!</div>', unsafe_allow_html=True)

elif st.session_state.search_mode == "voice":
    user_query = voice_recorder()

st.markdown('</div>', unsafe_allow_html=True)

# Main search functionality
if st.button("🔍 Find My Perfect Style!", disabled=not user_query.strip(), use_container_width=True):
    user_query = user_query.strip()
    
    # Detect vintage search
    is_vintage = is_vintage_search(user_query)
    
    if is_vintage:
        st.markdown('<div class="vintage-indicator">🌿 Sustainable Fashion Search</div>', unsafe_allow_html=True)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Refine query
        status_text.text("🧠 Analyzing your style preferences...")
        progress_bar.progress(25)
        
        refined_query = refine_search_query(user_query, is_vintage)
        
        # Step 2: Search
        status_text.text("🔍 Searching fashion platforms...")
        progress_bar.progress(50)
        
        response = search_fashion_items(refined_query, is_vintage)
        
        status_text.text("✨ Curating recommendations...")
        progress_bar.progress(75)
        
        if response and response.status_code == 200:
            data = response.json()
            
            progress_bar.progress(100)
            status_text.text("🎉 Your style curation is ready!")
            
            # Clear progress
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Display search info
            search_type = "Vintage/Secondhand" if is_vintage else "New Fashion"
            st.markdown(f"""
            <div class="search-container">
                <h3>🎯 {search_type} Search Results</h3>
                <p style="font-size: 1.1rem; color: #666;">Searched for: "{refined_query}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            # AI styling advice
            ai_response = data['choices'][0]['message']['content']
            st.markdown("## 💎 Your Personal Stylist Says:")
            st.markdown(f'<div class="search-container">{ai_response}</div>', unsafe_allow_html=True)
            
            # FIXED: Product display using search_results
            if 'search_results' in data and data['search_results']:
                shop_title = "🌿 Shop Sustainable Finds" if is_vintage else "🛍️ Shop These Picks"
                st.markdown(f"## {shop_title}")
                
                # Filter for valid product URLs
                valid_products = []
                for result in data['search_results']:
                    if is_valid_product_url(result.get('url', '')):
                        valid_products.append(result)
                
                if valid_products:
                    cols = st.columns(2)
                    
                    for i, product in enumerate(valid_products[:6]):  # Show max 6 products
                        col_idx = i % 2
                        
                        with cols[col_idx]:
                            st.markdown('<div class="product-card">', unsafe_allow_html=True)
                            
                            # FIXED: Display product image
                            if product.get('image_url'):
                                try:
                                    st.image(product['image_url'], use_column_width=True)
                                except:
                                    st.markdown('<div class="product-image" style="display: flex; align-items: center; justify-content: center; color: #999;">📸 Image unavailable</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="product-image" style="display: flex; align-items: center; justify-content: center; color: #999;">📸 No image</div>', unsafe_allow_html=True)
                            
                            # Product info
                            title = product.get('title', 'Fashion Item')
                            url = product.get('url', '#')
                            
                            st.markdown(f"**{title}**")
                            
                            # Extract domain for display
                            try:
                                domain = url.split('/')[2].replace('www.', '').title()
                                st.markdown(f"Shop at {domain}")
                            except:
                                st.markdown("Shop Now")
                            
                            # Shop button
                            if url and url != '#':
                                st.link_button("🛍️ Shop This Item", url, use_container_width=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No direct product links found. Try a more specific search term.")
            else:
                st.info("No products found. Try different search terms or check your internet connection.")
                
        else:
            error_code = response.status_code if response else "Network Error"
            st.error(f"Search temporarily unavailable (Error: {error_code}). Please try again!")
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Something went wrong: {str(e)}")
        st.info("Please try again or use different search terms.")

# Footer
st.markdown("""
---
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.8);">
    <p style="font-family: 'Playfair Display', serif; font-size: 1.1rem;">
        ✨ <strong>Style Scout</strong> - Where AI meets fashion discovery ✨
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        Powered by OpenAI & Perplexity
    </p>
</div>
""", unsafe_allow_html=True)
