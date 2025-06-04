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

# Custom CSS - Enhanced and optimized
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
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
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
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        overflow-wrap: break-word;
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .product-image {
        border-radius: 10px;
        width: 100%;
        height: 200px;
        object-fit: cover;
        margin-bottom: 1rem;
    }
    
    .shop-button {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .shop-button:hover {
        background: linear-gradient(45deg, #ff5252, #ff7979);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(255,107,107,0.4);
        color: white;
        text-decoration: none;
    }
    
    .vintage-button {
        background: linear-gradient(45deg, #8B4513, #D2691E);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 25px;
        font-weight: 500;
        text-decoration: none;
        display: inline-block;
        transition: all 0.3s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .vintage-button:hover {
        background: linear-gradient(45deg, #A0522D, #CD853F);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(139,69,19,0.4);
        color: white;
        text-decoration: none;
    }
    
    .voice-recording {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
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
    
    .stTextInput > div > div > input {
        border-radius: 15px;
        border: 2px solid rgba(255,255,255,0.3);
        padding: 1rem;
        font-size: 1.1rem;
        background: rgba(255,255,255,0.8);
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #ff6b6b;
        box-shadow: 0 0 20px rgba(255,107,107,0.3);
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #5a6fd8, #6a4190);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.4);
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
        overflow-wrap: break-word;
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

@st.cache_data(ttl=3600)  # Cache for 1 hour to save OpenAI tokens
def refine_search_query(user_query, is_vintage):
    """Refine user query with AI - cached to save tokens"""
    if is_vintage:
        system_prompt = """You are an expert vintage fashion curator. Convert the user's request into search terms for secondhand platforms like Depop, Poshmark, eBay, TheRealReal.

Consider vintage eras, designer brands, condition, and sustainable fashion terminology.

Examples:
- "something like Kurt Cobain wore" → "vintage 90s grunge flannel band tee leather jacket"
- "designer bag but affordable" → "pre-owned luxury handbag Kate Spade Coach vintage"

Keep under 15 words."""
    else:
        system_prompt = """You are an expert fashion stylist. Convert the user's fashion request into effective search terms for modern retailers.

Consider current trends, price points, styling, and specific product details.

Examples:
- "something trendy for dates" → "date night outfit women trendy 2024 midi dress"
- "professional but not boring" → "modern business casual women blazer contemporary"

Keep under 15 words."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.4,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Query refinement failed: {e}")
        return user_query  # Fallback to original

def is_vintage_search(query):
    """Detect vintage/secondhand searches"""
    vintage_keywords = [
        'vintage', 'used', 'secondhand', 'second hand', 'pre-owned', 'thrift', 
        'consignment', 'pre-loved', 'previously owned', 'estate', 'antique',
        'retro', 'preloved', 'gently used', 'resale'
    ]
    return any(keyword in query.lower() for keyword in vintage_keywords)

def voice_recorder():
    """Real voice recording with st_audiorec"""
    st.markdown('<div class="voice-recording">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Voice Fashion Search")
    st.markdown("Click the microphone to start recording:")
    
    # Real microphone recording
    audio_bytes = st_audiorec()
    
    if audio_bytes is not None:
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("🎧 Transcribing your voice..."):
            try:
                # Prepare for Whisper
                wav_io = io.BytesIO(audio_bytes)
                wav_io.name = "speech.wav"
                wav_io.seek(0)  # Important: rewind the buffer
                
                # Transcribe with OpenAI Whisper
                transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=wav_io,
                    response_format="text"
                )
                
                st.session_state.transcribed_text = transcript.strip()
                st.success(f"✨ I heard: '{transcript}'")
                
            except Exception as e:
                st.error(f"Transcription error: {str(e)}")
                st.info("Try recording again or use text search.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    return st.session_state.transcribed_text

@retry_with_backoff(max_retries=3)
def search_fashion_items(refined_query, is_vintage):
    """Search with Perplexity using new search_results format"""
    headers = {
        "Authorization": f"Bearer {PPLX_KEY}",
        "Content-Type": "application/json"
    }
    
    # Updated domains - max 3 per API limitation
    if is_vintage:
        search_domains = ["depop.com", "poshmark.com", "therealreal.com"]
        search_prompt = f"""Find specific vintage/secondhand clothing for: {refined_query}

Provide detailed product recommendations with:
1. Item descriptions and estimated prices
2. Authenticity and condition tips
3. Direct shopping links from vintage platforms
4. Era identification and styling advice

Focus on sustainable fashion from secondhand marketplaces."""
    else:
        search_domains = ["zara.com", "nordstrom.com", "asos.com"]
        search_prompt = f"""Find specific clothing items for: {refined_query}

Provide detailed recommendations with:
1. Product names, descriptions, and prices
2. Styling suggestions and outfit ideas
3. Direct shopping links from retailers
4. Size and fit information

Focus on current fashion from trusted retailers."""
    
    body = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [{"role": "user", "content": search_prompt}],
        "return_images": True,  # Enable images
        "search_domain_filter": search_domains
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=body,
        timeout=30
    )
    
    return response

# Fashion tips
fashion_tips = [
    "💡 **Pro Tip**: Mention specific designers, colors, or occasions for better results!",
    "✨ **Style Hack**: Try describing the vibe - 'coastal grandmother', 'dark academia', 'cottagecore'",
    "🎯 **Search Smarter**: Include your budget range for more relevant results",
    "👑 **Celebrity Inspo**: Reference your style icons - 'like Zendaya's red carpet look'",
    "🌟 **Occasion Matters**: Specify if it's for work, dates, casual, or special events",
    "♻️ **Sustainable Style**: Try 'vintage', 'secondhand', or 'preloved' for eco-friendly finds!"
]

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
        placeholder="✨ Try: 'vintage leather jacket like Kurt Cobain' or 'cottagecore dress for spring'",
        key="fashion_query"
    )
    
    # Show random fashion tip
    import random
    st.markdown(f'<div class="fashion-tip">{random.choice(fashion_tips)}</div>', unsafe_allow_html=True)

elif st.session_state.search_mode == "voice":
    user_query = voice_recorder()

st.markdown('</div>', unsafe_allow_html=True)

# Input validation
if not user_query.strip() and st.button("🔍 Find My Perfect Style!", use_container_width=True):
    st.warning("✨ Tell me what you're looking for!")
    st.stop()

# Main search functionality
if st.button("🔍 Find My Perfect Style!", disabled=not user_query.strip(), use_container_width=True):
    user_query = user_query.strip()
    
    # Detect vintage search
    is_vintage = is_vintage_search(user_query)
    
    if is_vintage:
        st.markdown('<div class="vintage-indicator">🌿 Sustainable Fashion Search Detected</div>', unsafe_allow_html=True)
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Refine query
        status_text.text("🧠 AI is analyzing your style preferences...")
        progress_bar.progress(25)
        
        refined_query = refine_search_query(user_query, is_vintage)
        
        # Step 2: Search
        status_text.text("🔍 Searching fashion platforms...")
        progress_bar.progress(50)
        
        response = search_fashion_items(refined_query, is_vintage)
        
        status_text.text("✨ Curating your personalized recommendations...")
        progress_bar.progress(75)
        
        if response and response.status_code == 200:
            data = response.json()
            
            progress_bar.progress(100)
            status_text.text("🎉 Your style curation is ready!")
            
            # Clear progress
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            search_type = "Vintage/Secondhand" if is_vintage else "New Fashion"
            st.markdown(f"""
            <div class="search-container">
                <h3>🎯 Your {search_type} Search</h3>
                <p style="font-size: 1.1rem; color: #666; font-style: italic;">"{refined_query}"</p>
            </div>
            """, unsafe_allow_html=True)
            
            # AI styling advice
            ai_response = data['choices'][0]['message']['content']
            advisor_title = "💎 Your Vintage Curator Says:" if is_vintage else "💎 Your Personal Stylist Says:"
            st.markdown(f"## {advisor_title}")
            st.markdown(f'<div class="search-container">{ai_response}</div>', unsafe_allow_html=True)
            
            # Product grid - Using new search_results format
            if 'search_results' in data and data['search_results']:
                shop_title = "🌿 Shop Sustainable Fashion" if is_vintage else "🛍️ Shop These Curated Picks"
                st.markdown(f"## {shop_title}")
                
                cols = st.columns(2)
                
                for i, result in enumerate(data['search_results'][:8]):
                    col_idx = i % 2
                    
                    with cols[col_idx]:
                        # Extract info from search result
                        title = result.get('title', 'Fashion Item')
                        url = result.get('url', '#')
                        img_url = result.get('image_url')
                        
                        # Clean URL for display
                        domain = url.split('/')[2].replace('www.', '').title() if url != '#' else 'Shop'
                        
                        # Button styling
                        button_class = "vintage-button" if is_vintage else "shop-button"
                        button_emoji = "♻️" if is_vintage else "✨"
                        
                        st.markdown('<div class="product-card">', unsafe_allow_html=True)
                        
                        # Show image if available
                        if img_url:
                            try:
                                st.image(img_url, use_column_width=True)
                            except:
                                pass  # Skip broken images
                        
                        # Product info
                        st.markdown(f"**{title}**")
                        st.markdown(f"Shop at {domain}")
                        st.markdown(
                            f'<a href="{url}" target="_blank" class="{button_class}">{button_emoji} Shop Now</a>',
                            unsafe_allow_html=True
                        )
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Fashion advice
            if is_vintage:
                st.markdown("""
                ## 🌿 Sustainable Shopping Tips
                <div class="fashion-tip">
                    <strong>Vintage Curator's Note:</strong> Always check measurements rather than size labels for vintage pieces. Ask about condition, authenticity for designer items, and return policies. Vintage sizing typically runs smaller than modern!
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                ## 💫 Styling Tips
                <div class="fashion-tip">
                    <strong>Stylist's Note:</strong> Check size guides carefully and mix high/low pieces. The best outfits combine investment pieces with trendy, affordable finds!
                </div>
                """, unsafe_allow_html=True)
                
        else:
            st.error(f"Search temporarily unavailable (Error: {response.status_code if response else 'Network'}). Please try again!")
            
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Something went wrong: {str(e)}")
        st.info("Please try again or contact support if the issue persists.")

# Footer
st.markdown("""
---
<div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.8);">
    <p style="font-family: 'Playfair Display', serif; font-size: 1.1rem;">
        ✨ <strong>Style Scout</strong> - Where AI meets haute couture ✨
    </p>
    <p style="font-size: 0.9rem; margin-top: 0.5rem;">
        Powered by OpenAI & Perplexity | Made with 💖 for fashion lovers
    </p>
</div>
""", unsafe_allow_html=True)
