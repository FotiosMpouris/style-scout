import streamlit as st
import openai
import requests
import json
import re
from PIL import Image
import io
import base64
import time

# Page configuration with fashion-forward styling
st.set_page_config(
    page_title="Style Scout ✨",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for that high-fashion magazine look
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
    }
    
    .product-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
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
    
    .recording-button {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        color: white;
        border: none;
        padding: 1.5rem 3rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 5px 20px rgba(78,205,196,0.3);
    }
    
    .recording-button:hover {
        background: linear-gradient(45deg, #26d0ce, #2a5c5a);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(78,205,196,0.4);
    }
    
    .recording-active {
        background: linear-gradient(45deg, #ff6b6b, #ff8e8e) !important;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "text"
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""

# Header section
st.markdown("""
<div class="main-header">
    <h1 class="main-title">Style Scout</h1>
    <p class="subtitle">Your AI-powered personal fashion curator ✨</p>
</div>
""", unsafe_allow_html=True)

# Load API keys
try:
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    PPLX_KEY = st.secrets["PPLX_API_KEY"]
    openai.api_key = OPENAI_KEY
except Exception as e:
    st.error("🔑 Please set up your API keys in Streamlit Cloud secrets.")
    st.stop()

# Detect if search is for vintage/secondhand items
def is_vintage_search(query):
    """Detect if the search is for vintage, used, or secondhand items"""
    vintage_keywords = [
        'vintage', 'used', 'secondhand', 'second hand', 'pre-owned', 'thrift', 
        'consignment', 'pre-loved', 'previously owned', 'estate', 'antique',
        'retro', 'preloved', 'gently used', 'resale'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in vintage_keywords)

# Extract product info from text
def extract_product_info(text):
    """Extract structured product information from AI response"""
    products = []
    sections = re.split(r'\n\s*\n|\d+\.\s+|\*\*|\#\#', text)
    
    for section in sections:
        if len(section.strip()) < 20:
            continue
            
        price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', section)
        price = price_match.group() if price_match else None
        
        url_match = re.search(r'https?://[^\s<>"]+', section)
        url = url_match.group() if url_match else None
        
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        title = lines[0] if lines else section[:100] + "..."
        
        title = re.sub(r'^[\d\.\-\*\s]+', '', title)
        title = re.sub(r'\*+', '', title)
        title = title.strip()
        
        if title and len(title) > 5:
            products.append({
                'title': title,
                'price': price,
                'url': url,
                'description': section.strip()
            })
    
    return products[:6]

# Voice recording component
def voice_recorder():
    """Create voice recording interface"""
    st.markdown('<div class="voice-recording">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Voice Fashion Search")
    st.markdown("Click to record your fashion request:")
    
    # Audio recorder using st.audio_input (available in newer Streamlit versions)
    audio_file = st.file_uploader("Record or upload audio", type=['wav', 'mp3', 'm4a'], key="audio_upload")
    
    if audio_file is not None:
        st.audio(audio_file)
        
        if st.button("🔄 Transcribe Audio", key="transcribe_btn"):
            with st.spinner("🎧 Transcribing your voice..."):
                try:
                    # Prepare audio file for OpenAI Whisper
                    audio_bytes = audio_file.read()
                    
                    # Create a temporary file-like object
                    audio_file_obj = io.BytesIO(audio_bytes)
                    audio_file_obj.name = "audio.wav"
                    
                    # Transcribe with OpenAI Whisper
                    transcript = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file_obj,
                        response_format="text"
                    )
                    
                    st.session_state.transcribed_text = transcript
                    st.success(f"✨ I heard: '{transcript}'")
                    
                except Exception as e:
                    st.error(f"Transcription error: {str(e)}")
    
    # Manual voice input as backup
    st.markdown("**Or type what you would say:**")
    manual_voice = st.text_input("", placeholder="Type your voice search here...", key="manual_voice")
    
    if manual_voice:
        st.session_state.transcribed_text = manual_voice
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return st.session_state.transcribed_text

# Fashion tips
fashion_tips = [
    "💡 **Pro Tip**: Mention specific designers, colors, or occasions for better results!",
    "✨ **Style Hack**: Try describing the vibe - 'coastal grandmother', 'dark academia', 'cottagecore'",
    "🎯 **Search Smarter**: Include your budget range for more relevant results",
    "👑 **Celebrity Inspo**: Reference your style icons - 'like Zendaya's red carpet look'",
    "🌟 **Occasion Matters**: Specify if it's for work, dates, casual, or special events",
    "♻️ **Sustainable Style**: Try 'vintage', 'secondhand', or 'preloved' for eco-friendly finds!"
]

# Search container
st.markdown('<div class="search-container">', unsafe_allow_html=True)

# Input mode selection
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("💬 Text Search", use_container_width=True):
        st.session_state.search_mode = "text"

with col2:
    if st.button("🎙️ Voice Search", use_container_width=True):
        st.session_state.search_mode = "voice"

user_query = ""

if st.session_state.search_mode == "text":
    st.markdown("### ✍️ Describe Your Perfect Look")
    user_query = st.text_input(
        "",
        placeholder="✨ Try: 'vintage leather jacket like Kurt Cobain' or 'secondhand designer bag under $300'",
        key="fashion_query"
    )
    
    # Display a random fashion tip
    import random
    st.markdown(f'<div class="fashion-tip">{random.choice(fashion_tips)}</div>', unsafe_allow_html=True)

elif st.session_state.search_mode == "voice":
    user_query = voice_recorder()

st.markdown('</div>', unsafe_allow_html=True)

# Search button and processing
if st.button("🔍 Find My Perfect Style!", disabled=not user_query, use_container_width=True, key="main_search"):
    if user_query:
        # Check if this is a vintage/secondhand search
        is_vintage = is_vintage_search(user_query)
        
        if is_vintage:
            st.markdown('<div class="vintage-indicator">🌿 Sustainable Fashion Search Detected</div>', unsafe_allow_html=True)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🧠 AI is analyzing your style preferences...")
        progress_bar.progress(25)
        
        try:
            # Step 1: Refine query with fashion expertise
            if is_vintage:
                system_prompt = """You are an expert vintage fashion curator and sustainable style consultant. The user is looking for secondhand, vintage, or pre-owned fashion items. Convert their request into search terms that work well for vintage/secondhand platforms like Depop, Poshmark, eBay, TheRealReal, Vestiaire Collective, and ThredUp.

Consider:
- Vintage eras and specific style periods (90s, Y2K, 70s boho, etc.)
- Designer brands that hold value in resale markets
- Condition descriptions (excellent, good, fair)
- Authentication concerns for luxury items
- Size variations in vintage clothing
- Sustainable fashion terminology

Format for secondhand platforms. Include era, style descriptors, and brand suggestions.

Examples:
- "something like Kurt Cobain wore" → "vintage 90s grunge flannel band tee leather jacket"
- "designer bag but affordable" → "pre-owned luxury handbag Kate Spade Coach vintage"
- "retro dress for date night" → "vintage midi dress 70s boho or 90s slip dress"

Keep it focused and under 15 words."""
            else:
                system_prompt = """You are an expert fashion stylist and personal shopper with deep knowledge of current trends, designers, and shopping. Convert the user's fashion request into a sophisticated search strategy that captures both their style preferences and practical shopping needs.

Consider:
- Current fashion trends and seasonal appropriateness
- Price points and value for money
- Versatility and styling potential
- Brand recommendations for different budgets
- Specific product details that matter (fit, fabric, etc.)

Format your refined search to work well with online shopping searches. Include key descriptive terms, style categories, and any mentioned price ranges.

Examples:
- "something trendy for dates" → "date night outfit women trendy 2024 midi dress or cute top"
- "professional but not boring" → "modern business casual women blazer contemporary work outfit"
- "like what Bella Hadid wears off-duty" → "model off duty street style oversized blazer vintage jeans"

Keep it focused and under 15 words."""

            chat_response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.4,
                max_tokens=100
            )
            
            refined_query = chat_response.choices[0].message.content.strip()
            
            status_text.text("🔍 Searching the best fashion retailers...")
            progress_bar.progress(50)
            
            # Step 2: Choose search domains based on vintage detection
            if is_vintage:
                search_domains = [
                    "depop.com", "poshmark.com", "ebay.com", "therealreal.com",
                    "vestiairecollective.com", "thredup.com", "rebag.com",
                    "fashionphile.com", "tradesy.com", "vinted.com"
                ]
                search_focus = "vintage, secondhand, and pre-owned fashion platforms"
            else:
                search_domains = [
                    "zara.com", "hm.com", "nordstrom.com", "asos.com", 
                    "urbanoutfitters.com", "target.com", "amazon.com",
                    "mango.com", "revolve.com", "shopbop.com"
                ]
                search_focus = "trusted fashion retailers"
            
            # Search with Perplexity
            headers = {
                "Authorization": f"Bearer {PPLX_KEY}",
                "Content-Type": "application/json"
            }
            
            if is_vintage:
                search_prompt = f"""Find specific vintage/secondhand clothing items for: {refined_query}

Please provide:
1. Specific vintage or pre-owned items with detailed descriptions
2. Estimated price ranges for secondhand market
3. Direct links from secondhand platforms like Depop, Poshmark, eBay, TheRealReal, ThredUp, Vestiaire Collective
4. Tips for buying vintage/secondhand (sizing, authenticity, condition)
5. Era identification and styling suggestions

Focus on sustainable fashion and secondhand marketplaces. Include authentication tips for designer items."""
            else:
                search_prompt = f"""Find specific clothing items for: {refined_query}

Please provide:
1. Specific product recommendations with exact names and descriptions
2. Price ranges when available  
3. Direct shopping links from reputable retailers
4. Brief styling suggestions

Focus on these trusted retailers: Zara, H&M, Nordstrom, ASOS, Urban Outfitters, Target, Amazon Fashion, Mango, and similar popular fashion sites.

Include product details like colors, sizes available, and key features."""
            
            body = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": search_prompt}],
                "return_citations": True,
                "return_images": False,
                "search_domain_filter": search_domains
            }
            
            status_text.text("✨ Curating your personalized style recommendations...")
            progress_bar.progress(75)
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=body,
                timeout=30
            )
            
            progress_bar.progress(100)
            status_text.text("🎉 Your style curation is ready!")
            
            if response.status_code == 200:
                data = response.json()
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display refined search
                search_type = "Vintage/Secondhand" if is_vintage else "New Fashion"
                st.markdown(f"""
                <div class="search-container">
                    <h3>🎯 Your {search_type} Search</h3>
                    <p style="font-size: 1.1rem; color: #666; font-style: italic;">"{refined_query}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get AI response
                ai_response = data['choices'][0]['message']['content']
                
                # Display AI styling advice
                advisor_title = "💎 Your Vintage Curator Says:" if is_vintage else "💎 Your Personal Stylist Says:"
                st.markdown(f"## {advisor_title}")
                st.markdown(f'<div class="search-container">{ai_response}</div>', unsafe_allow_html=True)
                
                # Display shopping links in a beautiful grid
                if 'citations' in data and data['citations']:
                    shop_title = "🌿 Shop Sustainable Fashion" if is_vintage else "🛍️ Shop These Curated Picks"
                    st.markdown(f"## {shop_title}")
                    
                    # Create columns for product grid
                    cols = st.columns(2)
                    
                    for i, citation in enumerate(data['citations'][:8]):
                        col_idx = i % 2
                        
                        with cols[col_idx]:
                            # Extract domain for styling
                            domain = citation.split('/')[2].replace('www.', '').title()
                            
                            # Choose button style based on platform type
                            button_class = "vintage-button" if is_vintage else "shop-button"
                            button_emoji = "♻️" if is_vintage else "✨"
                            
                            st.markdown(f"""
                            <div class="product-card">
                                <div class="product-title">Shop at {domain}</div>
                                <a href="{citation}" target="_blank" class="{button_class}">
                                    {button_emoji} Shop Now
                                </a>
                                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
                                    {citation[:50]}...
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Fashion advice section
                if is_vintage:
                    st.markdown("""
                    ## 🌿 Sustainable Shopping Tips
                    <div class="fashion-tip">
                        <strong>Vintage Curator's Note:</strong> Always check measurements rather than size labels for vintage pieces. Ask sellers about condition, authenticity for designer items, and return policies. Vintage sizing runs smaller than modern!
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    ## 💫 Styling Tips
                    <div class="fashion-tip">
                        <strong>Stylist's Note:</strong> Remember to check size guides carefully, and don't be afraid to mix high and low-end pieces. The best outfits often combine investment pieces with trendy, affordable finds!
                    </div>
                    """, unsafe_allow_html=True)
                
            else:
                st.error(f"Search temporarily unavailable (Error: {response.status_code}). Please try again!")
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Oops! Something went wrong: {str(e)}")

# Footer with style
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
