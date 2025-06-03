import streamlit as st
import openai
import requests
import json
import re
from PIL import Image
import io
import base64

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
    
    .product-image {
        border-radius: 10px;
        width: 100%;
        height: 200px;
        object-fit: cover;
        margin-bottom: 1rem;
    }
    
    .product-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .product-price {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.2rem;
        color: #ff6b6b;
        margin-bottom: 0.5rem;
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
    
    .voice-button {
        background: linear-gradient(45deg, #4ecdc4, #44a08d);
        color: white;
        border: none;
        padding: 1rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1.1rem;
        margin: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .voice-button:hover {
        background: linear-gradient(45deg, #26d0ce, #2a5c5a);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(78,205,196,0.4);
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
</style>
""", unsafe_allow_html=True)

# Header section with gorgeous styling
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

# Voice recording function
def record_audio():
    """Handle voice recording with HTML5 Audio API"""
    audio_html = """
    <div style="text-align: center; margin: 2rem 0;">
        <button id="recordBtn" class="voice-button" onclick="toggleRecording()">
            🎙️ Start Recording
        </button>
        <div id="status" style="margin-top: 1rem; font-weight: 500;"></div>
        <audio id="audioPlayback" controls style="display: none; margin-top: 1rem; width: 100%;"></audio>
    </div>
    
    <script>
    let mediaRecorder;
    let audioChunks = [];
    let isRecording = false;
    
    async function toggleRecording() {
        const btn = document.getElementById('recordBtn');
        const status = document.getElementById('status');
        
        if (!isRecording) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = document.getElementById('audioPlayback');
                    audio.src = audioUrl;
                    audio.style.display = 'block';
                    
                    // Convert to base64 for Streamlit
                    const reader = new FileReader();
                    reader.onloadend = () => {
                        const base64 = reader.result.split(',')[1];
                        window.parent.postMessage({
                            type: 'audio_recorded',
                            data: base64
                        }, '*');
                    };
                    reader.readAsDataURL(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                btn.textContent = '⏹️ Stop Recording';
                btn.style.background = 'linear-gradient(45deg, #ff6b6b, #ff8e8e)';
                status.textContent = '🔴 Recording... Speak now!';
                status.style.color = '#ff6b6b';
                
            } catch (err) {
                status.textContent = '❌ Microphone access denied';
                status.style.color = '#ff6b6b';
            }
        } else {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
            isRecording = false;
            btn.textContent = '🎙️ Start Recording';
            btn.style.background = 'linear-gradient(45deg, #4ecdc4, #44a08d)';
            status.textContent = '✅ Recording complete! Processing...';
            status.style.color = '#4ecdc4';
        }
    }
    </script>
    """
    return audio_html

# Fashion tips for better UX
fashion_tips = [
    "💡 **Pro Tip**: Mention specific designers, colors, or occasions for better results!",
    "✨ **Style Hack**: Try describing the vibe - 'coastal grandmother', 'dark academia', 'cottagecore'",
    "🎯 **Search Smarter**: Include your budget range for more relevant results",
    "👑 **Celebrity Inspo**: Reference your style icons - 'like Zendaya's red carpet look'",
    "🌟 **Occasion Matters**: Specify if it's for work, dates, casual, or special events"
]

# Extract image URLs from text using regex
def extract_image_urls(text):
    """Extract image URLs from text using various patterns"""
    patterns = [
        r'https?://[^\s<>"]*\.(?:jpg|jpeg|png|gif|webp)(?:\?[^\s<>"]*)?',
        r'!\[.*?\]\((https?://[^\s)]+)\)',  # Markdown images
        r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # HTML img tags
    ]
    
    urls = []
    for pattern in patterns:
        urls.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return list(set(urls))  # Remove duplicates

# Extract product info from text
def extract_product_info(text):
    """Extract structured product information from AI response"""
    products = []
    
    # Split by common separators and look for product-like entries
    sections = re.split(r'\n\s*\n|\d+\.\s+|\*\*|\#\#', text)
    
    for section in sections:
        if len(section.strip()) < 20:  # Skip very short sections
            continue
            
        # Look for price patterns
        price_match = re.search(r'\$[\d,]+(?:\.\d{2})?', section)
        price = price_match.group() if price_match else None
        
        # Look for URLs
        url_match = re.search(r'https?://[^\s<>"]+', section)
        url = url_match.group() if url_match else None
        
        # Extract title (first substantial line)
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        title = lines[0] if lines else section[:100] + "..."
        
        # Clean title
        title = re.sub(r'^[\d\.\-\*\s]+', '', title)  # Remove numbering
        title = re.sub(r'\*+', '', title)  # Remove asterisks
        title = title.strip()
        
        if title and len(title) > 5:
            products.append({
                'title': title,
                'price': price,
                'url': url,
                'description': section.strip()
            })
    
    return products[:6]  # Limit to 6 products

# Search container
st.markdown('<div class="search-container">', unsafe_allow_html=True)

# Input mode selection with gorgeous styling
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("💬 Text Search", use_container_width=True):
        st.session_state.search_mode = "text"

with col2:
    if st.button("🎙️ Voice Search", use_container_width=True):
        st.session_state.search_mode = "voice"

# Initialize search mode
if 'search_mode' not in st.session_state:
    st.session_state.search_mode = "text"

user_query = ""

if st.session_state.search_mode == "text":
    st.markdown("### ✍️ Describe Your Perfect Look")
    user_query = st.text_input(
        "",
        placeholder="✨ Try: 'Hailey Bieber airport style under $200' or 'cottagecore dress for spring wedding'",
        key="fashion_query"
    )
    
    # Display a random fashion tip
    import random
    st.markdown(f'<div class="fashion-tip">{random.choice(fashion_tips)}</div>', unsafe_allow_html=True)

elif st.session_state.search_mode == "voice":
    st.markdown("### 🎙️ Voice Search")
    st.markdown("Click the button below to record your fashion request:")
    
    # Voice recording interface
    audio_component = st.components.v1.html(record_audio(), height=200)
    
    # Handle audio transcription (placeholder for now)
    if st.button("🔄 Process Voice Recording"):
        st.info("Voice processing will be implemented in the next update! Please use text search for now.")

st.markdown('</div>', unsafe_allow_html=True)

# Search button and processing
if st.button("🔍 Find My Perfect Style!", disabled=not user_query, use_container_width=True, key="main_search"):
    if user_query:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("🧠 AI is analyzing your style preferences...")
        progress_bar.progress(25)
        
        try:
            # Step 1: Refine query with fashion expertise
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
            
            # Step 2: Search with Perplexity for shopping results
            headers = {
                "Authorization": f"Bearer {PPLX_KEY}",
                "Content-Type": "application/json"
            }
            
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
                "search_domain_filter": [
                    "zara.com", "hm.com", "nordstrom.com", "asos.com", 
                    "urbanoutfitters.com", "target.com", "amazon.com",
                    "mango.com", "revolve.com", "shopbop.com"
                ]
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
                st.markdown(f"""
                <div class="search-container">
                    <h3>🎯 Your Style Search</h3>
                    <p style="font-size: 1.1rem; color: #666; font-style: italic;">"{refined_query}"</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Get AI response
                ai_response = data['choices'][0]['message']['content']
                
                # Extract structured product information
                products = extract_product_info(ai_response)
                
                # Display AI styling advice
                st.markdown("## 💎 Your Personal Stylist Says:")
                st.markdown(f'<div class="search-container">{ai_response}</div>', unsafe_allow_html=True)
                
                # Display shopping links in a beautiful grid
                if 'citations' in data and data['citations']:
                    st.markdown("## 🛍️ Shop These Curated Picks")
                    
                    # Create columns for product grid
                    cols = st.columns(2)
                    
                    for i, citation in enumerate(data['citations'][:8]):
                        col_idx = i % 2
                        
                        with cols[col_idx]:
                            # Extract domain for styling
                            domain = citation.split('/')[2].replace('www.', '').title()
                            
                            st.markdown(f"""
                            <div class="product-card">
                                <div class="product-title">Shop at {domain}</div>
                                <a href="{citation}" target="_blank" class="shop-button">
                                    ✨ Shop Now
                                </a>
                                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #666;">
                                    {citation[:50]}...
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Fashion advice section
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
