import streamlit as st
import openai
import requests
import json

# Page config
st.set_page_config(
    page_title="Style Scout",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Style Scout")
st.write("Find your perfect outfit with AI-powered search!")

# Load API keys from Streamlit secrets
try:
    OPENAI_KEY = st.secrets["openai"]["OPENAI_API_KEY"]
    PPLX_KEY = st.secrets["perplexity"]["PPLX_API_KEY"]
    openai.api_key = OPENAI_KEY
except Exception as e:
    st.error("Please set up your API keys in Streamlit Cloud secrets.")
    st.stop()

# Input mode selection
mode = st.radio("How would you like to search?", ["Type your request", "Voice search (coming soon)"])

user_query = ""

if mode == "Type your request":
    user_query = st.text_input(
        "Describe what you're looking for:",
        placeholder="e.g., 'cozy oversized sweater like Taylor Swift wears' or 'black leather boots under $100'"
    )

# For now, disable voice search until we add the audio component
if mode == "Voice search (coming soon)":
    st.info("Voice search will be available in the next update! Please use text search for now.")

# Search button and processing
if st.button("🔍 Find My Style!", disabled=not user_query) and user_query:
    with st.spinner("AI is analyzing your style request..."):
        try:
            # Step 1: Use OpenAI to refine the search query
            system_prompt = """You are a fashion search assistant. Convert the user's casual fashion request into a concise, searchable product query that would work well for online shopping. Focus on key attributes like item type, color, style, and price range if mentioned.

Examples:
- "something like what celebrities wear to airports" → "casual chic airport outfit women"
- "cozy sweater for winter dates" → "oversized knit sweater women winter"
- "professional but trendy work clothes" → "business casual blazer women modern"

Keep it under 10 words and focus on searchable terms."""

            chat_response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.3,
                max_tokens=50
            )
            
            refined_query = chat_response.choices[0].message.content.strip()
            st.write(f"🎯 Searching for: **{refined_query}**")
            
        except Exception as e:
            st.error(f"Error processing with OpenAI: {str(e)}")
            refined_query = user_query  # Fallback to original query

    with st.spinner("Finding the best matches..."):
        try:
            # Step 2: Search with Perplexity
            headers = {
                "Authorization": f"Bearer {PPLX_KEY}",
                "Content-Type": "application/json"
            }
            
            search_prompt = f"Find online shopping links for: {refined_query}. Focus on reputable retailers and include specific product details."
            
            body = {
                "model": "llama-3.1-sonar-small-128k-online",
                "messages": [{"role": "user", "content": search_prompt}],
                "search_domain_filter": ["amazon.com", "nordstrom.com", "zara.com", "hm.com", "target.com", "walmart.com"],
                "return_citations": True,
                "return_images": False
            }
            
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=body,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Display AI response
                ai_response = data['choices'][0]['message']['content']
                st.write("### 🛍️ Style Recommendations")
                st.write(ai_response)
                
                # Display citations if available
                if 'citations' in data and data['citations']:
                    st.write("### 🔗 Shopping Links")
                    for i, citation in enumerate(data['citations'][:8], 1):
                        st.write(f"{i}. [{citation}]({citation})")
                        
            else:
                st.error(f"Search API error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Error searching: {str(e)}")

# Footer
st.markdown("---")
st.markdown("💡 **Tips**: Be specific! Mention colors, styles, occasions, or even celebrity inspiration for better results.")
