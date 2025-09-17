import streamlit as st
import fitz  # PyMuPDF
import requests
import json
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Analysis Tool",
    page_icon="📄",
    layout="centered",
)

# --- Gemini API Configuration ---
# The API key is now hard-coded below.
# Warning: For security, it's better to use st.secrets for any shared or public app.
API_KEY = "AIzaSyC4CvDL1M7ykzWBU953xk6ku6clC2zHzbQ" 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

# --- Backend Functions ---

def extract_text_from_pdf(pdf_file):
    """Extracts text from an uploaded PDF file."""
    try:
        # Open the PDF from the uploaded file's bytes
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def analyze_text_with_gemini(text):
    """Sends text to the Gemini API for analysis with exponential backoff."""
    truncated_text = text[:15000] # Limit text to avoid exceeding API limits

    payload = {
        "contents": [{
            "parts": [{
                "text": f"Analyze the following text from a document. Provide a concise summary, extract 5-7 key topics or keywords, determine the overall sentiment (e.g., Positive, Negative, Neutral), and identify the language of the text.\n\nText: \"{truncated_text}\""
            }]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "summary": {"type": "STRING", "description": "A concise summary of the text."},
                    "keywords": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "An array of 5 to 7 key topics or keywords."
                    },
                    "sentiment": {"type": "STRING", "description": "The overall sentiment (e.g., Positive, Negative, Neutral)."},
                    "language": {"type": "STRING", "description": "The detected language of the text."}
                },
                "required": ["summary", "keywords", "sentiment", "language"]
            }
        }
    }

    headers = {'Content-Type': 'application/json'}
    
    # Exponential backoff for retries
    for i in range(3): # Retry up to 3 times
        try:
            response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            result = response.json()
            
            if "candidates" in result and result["candidates"][0]["content"]["parts"][0]["text"]:
                return json.loads(result["candidates"][0]["content"]["parts"][0]["text"])
            else:
                 # Handle cases where the API returns a 200 but the response is empty/malformed
                st.error("The AI model returned an unexpected response. It might be due to safety filters or other issues.")
                return None

        except requests.exceptions.RequestException as e:
            if i < 2:
                time.sleep(2 ** i) # Wait 1s, 2s
            else:
                st.error(f"Failed to connect to the AI model after several retries: {e}")
                return None
        except (KeyError, IndexError, json.JSONDecodeError):
             st.error("Failed to parse the AI model's response. The format was unexpected.")
             return None

    return None


# --- Streamlit UI ---
st.title("📄 PDF Analysis Tool")
st.markdown("Upload a PDF to get an AI-powered summary, key topics, and sentiment analysis.")

if not API_KEY:
    st.error("Gemini API Key is not configured. Please add it to your Streamlit secrets.")
else:
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document for analysis."
    )
    
    if uploaded_file is not None:
        with st.spinner("Analyzing your document... This may take a moment."):
            # Step 1: Extract text
            document_text = extract_text_from_pdf(uploaded_file)
    
            if document_text and len(document_text.strip()) > 50:
                # Step 2: Analyze text with Gemini
                analysis_result = analyze_text_with_gemini(document_text)
    
                # Step 3: Display results
                if analysis_result:
                    st.success("Analysis Complete!")
                    st.divider()

                    # Summary
                    st.subheader("📝 Summary")
                    st.write(analysis_result.get("summary", "No summary available."))
    
                    # Keywords
                    st.subheader("🔑 Key Topics")
                    keywords = analysis_result.get("keywords", [])
                    if keywords:
                        # Dynamically adjust columns based on number of keywords
                        num_keywords = len(keywords)
                        cols = st.columns(num_keywords if num_keywords > 0 else 1)
                        for i, keyword in enumerate(keywords):
                            with cols[i]:
                                st.info(keyword)
                    else:
                        st.write("No keywords were extracted.")
                    
                    st.divider()

                    # Additional Info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🗣️ Language")
                        st.write(f"**{analysis_result.get('language', 'N/A')}**")
                    
                    with col2:
                        st.subheader("😃 Sentiment")
                        sentiment = analysis_result.get('sentiment', 'N/A')
                        if sentiment.lower() == 'positive':
                            st.write(f"**<p style='color:green;'>{sentiment}</p>**", unsafe_allow_html=True)
                        elif sentiment.lower() == 'negative':
                            st.write(f"**<p style='color:red;'>{sentiment}</p>**", unsafe_allow_html=True)
                        else:
                            st.write(f"**<p style='color:orange;'>{sentiment}</p>**", unsafe_allow_html=True)

            else:
                st.warning("Could not extract enough text from the PDF. The document might be image-based, empty, or corrupted.")

