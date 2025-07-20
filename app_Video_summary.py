import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from youtube_transcript_api import YouTubeTranscriptApi  # for getting YT subtitles
import requests  # for fetching webpage
from bs4 import BeautifulSoup  # for extracting text from HTML
st.set_page_config(page_title="YT/Web Summary", page_icon="🦜")
st.title("🦜 LangChain: Summarize YT Video or Website")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
url = st.text_input("Enter YouTube or Website URL")

# Prompt template for summarization
prompt = PromptTemplate(
    template="Summarize the following content in Points and about 1000 words:\n{text}",
    input_variables=["text"]
)
def get_yt_text(url):
    # extract video id from URL
    vid = url.split("watch?v=")[-1].split("&")[0] if "watch?v=" in url else url.split("youtu.be/")[-1]
    transcript = YouTubeTranscriptApi.get_transcript(vid)  # gets captions
    return " ".join(t['text'] for t in transcript)

def get_web_text(url):
    resp = requests.get(url, timeout=10)  # fetch page
    soup = BeautifulSoup(resp.content, "html.parser")  # parse HTML
    return soup.get_text(separator="\n", strip=True)  # extract visible text
if st.button("Summarize"):
    if not groq_api_key or not url:
        st.error("Please enter both API key and URL.")
    elif not validators.url(url):
        st.error("Invalid URL.")
    else:
        try:
            with st.spinner("Summarizing..."):
                llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)

                # Decide source type: YouTube or web
                if "youtube.com" in url or "youtu.be" in url:
                    text = get_yt_text(url)  # YT captions
                else:
                    text = get_web_text(url)  # webpage content
                doc = Document(page_content=text)
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run([doc])
                st.success(summary)
        except Exception as e:
            st.exception(e)
