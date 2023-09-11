from langchain.document_loaders import YoutubeLoader
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain 
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

import streamlit as st
import os

load_dotenv()

# Hard code the YouTube URL
youtube_url = "https://www.youtube.com/watch?v=lpWE82y11hs"

# Display the Page Title
st.title('Video 1: The Economics of Digital')

# Embed the YouTube video
st.video(youtube_url)

# Set the API key
api_key = os.environ.get("OPENAI_API_KEY")
llm = OpenAI(temperature=0)  # Temp controls the randomness of the text

# Load video transcript and info
loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
docs = loader.load()

# Split the document for summarization
text_splitter = CharacterTextSplitter(chunk_size=1000, separator=" ", chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Display loading screen
loading_message = st.empty()
loading_message.text("Utilizing AI to summarize hours of thought and preparation into a single paragraph...")

# Summarize the video
chain = load_summarize_chain(llm, chain_type="map_reduce")
summary = chain.run(split_docs)

# Remove loading screen
loading_message.empty()

st.subheader("Summary")
st.write(summary)

# Allow users to ask questions about the video
question = st.text_input("Ask a question about the video:")
if question:
    try:
        response = llm.generate(prompts=[question], max_tokens=2000)
        # Extract and display the 'text' attribute of the first item in response.generations
        answer = response.generations[0][0].text.strip()
        st.write(answer)
    except Exception as e:
        st.write(f"Error: {e}")

# Display the first 256 chars of the transcript with an option to expand
st.subheader("Full Transcript")
transcript_preview = docs[0].page_content[:256] + "...more"
transcript_full = docs[0].page_content
if st.button("Show Full Transcript"):
    st.write(transcript_full)
else:
    st.write(transcript_preview)