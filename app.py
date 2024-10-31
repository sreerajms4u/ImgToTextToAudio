
#https://www.youtube.com/watch?v=_j7JEDWuqLE

from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI

from gtts import gTTS

import requests
import os

import streamlit as st

load_dotenv(find_dotenv())
HUGGINGHUB_API_TOKEN = os.getenv("HUGGINGFACE_HUB_API_TOKEN")
#img2Text
def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text=image_to_text(
        url)[0]['generated_text']
    print(text)
    return text
#llm
def generate_story(scenario):
    #template="""
    #You are a story teller;
    #You can generate a short story based on a simple narrative, the story should be no more than 20 words;

    #CONTEXT: {scenario}
    #STORY:
    #"""

    #prompt = PromptTemplate(template=template, input_variables=["scenario"])

    #story_llm = LLMChain(llm=OpenAI(
        #model_name="gpt-3.5-turbo",temperature=1), prompt=prompt, verbose=True)

    #story = story_llm.predict(scenario=scenario)

    
    generator = pipeline("text-generation", model="gpt2")
    result = generator(scenario, max_length=200, num_return_sequences=1)
    return result[0]["generated_text"]

#text to Speech
def text2speech(message):  
    # Create the TTS object
    tts = gTTS(message, lang="en")
    # Save the audio to a file
    tts.save("hello.mp3")

    #API_URL="https://huggingface.co/espnet/kan-bayashi_ljspeech_vits"
    #headers = {"Authorization": f"Bearer {HUGGINGHUB_API_TOKEN}"}
    #payloads = {
        #"inputs": message
    #}

    #response = requests.post(API_URL, headers=headers, json=payloads)
    #with open('audio1.mp3', 'wb') as file:
        #file.write(response.content)
def main():
    st.set_page_config(page_title="imag 2 audio story")

    st.header("Turn an Imageinto an Audio story")
    uploaded_file = st.file_uploader("Choose an image...",type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name,"wb") as file:
            file.write(bytes_data)
        scenario = img2text(uploaded_file.name)
        story = generate_story(scenario)
        text2speech(story)

        with st.expander("Content of the Image"):
            st.write(scenario)
        with st.expander("Story Of the Image"):
            st.write(story)

        st.audio("hello.mp3")


if __name__ == '__main__':
    main()
        