import streamlit as st
from imagecaption import generate_caption
from database import add_data,view_data
from classifier import classify_text


st.title("Toxic Classifier")
choice = st.radio("Choose Input type",["Text","Photo"])
if choice=="Text":
    text_input = st.text_input("Enter Text")
elif choice== "Photo":
    image_input = st.file_uploader("Upload Photo", ["jpg","jpeg","png","jfif"])
if choice== "Photo" and st.button("Classify"):
    caption = generate_caption(image_input)
    classification = classify_text(caption)
    st.write(f"Caption: {caption}")
    st.write(f"Classification: {classification}")  
    add_data(caption,"image",classification)
elif choice== "Text" and st.button("Classify"):
    classification = classify_text(text_input)
    st.write(f"Classification: {classification}",)
    add_data(text_input,"text",classification)

if st.button("View Database"):
    data = view_data()
    st.dataframe(data)

