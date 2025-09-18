import streamlit as st
import google.generativeai as genai
import ollama
import io
import pytesseract
from PIL import Image
import numpy as np
import cv2
import os
from datetime import datetime

st.set_page_config(page_title="Chat + OCR", page_icon="🤖", layout="wide")

# --- CONFIG ---
genai.configure(api_key="AIzaSyBN_bOwZpV4qterCrRWiZMclqai6CkZKhQ")  # 🔑 Replace with your API key
GEMINI_MODEL = "gemini-1.5-flash"
OLLAMA_MODEL = "llama3.1:8b"

# --- STORAGE ---
os.makedirs("storage/images", exist_ok=True)
os.makedirs("storage/texts", exist_ok=True)

def process_and_save(image, filename_prefix="chatocr"):  
    """Run OCR on an image and save results."""
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)  # ✅ OCR extraction

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = f"storage/images/{filename_prefix}_{timestamp}.png"
    txt_path = f"storage/texts/{filename_prefix}_{timestamp}.txt"

    image.save(img_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text, img_path, txt_path


# --- SESSION STATE ---
if "history" not in st.session_state: st.session_state.history = {}
if "chat_counter" not in st.session_state: st.session_state.chat_counter = 1
if "current_chat" not in st.session_state: st.session_state.current_chat = None
if "rename_chat" not in st.session_state: st.session_state.rename_chat = None
if "system_prompt" not in st.session_state: st.session_state.system_prompt = "You are a helpful assistant."
if "model_choice" not in st.session_state: st.session_state.model_choice = "Gemini API"

# --- INIT CHAT ---
if not st.session_state.history:
    cid = f"chat_{st.session_state.chat_counter}"
    st.session_state.chat_counter += 1
    st.session_state.current_chat = cid
    st.session_state.history[cid] = {
        "name": "Default Chat",
        "messages": [{"role": "system", "content": st.session_state.system_prompt}],
    }
elif st.session_state.current_chat is None:
    st.session_state.current_chat = list(st.session_state.history.keys())[0]


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🤖 Choose Model")
    st.session_state.model_choice = st.radio("Select Model:", ["Gemini API", "Ollama Local"])

    if st.button("➕ New Chat", use_container_width=True):
        cid = f"chat_{st.session_state.chat_counter}"
        st.session_state.chat_counter += 1
        st.session_state.current_chat = cid
        st.session_state.history[cid] = {
            "name": "New Chat",
            "messages": [{"role": "system", "content": st.session_state.system_prompt}],
        }

    st.markdown("## 💾 Chat History")
    for cid, chat in st.session_state.history.items():
        cols = st.columns([0.8, 0.2])
        if cols[0].button(chat["name"], key=f"chat_{cid}", use_container_width=True):
            st.session_state.current_chat = cid
        if cols[1].button("⋮", key=f"menu_{cid}"): 
            st.session_state.rename_chat = cid
        if st.session_state.rename_chat == cid:
            new_name = st.text_input("Rename", chat["name"], key=f"rename_input_{cid}")
            if st.button("✅ Save", key=f"save_{cid}"):
                st.session_state.history[cid]["name"] = new_name or chat["name"]
                st.session_state.rename_chat=None
            if st.button("🗑 Delete", key=f"delete_{cid}"):
                del st.session_state.history[cid]
                st.session_state.rename_chat=None
                st.session_state.current_chat = list(st.session_state.history.keys())[0] if st.session_state.history else None
                st.rerun()


# --- MAIN CHAT ---
st.title("💬 Chat + OCR")

if st.session_state.current_chat:
    chat = st.session_state.history[st.session_state.current_chat]

    # --- DISPLAY PREVIOUS MESSAGES ---
    for msg in chat["messages"]:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]): 
                st.markdown(msg["content"])

    # --- IMAGE UPLOAD IN CHAT (OCR, silent inject) ---
    uploaded_file = st.file_uploader("📷 Upload/Paste an Image for OCR", 
                                     type=["png","jpg","jpeg","bmp","tiff","webp"], 
                                     key="ocr_upload")
    if uploaded_file:
        image = Image.open(uploaded_file)
        text, img_path, txt_path = process_and_save(image)

        with st.chat_message("user"):
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # ✅ Hidden injection of OCR text into conversation
        chat["messages"].append({"role": "user", "content": f"Extracted from image:\n{text}"})

    # --- TEXT PROMPT INPUT ---
    prompt = st.chat_input("Ask something...")
    if prompt:
        if chat["name"] in ["New Chat", "Default Chat"]:
            chat["name"] = prompt[:40]+("..." if len(prompt)>40 else "")
        chat["messages"].append({"role":"user","content":prompt})
        with st.chat_message("user"): 
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder, full_text = st.empty(), ""
            try:
                if st.session_state.model_choice == "Gemini API":
                    model = genai.GenerativeModel(GEMINI_MODEL)
                    stream = model.generate_content(
                        [m["content"] for m in chat["messages"] if m["role"] in ("system","user")],
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.text:
                            full_text += chunk.text
                            placeholder.markdown(full_text)

                else:
                    stream = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=chat["messages"],
                        stream=True
                    )
                    for chunk in stream:
                        if "message" in chunk and "content" in chunk["message"]:
                            c = chunk["message"]["content"]
                            full_text += c
                            placeholder.markdown(full_text)

            except Exception as e:
                full_text=f"❌ Error: {e}"
                placeholder.error(full_text)

            chat["messages"].append({"role":"assistant","content":full_text})

    # --- DOWNLOAD CHAT HISTORY ---
    if len(chat["messages"]) > 1:
        txt = "".join([f"{m['role'].upper()}: {m['content']}\n\n" 
                       for m in chat["messages"] if m["role"]!="system"])
        st.download_button("📤 Share Current Chat", 
                           data=io.BytesIO(txt.encode()),
                           file_name=f"{chat['name'].replace(' ','_')}.txt", 
                           mime="text/plain")

else:
    st.info("Start a new chat from the sidebar ➕")
