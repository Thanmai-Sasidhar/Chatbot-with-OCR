# app.py
import os
import io
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import numpy as np
import cv2
import pytesseract

# Optional model imports (keep them; don't hardcode keys)
import google.generativeai as genai
import ollama

# Load env vars from .env locally (won't be committed if .gitignore is used)
load_dotenv()

# --- CONFIG ---
st.set_page_config(page_title="Chat + OCR", page_icon="🤖", layout="wide")

# Read API keys from environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")  # if using ollama remotely
# Do not hardcode keys here!

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# --- STORAGE (excluded from git via .gitignore) ---
os.makedirs("storage/images", exist_ok=True)
os.makedirs("storage/texts", exist_ok=True)

# If tesseract is not in PATH on your environment, uncomment and set path:
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # or your tesseract executable

def process_and_save(image: Image.Image, filename_prefix="chatocr"):
    """
    Resize -> run OCR -> save image and text.
    Returns: (extracted_text, img_path, txt_path)
    """
    # Resize to reasonable max dimension to avoid huge uploads
    max_dim = 800
    image = image.convert("RGB")
    image.thumbnail((max_dim, max_dim))

    # Convert and OCR
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_path = f"storage/images/{filename_prefix}_{timestamp}.png"
    txt_path = f"storage/texts/{filename_prefix}_{timestamp}.txt"

    image.save(img_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text, img_path, txt_path


# --- SESSION STATE initialization ---
if "history" not in st.session_state:
    st.session_state.history = {}
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "rename_chat" not in st.session_state:
    st.session_state.rename_chat = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Gemini API"

# Create initial chat if none
if not st.session_state.history:
    cid = f"chat_{st.session_state.chat_counter}"
    st.session_state.chat_counter += 1
    st.session_state.current_chat = cid
    st.session_state.history[cid] = {
        "name": "Default Chat",
        # messages: first message is system prompt (hidden from UI by display logic)
        "messages": [{"role": "system", "content": st.session_state.system_prompt}],
        # hidden_ocr: store extracted OCR texts here (NOT displayed)
        "hidden_ocr": [],
        # optional store paths
        "images": [],
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
            "hidden_ocr": [],
            "images": [],
        }

    st.markdown("## 💾 Chat History")
    # unique keys per chat button to avoid duplicate element id issues
    for cid, chatmeta in list(st.session_state.history.items()):
        cols = st.columns([0.8, 0.2])
        if cols[0].button(chatmeta["name"], key=f"chat_btn_{cid}", use_container_width=True):
            st.session_state.current_chat = cid
        if cols[1].button("⋮", key=f"menu_{cid}"):
            st.session_state.rename_chat = cid

        if st.session_state.rename_chat == cid:
            new_name = st.text_input("Rename", chatmeta["name"], key=f"rename_input_{cid}")
            if st.button("✅ Save", key=f"save_{cid}"):
                st.session_state.history[cid]["name"] = (new_name or chatmeta["name"])
                st.session_state.rename_chat = None
            if st.button("🗑 Delete", key=f"delete_{cid}"):
                del st.session_state.history[cid]
                st.session_state.rename_chat = None
                st.session_state.current_chat = list(st.session_state.history.keys())[0] if st.session_state.history else None
                st.experimental_rerun()


# --- MAIN CHAT UI ---
st.title("💬 Chat + OCR")

if st.session_state.current_chat:
    chat = st.session_state.history[st.session_state.current_chat]

    # Display previous messages (skip system messages)
    for msg in chat["messages"]:
        if msg["role"] != "system":
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Image upload (OCR). Keyed by chat id to avoid duplicate-element issues
    uploaded_file = st.file_uploader(
        "📷 Upload/Paste an Image for OCR",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key=f"ocr_upload_{st.session_state.current_chat}",
    )
    if uploaded_file:
        image = Image.open(uploaded_file)
        text, img_path, txt_path = process_and_save(image)

        # Display the uploaded image to the user (but DO NOT display OCR text)
        with st.chat_message("user"):
            st.image(image, caption="Uploaded Image (OCR applied, text kept hidden)", use_container_width=True)

        # Hidden: store the extracted OCR text in chat metadata (NOT appended to visible messages)
        chat.setdefault("hidden_ocr", []).append(text)
        chat.setdefault("images", []).append(img_path)

    # Text prompt input (unique key per chat)
    prompt = st.chat_input("Ask something...", key=f"chat_input_{st.session_state.current_chat}")
    if prompt:
        # optional: rename chat from first user prompt
        if chat["name"] in ["New Chat", "Default Chat"]:
            chat["name"] = prompt[:40] + ("..." if len(prompt) > 40 else "")

        chat["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Prepare messages for the model: include system prompt + hidden OCR + conversation ---
        # Build a list of textual segments (original code used list-of-strings interface)
        messages_for_model = []
        # include stored system prompt (first message)
        if chat["messages"] and chat["messages"][0]["role"] == "system":
            messages_for_model.append(chat["messages"][0]["content"])
            # then hidden OCR entries
            for o in chat.get("hidden_ocr", []):
                messages_for_model.append(f"[EXTRACTED IMAGE TEXT - hidden]\n{o}")
            # then the remaining messages in chronological order
            for m in chat["messages"][1:]:
                messages_for_model.append(m["content"])
        else:
            messages_for_model.append(st.session_state.system_prompt)
            for o in chat.get("hidden_ocr", []):
                messages_for_model.append(f"[EXTRACTED IMAGE TEXT - hidden]\n{o}")
            for m in chat["messages"]:
                messages_for_model.append(m["content"])

        # --- Call the chosen model and stream response ---
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_text = ""
            try:
                if st.session_state.model_choice == "Gemini API":
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    stream = model.generate_content(
                        messages_for_model,
                        stream=True,
                    )
                    for chunk in stream:
                        if chunk.text:
                            full_text += chunk.text
                            placeholder.markdown(full_text)

                else:
                    # Ollama local usage (keep same streaming pattern)
                    stream = ollama.chat(
                        model="llama3.1:8b",
                        messages=chat["messages"],  # ollama lib expects messages list
                        stream=True
                    )
                    for chunk in stream:
                        if "message" in chunk and "content" in chunk["message"]:
                            c = chunk["message"]["content"]
                            full_text += c
                            placeholder.markdown(full_text)

            except Exception as e:
                full_text = f"❌ Error: {e}"
                placeholder.error(full_text)

            # Save assistant message to conversation history
            chat["messages"].append({"role": "assistant", "content": full_text})

    # --- Download chat history (optional include hidden OCR) ---
    if len(chat["messages"]) > 1:
        include_ocr = st.checkbox("Include OCR text in download", value=False)
        txt = ""
        for m in chat["messages"]:
            if m["role"] != "system":
                txt += f"{m['role'].upper()}: {m['content']}\n\n"
        if include_ocr and chat.get("hidden_ocr"):
            txt += "\n=== OCR EXTRACTS (hidden) ===\n"
            for i, o in enumerate(chat.get("hidden_ocr", []), 1):
                txt += f"OCR {i}:\n{o}\n\n"

        st.download_button(
            "📤 Share Current Chat",
            data=io.BytesIO(txt.encode()),
            file_name=f"{chat['name'].replace(' ', '_')}.txt",
            mime="text/plain",
        )

else:
    st.info("Start a new chat from the sidebar ➕")
