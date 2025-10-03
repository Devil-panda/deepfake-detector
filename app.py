import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import face_recognition
import cv2
import numpy as np
import os
import tempfile
import pandas as pd
from PIL import Image
import database as db
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import yt_dlp
import base64

st.set_page_config(
    page_title="Deepfake Detection",
    page_icon="🤖",
    layout="wide"
)

def local_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_name}. Using default styles.")

def set_login_background(image_file):
    """
    Sets a background image and custom theme for the login page.
    Args:
    image_file (str): The path to the image file.
    """
    if not os.path.exists(image_file):
        st.warning(f"Background image file not found: {image_file}. Skipping background.")
        return

    with open(image_file, "rb") as f:
        img_bytes = f.read()
    encoded_string = base64.b64encode(img_bytes).decode()

    page_bg_img = f"""
    <style>
    /* --- Page Layout --- */
    [data-testid="stAppViewContainer"] > .main {{
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        justify-content: space-between;
        align-items: center;
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* --- Footer Styling --- */
    .login-footer {{
        text-align: center;
        padding: 1rem;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }}

    /* --- Themed Header --- */
    h1 {{
        color: #ff3b30;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
        padding-top: 1rem;
        margin-bottom: 6.5rem;
    }}

    /* --- Mirror Blur Form Container --- */
    [data-testid="stForm"] {{
        background: rgba(10, 10, 10, 0.65);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 59, 48, 0.5);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-radius: 12px;
        padding: 24px;
    }}

    /* --- NEW: Keep button text white when clicked/active --- */
    div[data-testid="stForm"] .stButton>button:active,
    div[data-testid="stForm"] .stButton>button:focus {{
        color: white !important;
        background-color: #ff3b30 !important;
    }}

    /* --- Hide "Press Enter to submit" text --- */
    [data-testid="stForm"] div[data-baseweb="form-control-caption"] {{
        display: none !important;
    }}

    /* --- Themed Input Fields --- */
    .stTextInput > div > div {{
        background-color: rgba(30, 30, 30, 0.5);
        border: 1px solid rgba(255, 59, 48, 0.4);
        border-radius: 8px;
    }}
    /* This makes the actual input area transparent inside the container */
    .stTextInput > div > div > input {{
        background: transparent;
        border: none;
        color: #fafafa;
    }}
    /* This styles the container when you click inside it */
    .stTextInput > div > div:focus-within {{
        border-color: #ff3b30;
        box-shadow: 0 0 0 3px rgba(255, 59, 48, 0.4);
    }}

    /* --- Align Submit Button to the Right --- */
    div[data-testid="stForm"] .stButton {{
        display: flex;
        justify-content: flex-end;
    }}

    /* --- Themed Button in Login Form --- */
    div[data-testid="stForm"] .stButton>button {{
        background-color: #ff3b30;
        color: white;
        border: none;
        border-radius: 25px;
    }}
    div[data-testid="stForm"] .stButton>button:hover {{
        background-color: #ff6a6a;
    }}

    /* --- Themed Tabs --- */
    [data-baseweb="tab-list"] {{
        background-color: transparent;
        border-radius: 8px;
        padding: 4px;
        border-bottom: none !important;
        gap: 3rem; /* NEW: Adds space between the tab buttons */
    }}
    [data-baseweb="tab"][aria-selected="true"] {{
        background-color: #ff3b30 !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(255, 59, 48, 0.4) !important;
        border-radius: 6px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

def get_face_from_video(video_path, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    faces = []
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if frame_count % frame_skip == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    faces.append(Image.fromarray(rgb_frame[top:bottom, left:right]))
            frame_count += 1
        else: break
    cap.release()
    return faces

def get_face_from_image(image):
    rgb_image = np.array(image.convert('RGB'))
    face_locations = face_recognition.face_locations(rgb_image)
    if face_locations:
        top, right, bottom, left = face_locations[0]
        return Image.fromarray(rgb_image[top:bottom, left:right])
    return None

class validation_dataset(Dataset):
    def __init__(self, video_path, sequence_length=20, transform=None):
        self.video_path, self.transform, self.count = video_path, transform, sequence_length
    def __len__(self): return 1
    def __getitem__(self, idx):
        frames = get_face_from_video(self.video_path)
        if not frames: return torch.zeros((self.count, 3, 112, 112))
        if len(frames) > self.count:
            indices = np.linspace(0, len(frames) - 1, self.count, dtype=int)
            frames = [frames[i] for i in indices]
        frames_tr = [self.transform(frame) for frame in frames]
        if len(frames_tr) < self.count:
            frames_tr.extend([torch.zeros((3, 112, 112))] * (self.count - len(frames_tr)))
        return torch.stack(frames_tr)

def predict(model, input_tensor, device):
    model.eval()
    with torch.no_grad():
        if len(input_tensor.shape) == 3: input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
        elif len(input_tensor.shape) == 4: input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)
        _, outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, preds = torch.max(probabilities, 1)
        return (preds.item(), confidence.item(), probabilities.cpu().numpy().flatten())

@st.cache_resource
def load_model(device):
    try:
        model = Model(2).to(device)
        path_to_model = 'checkpoint_new.pt'
        if not os.path.exists(path_to_model):
            st.error(f"Model checkpoint not found at '{path_to_model}'. Please place it in the root directory.")
            return None
        model.load_state_dict(torch.load(path_to_model, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def download_video_from_url(url):
    try:
        temp_dir = tempfile.gettempdir()
        ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': os.path.join(temp_dir, '%(title)s.%(ext)s'), 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get('title', 'video_from_url')
            filepath = ydl.prepare_filename(info_dict)
            return filepath, video_title
    except Exception as e:
        st.error(f"Error downloading video: {e}")
        return None, None

def clear_last_analysis():
    st.session_state.processing = False
    if 'last_analysis' in st.session_state:
        del st.session_state.last_analysis

def main_app():
    header_cols = st.columns([1, 2, 1])

    with header_cols[0]:
        col1, col2 = st.columns([0.8, 1.5]) 
        with col1:
            st.markdown(f"""
                <div style='font-size: 1.2em; padding-top: 8px;'> 
                    Welcome, <strong>{st.session_state['username']}</strong>!
                    {'👑' if st.session_state.get('is_admin', False) else ''}
                </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("🚪", help="Logout"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    with header_cols[1]:
        st.markdown("<h1 style='text-align: center; color: #ff3b30;'>🤖 DeepFake Detection Model</h1>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    tab_list = ["🏠 Home", "🔍 Analysis Tools", "📜 History"]
    if st.session_state.get('is_admin', False):
        tab_list.insert(0, "👑 Admin Dashboard")
    
    tabs = st.tabs(tab_list)
    
    admin_tab_index = 0 if st.session_state.get('is_admin', False) else -1
    home_tab_index = 1 if st.session_state.get('is_admin', False) else 0
    analysis_tab_index = 2 if st.session_state.get('is_admin', False) else 1
    history_tab_index = 3 if st.session_state.get('is_admin', False) else 2

    if st.session_state.get('is_admin', False):
        with tabs[admin_tab_index]:
            st.header("Administrator Dashboard")
            st.subheader("Platform Statistics")
            
            total_users = db.get_total_user_count()
            total_analyses = db.get_total_analysis_count()

            col1, col2 = st.columns(2)
            col1.metric("Total Registered Users", total_users)
            col2.metric("Total Analyses Performed", total_analyses)
            st.markdown("---")

            st.subheader("User Management")
            all_users = db.get_all_users()
            current_user_id = db.get_user_id(st.session_state['username'])

            if all_users:
                for user in all_users:
                    user_id, username, is_admin = user
                    if user_id == current_user_id: continue

                    with st.expander(f"Manage User: **{username}** {'(Admin)' if is_admin else ''}"):
                        st.write(f"User ID: {user_id}")
                        if st.button(f"Delete User {username}", key=f"delete_{user_id}", type="primary"):
                            db.delete_user_and_history(user_id)
                            st.success(f"Successfully deleted user {username} and all their data.")
                            st.rerun()
            else:
                st.info("No other users found.")
    
    with tabs[home_tab_index]:
        st.header(f"Welcome to the Deepfake Detection Engine, {st.session_state['username']}!")
        st.markdown("This dashboard is your first line of defense against digital deception. Here, you can learn about deepfakes and use our powerful tools to analyze suspicious media.")
        st.markdown("---")

        st.subheader("🤔 What is a Deepfake?")
        st.write(
            """
            A **"deepfake"** is a term for synthetic media created using artificial intelligence (AI). The name is a blend of "deep learning" and "fake." These techniques are used to manipulate or generate visual and audio content with a high potential to deceive. 
            
            Essentially, a deepfake can be a video where a person's face is swapped with someone else's, an image that has been altered, or an audio clip where a person's voice is cloned to say things they never actually said.
            """
        )

        st.subheader("🛠️ How Are Deepfakes Created?")
        st.write(
            """
            The most common method involves a type of AI model called a **Generative Adversarial Network (GAN)**. Think of it as a competition between two AIs:
            
            - **The Generator:** This AI tries to create realistic fake images or video frames.
            - **The Discriminator:** This AI acts as a detective, trying to distinguish between the Generator's fakes and real images.
            
            Over millions of cycles, the Generator becomes incredibly skilled at creating fakes that are nearly indistinguishable from reality.
            """
        )

        st.subheader("🔬 How Our Tool Detects Them")
        st.write(
            """
            Our engine uses a sophisticated AI model (a ResNeXt CNN combined with an LSTM network) to analyze media. It doesn't just look at a single picture; it examines sequences of frames in videos to spot inconsistencies that are often invisible to the human eye. The model looks for subtle clues, such as:
            - Unnatural facial movements or expressions.
            - Inconsistent lighting and shadows on the face.
            - Strange blinking patterns (or a complete lack of blinking).
            - Digital artifacts or distortions around the edges of the face.
            """
        )
        
        st.subheader("🧐 How You Can Spot a Deepfake")
        st.write("While AI detectors are powerful, you can also train your own eyes to spot potential fakes. Look for these red flags:")
        st.markdown(
            """
            - **👁️ Unnatural Eye Movement:** The person may blink strangely or not at all. Their gaze might seem fixed or misaligned.
            - **👄 Awkward Facial Expressions:** Emotions may appear flat or expressions may not match the context of the speech.
            - **🎨 Mismatched Lighting:** The lighting on the person's face might not match the lighting of the surrounding environment.
            - **模糊 Edges and Artifacts:** Look for blurry or distorted areas, especially where the face meets the hair or neck.
            - **🔊 Audio/Video Sync Issues:** The sound may not be perfectly synced with the lip movements.
            """
        )
        st.info("💡 **Remember:** When in doubt, use the 'Analysis Tools' tab to get an AI-powered second opinion!")


    with tabs[history_tab_index]:
        st.header("Your Analysis History")
        user_id = db.get_user_id(st.session_state['username'])
        if user_id:
            user_analysis_count = db.get_user_analysis_count(user_id)
            st.metric("Your Total Analyses", user_analysis_count)

            history = db.get_user_history(user_id)
            if history:
                st.write("---")
                col_spec = [3, 3, 2, 2, 2, 1]
                cols = st.columns(col_spec)
                cols[0].write("**Timestamp**")
                cols[1].write("**Source**")
                cols[2].write("**Type**")
                cols[3].write("**Result**")
                cols[4].write("**Confidence**")
                
                for record in history:
                    record_id, _, filename, analysis_type, prediction, confidence, timestamp = record
                    cols = st.columns(col_spec)
                    cols[0].text(datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
                    cols[1].text(filename)
                    cols[2].text(analysis_type)
                    cols[3].text(prediction)
                    cols[4].text(f"{confidence*100:.2f}%")
                    if cols[5].button("🗑️", key=f"delete_record_{record_id}", type="primary", help="Delete this record"):
                        db.delete_analysis_record(record_id)
                        st.rerun()
                st.write("---")
            else:
                st.info("You have no analysis history yet. Go to the 'Analysis Tools' tab to get started!")

    with tabs[analysis_tab_index]:
        im_size, mean, std = 112, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        transform_list = transforms.Compose([transforms.Resize((im_size, im_size)), transforms.ToTensor(), transforms.Normalize(mean, std)])
        
        st.header("Analysis Tools")
        st.write("Upload a video or image file, or provide a URL to a video, to analyze its authenticity.")
        
        upload_video_tab, upload_image_tab, url_tab = st.tabs(["📹 Upload Video", "🖼️ Upload Image", "🌐 From URL (Beta)"])
        
        with upload_video_tab:
            st.subheader("Upload a Video for Analysis")
            uploaded_video = st.file_uploader("Choose a video...", type=["mp4"], key="video_uploader", on_change=clear_last_analysis)
            if uploaded_video and model and not st.session_state.get('processing', False):
                st.session_state.processing = True
                with st.spinner('Analyzing video...'):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                        tfile.write(uploaded_video.getvalue())
                        temp_video_path = tfile.name
                    
                    video_tensor = validation_dataset(temp_video_path, sequence_length=20, transform=transform_list)[0]
                    prediction, confidence, probabilities = predict(model, video_tensor, device)
                    os.remove(temp_video_path)
                    
                    user_id = db.get_user_id(st.session_state['username'])
                    if user_id:
                        result_text = "REAL" if prediction == 1 else "FAKE"
                        db.add_analysis_record(user_id, uploaded_video.name, 'Video', result_text, confidence)
                        st.session_state.last_analysis = {'source_type': 'upload_video', 'filename': uploaded_video.name, 'file_bytes': uploaded_video.getvalue(), 'prediction': prediction, 'confidence': confidence, 'probabilities': probabilities}
                        st.rerun()

            if 'last_analysis' in st.session_state and st.session_state.last_analysis.get('source_type') == 'upload_video':
                res = st.session_state.last_analysis
                if uploaded_video and res['filename'] == uploaded_video.name:
                    st.markdown("---"); st.subheader("Analysis Result")
                    col1, col2 = st.columns(2)
                    with col1: st.video(res['file_bytes'])
                    with col2:
                        result_text = "REAL" if res['prediction'] == 1 else "FAKE"
                        if res['prediction'] == 1: st.success(f"**Result: {result_text}**")
                        else: st.error(f"**Result: {result_text}**")
                        st.metric(label="Confidence", value=f"{res['confidence']*100:.2f}%")
                        st.bar_chart(pd.DataFrame({'Category': ['Fake', 'Real'], 'Probability': res['probabilities']}).set_index('Category'))
                    st.markdown("---")

        with upload_image_tab:
            st.subheader("Upload an Image for Analysis")
            uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png"], key="image_uploader", on_change=clear_last_analysis)
            if uploaded_image and model and not st.session_state.get('processing', False):
                st.session_state.processing = True
                with st.spinner('Analyzing image...'):
                    image = Image.open(uploaded_image)
                    face = get_face_from_image(image)
                    if face:
                        image_tensor = transform_list(face)
                        prediction, confidence, probabilities = predict(model, image_tensor, device)
                        user_id = db.get_user_id(st.session_state['username'])
                        if user_id:
                            result_text = "REAL" if prediction == 1 else "FAKE"
                            db.add_analysis_record(user_id, uploaded_image.name, 'Image', result_text, confidence)
                            st.session_state.last_analysis = {'source_type': 'upload_image', 'filename': uploaded_image.name, 'file_bytes': uploaded_image.getvalue(), 'prediction': prediction, 'confidence': confidence, 'probabilities': probabilities}
                            st.rerun()
                    else:
                        st.warning("No face detected in the uploaded image.")
                        st.session_state.processing = False

            if 'last_analysis' in st.session_state and st.session_state.last_analysis.get('source_type') == 'upload_image':
                res = st.session_state.last_analysis
                if uploaded_image and res['filename'] == uploaded_image.name:
                    st.markdown("---"); st.subheader("Analysis Result")
                    col1, col2 = st.columns(2)
                    with col1: st.image(res['file_bytes'])
                    with col2:
                        result_text = "REAL" if res['prediction'] == 1 else "FAKE"
                        if res['prediction'] == 1: st.success(f"**Result: {result_text}**")
                        else: st.error(f"**Result: {result_text}**")
                        st.metric(label="Confidence", value=f"{res['confidence']*100:.2f}%")
                        st.bar_chart(pd.DataFrame({'Category': ['Fake', 'Real'], 'Probability': res['probabilities']}).set_index('Category'))
                    st.markdown("---")
        
        with url_tab:
            st.subheader("Analyze Video from a URL")
            video_url = st.text_input("Enter a video URL")
            if st.button("Analyze URL") and video_url:
                clear_last_analysis()
                st.session_state.processing = True
                with st.spinner("Downloading video..."):
                    temp_video_path, video_title = download_video_from_url(video_url)
                if temp_video_path:
                    with st.spinner('Analyzing video...'):
                        video_tensor = validation_dataset(temp_video_path, sequence_length=20, transform=transform_list)[0]
                        prediction, confidence, probabilities = predict(model, video_tensor, device)
                        with open(temp_video_path, "rb") as f: video_bytes = f.read()
                        os.remove(temp_video_path)
                        user_id = db.get_user_id(st.session_state['username'])
                        if user_id:
                            result_text = "REAL" if prediction == 1 else "FAKE"
                            db.add_analysis_record(user_id, video_title, 'Video (URL)', result_text, confidence)
                            st.session_state.last_analysis = {'source_type': 'url', 'filename': video_title, 'file_bytes': video_bytes, 'prediction': prediction, 'confidence': confidence, 'probabilities': probabilities}
                            st.rerun()
            if 'last_analysis' in st.session_state and st.session_state.last_analysis.get('source_type') == 'url':
                res = st.session_state.last_analysis
                st.markdown("---"); st.subheader(f"Analysis Result for: *{res['filename']}*")
                col1, col2 = st.columns(2)
                with col1: st.video(res['file_bytes'])
                with col2:
                    result_text = "REAL" if res['prediction'] == 1 else "FAKE"
                    if res['prediction'] == 1: st.success(f"**Result: {result_text}**")
                    else: st.error(f"**Result: {result_text}**")
                    st.metric(label="Confidence", value=f"{res['confidence']*100:.2f}%")
                    st.bar_chart(pd.DataFrame({'Category': ['Fake', 'Real'], 'Probability': res['probabilities']}).set_index('Category'))
                st.markdown("---")

def login_register_page():
    set_login_background('test.jpg')
    
    st.markdown("<h1 style='text-align: center;'>🛡️ DeepFake Detection Engine</h1>", unsafe_allow_html=True)
    
    _, col2, _ = st.columns([1, 1.5, 1])
    
    with col2:
        login_tab, register_tab = st.tabs(["Login", "Create Account"])

        with login_tab:
            st.subheader("Welcome Back!")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type='password', placeholder="Enter your password")
                submitted = st.form_submit_button("Secure Login")
                
                if submitted:
                    user = db.get_user(username)
                    if user and check_password_hash(user['password'], password):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.session_state['is_admin'] = db.check_if_admin(username)
                        st.rerun()
                    else:
                        st.error("Incorrect username or password")

        with register_tab:
            st.subheader("Join Us!")
            with st.form("register_form"):
                new_username = st.text_input("New Username", placeholder="Choose a unique username")
                new_password = st.text_input("New Password", type='password', placeholder="Create a secure password")
                submitted = st.form_submit_button("Register Account")

                if submitted:
                    if not new_username or not new_password:
                        st.warning("Please fill out all fields.")
                    elif db.get_user(new_username):
                        st.warning("Username already exists. Please choose another one.")
                    else:
                        db.add_user(new_username, generate_password_hash(new_password))
                        st.success("Account created successfully! Please switch to the Login tab to sign in.")

    st.markdown(
        '<div class="login-footer">© 2025 Bhargavi. All rights reserved.</div>',
        unsafe_allow_html=True
    )

db.init_db()
local_css("style.css") # The base styles are still loaded
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if st.session_state.get('logged_in', False): 
    main_app()
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: grey;">© 2025 Bhargavi. All rights reserved.</div>',
        unsafe_allow_html=True
    )
else: 
    login_register_page()