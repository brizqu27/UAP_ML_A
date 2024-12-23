import streamlit as st
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

# ---------------------------------
# Define CNN Model
# ---------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------------------------
# Fungsi Untuk Load Model
# ---------------------------------
@st.cache_resource
def load_model(filepath, model_type, num_classes):
    if model_type == "CNN":
        model = CNN(num_classes)
    elif model_type == "ResNet":
        from torchvision.models import resnet18
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Invalid model type. Choose either 'CNN' or 'ResNet'.")

    model.load_state_dict(torch.load(filepath, map_location=torch.device("cpu")))
    model.eval()
    return model

# ---------------------------------
# Transformations for input images
# ---------------------------------
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ---------------------------------
# Prediction function
# ---------------------------------
def predict_image(image, model, classes):
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return classes[predicted.item()]

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="Image Classification", page_icon="🖼️", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 20px;
        }
        .sidebar-title {
            font-size: 1.2rem;
            font-weight: bold;
        }
        .card {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            text-align: center;
            padding: 10px;
            margin: 10px;
        }
        .stApp {
            background: url('_21_keep-calm-and-love-me-wallpaper_Quotes-Keep-Calm-Allah-Wallpaper-Desktop-HD-Wallpaper-.jpg') no-repeat center center fixed;
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='main-title'>🖼️ Image Classification with CNN and ResNet</div>", unsafe_allow_html=True)
st.write("Upload images to classify them using pre-trained CNN and ResNet models.")

# Sidebar
st.sidebar.markdown("<div class='sidebar-title'>⚙️ Configuration</div>", unsafe_allow_html=True)
model_type = st.sidebar.selectbox("Choose Model Type:", ["CNN", "ResNet"])
model_path = st.sidebar.text_input("Enter Model File Path:", "")
class_names_path = st.sidebar.text_input("Enter Class Names File Path:", "")
uploaded_files = st.sidebar.file_uploader("Upload Image Files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Load class names
if class_names_path:
    try:
        with open(class_names_path, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
        st.sidebar.success("Class names loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading class names: {e}")
else:
    class_names = []

# Load model
if model_path and class_names:
    try:
        num_classes = len(class_names)
        model = load_model(model_path, model_type, num_classes)
        st.sidebar.success(f"{model_type} model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
else:
    model = None

# Display uploaded images and predictions
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("📤 **Unggah Gambar Destinasi**", type=["jpg", "jpeg", "png"])
    prediction = None
    if uploaded_file is not None:
        st.image(uploaded_file, caption='🖼️ **Gambar yang Diunggah**', use_column_width=True)
        image = Image.open(uploaded_file).convert("RGB")
        if st.button("🚀 **Prediksi Kategori**"):
            with st.spinner("🔄 Memproses gambar..."):
                prediction = predict_image(image, model, class_names)
            st.success(f"🎯 **Prediksi Model ({model_type}): {prediction}**")
            st.balloons()

with col2:
    st.markdown("### ℹ️ **Deskripsi Kategori:**")
    descriptions = {
        "buildings": "🏢 **Buildings:** Struktur bangunan yang megah dan arsitektur unik.",
        "forest": "🌳 **Forest:** Hutan dengan vegetasi yang lebat dan hijau.",
        "glacier": "🧊 **Glacier:** Pemandangan gletser es yang luas.",
        "mountain": "⛰️ **Mountain:** Pemandangan pegunungan yang menjulang tinggi.",
        "sea": "🌊 **Sea:** Pemandangan laut yang menakjubkan.",
        "street": "🛤️ **Street:** Jalanan dengan suasana urban yang khas."
    }
    if prediction:
        st.write(descriptions.get(prediction, "Tidak ada deskripsi untuk kategori ini."))
    else:
        st.info("Unggah gambar untuk melihat deskripsi kategori.")
