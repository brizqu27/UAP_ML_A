import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# -------------------------------
# 🎯 Konfigurasi Halaman Streamlit
# -------------------------------
st.set_page_config(
    page_title="Klasifikasi Destinasi Wisata",
    page_icon="🏝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 📊 Daftar kelas
# -------------------------------
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# -------------------------------
# 🖌️ Transformasi gambar
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# -------------------------------
# 🛠️ Memuat Model CNN
# -------------------------------
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Load CNN Model
cnn_model = CNN(num_classes=len(CLASSES))
cnn_model.load_state_dict(torch.load('./models/cnn_model.pth', map_location=torch.device('cpu')))
cnn_model.eval()

# Load ResNet Model
resnet_model = models.resnet18(pretrained=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, len(CLASSES))
resnet_model.load_state_dict(torch.load('./models/resnet_model.pth', map_location=torch.device('cpu')))
resnet_model.eval()

# -------------------------------
# 🔍 Fungsi Prediksi
# -------------------------------
def predict_image(image, model):
    image = transform(image).unsqueeze(0)
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return CLASSES[predicted[0].item()]

# -------------------------------
# 🌟 Sidebar untuk Navigasi
# -------------------------------
st.sidebar.title("⚙️ **Pengaturan**")
st.sidebar.write("Pilih model yang akan digunakan untuk prediksi.")

model_choice = st.sidebar.radio("🔄 **Pilih Model:**", ["CNN", "ResNet"])
st.sidebar.write("📁 **Unggah gambar untuk memulai prediksi.**")
st.sidebar.markdown("---")
st.sidebar.write("🧠 **Dikembangkan oleh:**")
st.sidebar.write("**👤 Abd. Baasithur Rizqu - 202110370311241**")

# -------------------------------
# 🏞️ Halaman Utama
# -------------------------------
st.markdown("<h1 style='text-align: center;'>🏝️ Klasifikasi Destinasi Wisata</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Unggah gambar destinasi wisata dan pilih model untuk memprediksi kategorinya.</h4>", unsafe_allow_html=True)
st.markdown("---")

# Kolom untuk konten utama
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "📤 **Unggah Gambar Destinasi**", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption='🖼️ **Gambar yang Diunggah**', use_column_width=True, clamp=True)
        image = Image.open(uploaded_file).convert("RGB")
        
        if st.button("🚀 **Prediksi Kategori**"):
            with st.spinner("🔄 Memproses gambar..."):
                if model_choice == "CNN":
                    prediction = predict_image(image, cnn_model)
                else:
                    prediction = predict_image(image, resnet_model)
            
            st.success(f"🎯 **Prediksi Model ({model_choice}): {prediction}**")
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
    if uploaded_file:
        st.write(descriptions.get(prediction, "Tidak ada deskripsi untuk kategori ini."))
    else:
        st.info("Unggah gambar untuk melihat deskripsi kategori.")

# Footer
st.markdown("---")
st.caption("📚 **Aplikasi ini menggunakan model CNN dan ResNet untuk klasifikasi gambar destinasi wisata.**")

# Custom CSS for better styling
st.markdown("""
    <style>
        .stTest {
            background: url("https://wallpaperaccess.com/full/968829.jpg") no-repeat center center fixed;
            background-size: cover;
        }
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
    </style>
""", unsafe_allow_html=True)
