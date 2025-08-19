# Dockerfile

# Gunakan base image Python yang ringan
FROM python:3.12-slim

# Instal git dan git-lfs di dalam kontainer
RUN apt-get update && apt-get install -y git git-lfs

# Definisikan build argument untuk menerima token dari Railway
ARG GIT_TOKEN

# Set direktori kerja
WORKDIR /app

# Konfigurasi Git untuk menggunakan token agar bisa mengakses repo private
# Baris ini secara aman menggunakan token tanpa menampilkannya di log
RUN git config --global url."https://oauth2:${GIT_TOKEN}@github.com/".insteadOf "https://github.com/"

# KlONING repositori di dalam container, BUKAN di-COPY
# GANTI DENGAN URL REPO ANDA. Tanda titik (.) di akhir berarti kloning ke direktori saat ini (/app)
RUN git clone https://github.com/fahrulrzi/network-intrusion-detection.git .

# Jalankan perintah penting ini untuk mengunduh file LFS yang sebenarnya
# Perintah ini sekarang akan berhasil karena kita berada di dalam repo Git.
RUN git lfs pull

# Instal semua dependency Python dari requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Perintah untuk menjalankan aplikasi ketika kontainer dimulai
CMD python -m gunicorn --workers 2 --bind 0.0.0.0:$PORT "app:create_app()"