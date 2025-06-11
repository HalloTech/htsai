# ---------- Base image with CUDA 12.1 and Python 3.10 ----------
    FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

    # ---------- System setup ----------
    ENV DEBIAN_FRONTEND=noninteractive
    
    RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git curl unzip libgl1-mesa-glx libglib2.0-0 ffmpeg \
        && rm -rf /var/lib/apt/lists/*
    
    RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
    RUN python -m pip install --upgrade pip
    
    # ---------- Set working directory ----------
    WORKDIR /app
    
    # ---------- Copy code and requirements ----------
    COPY requirements.txt .
    
    # ---------- Install Python dependencies ----------
    RUN pip install --no-cache-dir -r requirements.txt
    
    # ---------- Copy all source code ----------
    COPY . .
    
    # ---------- Expose Flask port ----------
    EXPOSE 5000
    
    # ---------- Run Flask app ----------
    CMD ["python", "app.py"]