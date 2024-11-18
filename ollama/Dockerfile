FROM ollama/ollama

ENV OLLAMA_MODELS=/app/volumes/ollama-jarvis

COPY cuda-archive-keyring.gpg /usr/share/keyrings
COPY cuda-ubuntu2204-x86_64.list /etc/apt/sources.list.d

RUN mkdir /app

WORKDIR /app

# Setup Nvidia CUDA Toolkit
RUN apt update
RUN apt install -y curl gpg
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
RUN curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
#RUN cp /app/cuda-archive-keyring.gpg /usr/share/keyrings
#RUN cp /app/cuda-ubuntu2204-x86_64.list /etc/apt/sources.list.d
RUN apt update
RUN apt install -y nvidia-container-toolkit libnvidia-ml-dev libnvidia-container-dev nvidia-utils-535 nano
RUN nvidia-ctk runtime configure --runtime=docker --cdi.enabled
# Set cgroups = false in /etc/nvidia-container-runtime/config.toml
RUN echo 'no-cgroups = false' | tee -a /etc/nvidia-container-runtime/config.toml
#RUN nvidia-ctk --debug cdi generate --output=/etc/cdi/nvidia.yaml

EXPOSE 11434

ENTRYPOINT [ "/app/ollama/start.sh" ]