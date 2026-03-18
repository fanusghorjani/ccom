# DS01 Custom Image: ccom
# Created: Thu 12 Mar 2026 01:39:45 PM CET
# Framework: pytorch
# Use case: nlp
# Author: 248835@hertie-school.lan

FROM aimehub/pytorch-2.8.0-aime-cuda12.6.3

# DS01 metadata labels
LABEL maintainer="248835@hertie-school.lan"
LABEL maintainer.id="1722834247"
LABEL ds01.project="ccom"
LABEL ds01.framework="pytorch"
LABEL ds01.created="2026-03-12T13:39:45+01:00"
LABEL ds01.managed="true"

# Build arguments (set automatically by DS01)
ARG DS01_USER_ID=1722834247
ARG DS01_GROUP_ID=1722800513
ARG DS01_USERNAME=248835@hertie-school.lan

WORKDIR /workspace

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Jupyter packages skipped by user

# Core Data Science
RUN pip install --no-cache-dir \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn

# Use case specific packages
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    tokenizers \
    accelerate \
    sentencepiece \
    peft \
    safetensors \
    evaluate

# Custom additional packages
RUN pip install --no-cache-dir \
    faiss-cpu \
    sentence-transformers

# User/Group Setup (DS01 - baked into image to avoid docker commit at container creation)
ARG DS01_USER_ID
ARG DS01_GROUP_ID
ARG DS01_USERNAME

# Create user/group with specific UID:GID matching host user
RUN set -e && \
    if [ -n "$DS01_USER_ID" ] && [ -n "$DS01_GROUP_ID" ] && [ -n "$DS01_USERNAME" ]; then \
        echo "DS01: Setting up user $DS01_USERNAME (UID=$DS01_USER_ID, GID=$DS01_GROUP_ID)" && \
        # Step 1: Remove conflicting group if exists with different name
        EXISTING_GROUP=$(getent group $DS01_GROUP_ID 2>/dev/null | cut -d: -f1 || true) && \
        if [ -n "$EXISTING_GROUP" ] && [ "$EXISTING_GROUP" != "$DS01_USERNAME" ]; then \
            echo "DS01: Removing conflicting group $EXISTING_GROUP (has GID $DS01_GROUP_ID)" && \
            groupdel "$EXISTING_GROUP" 2>/dev/null || true; \
        fi && \
        # Step 2: Create group with specific GID (cascade handles numeric-only names)
        # Some distros (Debian) reject purely numeric group names even with groupadd
        # Note: uses variable guard instead of "if !" to avoid set -e swallowing cascade exits
        DS01_HAVE_GROUP=0 && \
        getent group $DS01_GROUP_ID >/dev/null 2>&1 && DS01_HAVE_GROUP=1 || true && \
        if [ "$DS01_HAVE_GROUP" = "0" ]; then \
            groupadd -g $DS01_GROUP_ID $DS01_USERNAME 2>/dev/null || \
            groupadd -g $DS01_GROUP_ID ds01_$DS01_USERNAME 2>/dev/null || true; \
        fi && \
        # Step 3: Remove conflicting user if exists with different name
        EXISTING_USER=$(getent passwd $DS01_USER_ID 2>/dev/null | cut -d: -f1 || true) && \
        if [ -n "$EXISTING_USER" ] && [ "$EXISTING_USER" != "$DS01_USERNAME" ]; then \
            echo "DS01: Removing conflicting user $EXISTING_USER (has UID $DS01_USER_ID)" && \
            userdel -r "$EXISTING_USER" 2>/dev/null || true; \
        fi && \
        # Step 4: Create user with specific UID and GID (cascade handles numeric-only names)
        # Try --badname first (newer shadow-utils), fall back to plain useradd
        DS01_HAVE_USER=0 && \
        getent passwd $DS01_USER_ID >/dev/null 2>&1 && DS01_HAVE_USER=1 || true && \
        if [ "$DS01_HAVE_USER" = "0" ]; then \
            useradd --badname -m -d /home/$DS01_USERNAME -u $DS01_USER_ID -g $DS01_GROUP_ID -s /bin/bash $DS01_USERNAME 2>/dev/null || \
            useradd -m -d /home/$DS01_USERNAME -u $DS01_USER_ID -g $DS01_GROUP_ID -s /bin/bash $DS01_USERNAME; \
        fi && \
        # Step 4b: CRITICAL - Truncate lastlog/faillog to prevent huge sparse files
        # High UIDs (e.g., 1722830498) cause these files to grow to 300GB+
        # which breaks Docker overlay2 layer export
        : > /var/log/lastlog && \
        : > /var/log/faillog && \
        # Step 5: Configure sudo access (NOPASSWD)
        usermod -aG sudo $DS01_USERNAME && \
        echo "$DS01_USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
        # Step 6: Create .local/bin directory for pip --user installs
        mkdir -p /home/$DS01_USERNAME/.local/bin && \
        chown -R $DS01_USER_ID:$DS01_GROUP_ID /home/$DS01_USERNAME && \
        # Step 7: Configure PATH in bashrc (matches AIME docker run setup)
        # Add ~/.local/bin to PATH for pip install --user packages
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/$DS01_USERNAME/.bashrc && \
        # Step 8: Configure HOME export (fixes LDAP username issues)
        echo "export HOME=/home/$DS01_USERNAME" >> /home/$DS01_USERNAME/.bashrc && \
        # Step 9: Configure PS1 prompt (matches AIME docker run setup)
        # Uses container hostname which Docker sets from --name or container ID
        # Format: [container-name] user@host:path$
        echo 'export PS1='"'"'[\h] \u@ds01:\w\$ '"'"'' >> /home/$DS01_USERNAME/.bashrc && \
        chown $DS01_USER_ID:$DS01_GROUP_ID /home/$DS01_USERNAME/.bashrc && \
        echo "DS01: User setup complete"; \
    else \
        echo "DS01: Skipping user setup (build args not provided)"; \
    fi

# DS01 Labels for container creation optimization
LABEL ds01.has_user_setup="true"
LABEL ds01.user_id="1722834247"
LABEL ds01.group_id="1722800513"
LABEL ds01.username="248835"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch
ENV MPLCONFIGDIR=/workspace/.cache/matplotlib

CMD ["/bin/bash"]
