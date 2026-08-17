FROM golang:1.26-bookworm AS proxy-builder
WORKDIR /src
COPY go/go.mod go/main.go ./
RUN GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go build -o /out/move_proxy .

FROM anaconda/miniconda:26.3.2

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    less \
    openssl \
    rsync \
    unzip \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
COPY requirements.txt /tmp/requirements.txt
# pip, not conda, resolves requirements.txt: Anaconda's pandas 3.0.3 metadata
# caps pyarrow <24, rejecting a combination pip installs fine (and prod runs).
RUN conda create -y -n "MoVE" python=3.14 && \
    conda run -n MoVE pip install --no-cache-dir -r /tmp/requirements.txt

# Self-signed TLS certificate for the Go reverse proxy. Generated at build
# time (not committed to git) so each image gets its own private key.
# Internal use only — browsers will warn unless this cert is separately
# trusted (e.g. installed via mkcert on lab machines).
# Lives under /move rather than /etc/move: LSF's docker wrapper on the
# compute nodes runs the container as the submitting user's (non-root) UID,
# which ran into permission-denied reading files under /etc. World-readable
# perms below are explicit so that works regardless of runtime UID.
RUN mkdir -p /move/certs && \
    openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
        -keyout /move/certs/server.key \
        -out /move/certs/server.crt \
        -subj "/CN=move-internal" \
        -addext "subjectAltName=DNS:*.ris.wustl.edu,DNS:localhost,IP:127.0.0.1" && \
    chmod 755 /move /move/certs && \
    chmod 644 /move/certs/server.key /move/certs/server.crt

COPY --from=proxy-builder /out/move_proxy /usr/local/bin/move_proxy
