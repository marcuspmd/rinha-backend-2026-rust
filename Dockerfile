# Stage 1: Build
FROM rust:1-slim-bookworm AS builder

# Instalar dependências de compilação necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    musl-tools \
    pkg-config \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Adicionar target musl de acordo com a arquitetura do builder
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        rustup target add x86_64-unknown-linux-musl; \
    elif [ "$ARCH" = "aarch64" ]; then \
        rustup target add aarch64-unknown-linux-musl; \
    fi

WORKDIR /usr/src/app

# Copiar todo o workspace do Rust e os recursos
COPY my-solution/ my-solution/
COPY resources/ resources/

# Compilar em modo release de acordo com a arquitetura detectada
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --manifest-path my-solution/Cargo.toml --release --target x86_64-unknown-linux-musl --bin api && \
        cp my-solution/target/x86_64-unknown-linux-musl/release/api /usr/src/app/api_binary; \
    elif [ "$ARCH" = "aarch64" ]; then \
        RUSTFLAGS="-C target-cpu=native" cargo build --manifest-path my-solution/Cargo.toml --release --target aarch64-unknown-linux-musl --bin api && \
        cp my-solution/target/aarch64-unknown-linux-musl/release/api /usr/src/app/api_binary; \
    fi

# Stage 2: Runtime (imagem scratch ultra-minimalista, sem OS/page cache)
FROM scratch

WORKDIR /app

# Copiar executável compilado estaticamente para musl
COPY --from=builder /usr/src/app/api_binary /app/api

# Copiar index binário pré-processado
COPY my-solution/index.bin /app/index.bin

# Copiar mcc_risk
COPY --from=builder /usr/src/app/resources/mcc_risk.json /app/resources/mcc_risk.json

EXPOSE 8080

CMD ["./api"]
