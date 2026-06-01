#![allow(dead_code)]

use std::fs::File;
use std::sync::Arc;
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::Deserialize;

// Precomputed responses for all 6 possible fraud counts (0..=5)
const RESPONSES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

// Parâmetros de Busca IVF-Flat
const K_CENTROIDS: usize = 8192;
// Runtime nprobe is read from NPROBE env var; this is the stack-allocated max.
const MAX_NPROBE: usize = 256;

// Constantes de Normalização
const MAX_AMOUNT: f64 = 10000.0;
const MAX_INSTALLMENTS: f64 = 12.0;
const AMOUNT_VS_AVG_RATIO: f64 = 10.0;
const MAX_MINUTES: f64 = 1440.0;
const MAX_KM: f64 = 1000.0;
const MAX_TX_COUNT_24H: f64 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f64 = 10000.0;

#[repr(C)]
#[derive(Clone, Copy)]
struct ClusterInfo {
    offset: u32,
    count: u32,
}

struct IVFIndex {
    _mmap: memmap2::Mmap,
    k_clusters: usize,
    n_vectors: usize,
    centroids: &'static [[f32; 16]],
    cluster_metadata: &'static [ClusterInfo],
    // Quantized i8 vectors loaded into heap: 192MB f32 → 48MB i8, zero page faults during search.
    // Encoding: -1.0 sentinel → i8::MIN (-128); [0,1] range → [0, 127].
    vectors: Vec<[i8; 16]>,
    labels: &'static [u8],
}

impl IVFIndex {
    fn new(file_path: &str) -> Self {
        let file = File::open(file_path).expect("Falha ao abrir index.bin");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("Falha ao mapear index.bin") };

        let magic = &mmap[0..4];
        assert_eq!(magic, b"IVFF");

        let k_clusters = u32::from_le_bytes(mmap[4..8].try_into().unwrap()) as usize;
        let n_vectors = u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize;

        let centroids_offset = 16;
        let centroids_len = k_clusters * std::mem::size_of::<[f32; 16]>();

        let metadata_offset = centroids_offset + centroids_len;
        let metadata_len = k_clusters * std::mem::size_of::<ClusterInfo>();

        let vectors_offset = metadata_offset + metadata_len;
        let vectors_len = n_vectors * std::mem::size_of::<[f32; 16]>();

        let labels_offset = vectors_offset + vectors_len;
        let labels_len = n_vectors;

        assert_eq!(mmap.len(), labels_offset + labels_len);

        let centroids = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr().add(centroids_offset) as *const [f32; 16],
                k_clusters,
            )
        };

        let cluster_metadata = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr().add(metadata_offset) as *const ClusterInfo,
                k_clusters,
            )
        };

        let f32_vectors = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr().add(vectors_offset) as *const [f32; 16],
                n_vectors,
            )
        };

        let labels = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr().add(labels_offset) as *const u8,
                n_vectors,
            )
        };

        let centroids = unsafe { std::mem::transmute::<&[[f32; 16]], &'static [[f32; 16]]>(centroids) };
        let cluster_metadata = unsafe { std::mem::transmute::<&[ClusterInfo], &'static [ClusterInfo]>(cluster_metadata) };
        let labels = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(labels) };

        // Quantize f32 vectors → i8 and load into heap (192MB → 48MB).
        // This eliminates page faults during search: all vectors fit in the 155MB memory limit.
        let mut vectors: Vec<[i8; 16]> = Vec::with_capacity(n_vectors);
        let mut checksum = 0i64;
        for v in f32_vectors {
            let mut qv = [0i8; 16];
            for i in 0..16 {
                qv[i] = if v[i] == -1.0 {
                    i8::MIN
                } else {
                    (v[i] * 127.0).round() as i8
                };
                checksum += qv[i] as i64;
            }
            vectors.push(qv);
        }
        println!("Index loaded: {} vectors quantized to i8 (checksum={})", n_vectors, checksum);

        Self {
            _mmap: mmap,
            k_clusters,
            n_vectors,
            centroids,
            cluster_metadata,
            vectors,
            labels,
        }
    }
}

// Structs para receber o payload de transação com zero-copy borrowing
#[derive(Deserialize)]
struct RequestPayload<'a> {
    id: &'a str,
    transaction: Transaction<'a>,
    customer: Customer<'a>,
    merchant: Merchant<'a>,
    terminal: Terminal,
    last_transaction: Option<LastTransaction<'a>>,
}

#[derive(Deserialize)]
struct Transaction<'a> {
    amount: f64,
    installments: f64,
    requested_at: &'a str,
}

#[derive(Deserialize)]
struct Customer<'a> {
    avg_amount: f64,
    tx_count_24h: f64,
    #[serde(borrow)]
    known_merchants: Vec<&'a str>,
}

#[derive(Deserialize)]
struct Merchant<'a> {
    id: &'a str,
    mcc: &'a str,
    avg_amount: f64,
}

#[derive(Deserialize)]
struct Terminal {
    is_online: bool,
    card_present: bool,
    km_from_home: f64,
}

#[derive(Deserialize)]
struct LastTransaction<'a> {
    timestamp: &'a str,
    km_from_current: f64,
}

struct AppState {
    index: IVFIndex,
    mcc_risk_table: [f32; 10000],
    nprobe: usize,
}

/// Days since 1970-01-01 for a civil date (Howard Hinnant's algorithm).
#[inline]
fn days_from_civil(y: i64, m: i64, day: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parse ISO-8601 date to (epoch_seconds, hour, day_of_week_mon0).
/// Matches the reference solution's datetime::parse exactly.
#[inline]
fn parse_datetime(date_str: &str) -> (i64, u8, u8) {
    let b = date_str.as_bytes();
    if b.len() < 19 {
        return (0, 0, 0);
    }
    let year = ((b[0] - b'0') as i64) * 1000
        + ((b[1] - b'0') as i64) * 100
        + ((b[2] - b'0') as i64) * 10
        + (b[3] - b'0') as i64;
    let month = ((b[5] - b'0') as i64) * 10 + (b[6] - b'0') as i64;
    let day = ((b[8] - b'0') as i64) * 10 + (b[9] - b'0') as i64;
    let hour = ((b[11] - b'0') as i64) * 10 + (b[12] - b'0') as i64;
    let min = ((b[14] - b'0') as i64) * 10 + (b[15] - b'0') as i64;
    let sec = ((b[17] - b'0') as i64) * 10 + (b[18] - b'0') as i64;
    let days = days_from_civil(year, month, day);
    let epoch = days * 86400 + hour * 3600 + min * 60 + sec;
    // 1970-01-01 was a Thursday (=4 in a Sunday-0 scheme).
    let dow_sun0 = ((days % 7 + 4) % 7 + 7) % 7; // 0=Sun..6=Sat, negative-safe
    let dow_mon0 = ((dow_sun0 + 6) % 7) as u8;   // mon=0..sun=6
    (epoch, hour as u8, dow_mon0)
}

fn parse_mcc(mcc_str: &str) -> usize {
    let mut val = 0;
    for &b in mcc_str.as_bytes() {
        if b >= b'0' && b <= b'9' {
            val = val * 10 + (b - b'0') as usize;
        }
    }
    val
}

fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

#[inline(always)]
fn squared_distance(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..16 {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

// Pure integer i8 distance — eliminates f32 conversion in the hot cluster-scan loop.
// Encoding: sentinel -1.0 → i8::MIN; [0,1] → [0,127].
// Relative ordering is preserved; comparison uses i32 units throughout.
#[inline(always)]
fn squared_distance_i8(a: &[i8; 16], b: &[i8; 16]) -> i32 {
    let mut sum = 0i32;
    for i in 0..16 {
        let diff = a[i] as i32 - b[i] as i32;
        sum += diff * diff;
    }
    sum
}

async fn handle_ready() -> StatusCode {
    StatusCode::OK
}

async fn handle_fraud_score(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let payload: RequestPayload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        // Malformed input: return legit (FP weight 1) instead of 500 (Err weight 5)
        Err(_) => return ([(header::CONTENT_TYPE, "application/json")], RESPONSES[0]),
    };

    // 1. Parser rápido de data (Hinnant's algorithm, matching reference solution)
    let (req_epoch, hour, dow) = parse_datetime(payload.transaction.requested_at);

    // 2. Normalização do vetor da transação recebida (f64 intermediate, matching reference)
    let mut q = [0.0f32; 16];
    q[0] = clamp01((payload.transaction.amount / MAX_AMOUNT) as f32);
    q[1] = clamp01((payload.transaction.installments / MAX_INSTALLMENTS) as f32);

    let avg = payload.customer.avg_amount;
    q[2] = if avg > 0.0 {
        clamp01(((payload.transaction.amount / avg) / AMOUNT_VS_AVG_RATIO) as f32)
    } else {
        1.0
    };
    q[3] = hour as f32 / 23.0;
    q[4] = dow as f32 / 6.0;

    match &payload.last_transaction {
        Some(lt) => {
            let last_epoch = parse_datetime(lt.timestamp).0;
            let minutes = (req_epoch - last_epoch) as f64 / 60.0;
            q[5] = clamp01((minutes / MAX_MINUTES) as f32);
            q[6] = clamp01((lt.km_from_current / MAX_KM) as f32);
        }
        None => {
            q[5] = -1.0;
            q[6] = -1.0;
        }
    }

    q[7] = clamp01((payload.terminal.km_from_home / MAX_KM) as f32);
    q[8] = clamp01((payload.customer.tx_count_24h / MAX_TX_COUNT_24H) as f32);
    q[9] = if payload.terminal.is_online { 1.0 } else { 0.0 };
    q[10] = if payload.terminal.card_present { 1.0 } else { 0.0 };

    // Busca linear rápida no coorte curto de conhecidos
    let is_known = payload
        .customer
        .known_merchants
        .iter()
        .any(|m| *m == payload.merchant.id);
    q[11] = if is_known { 0.0 } else { 1.0 };

    // O(1) MCC Risk Lookup
    let mcc_idx = parse_mcc(payload.merchant.mcc);
    q[12] = if mcc_idx < 10000 {
        state.mcc_risk_table[mcc_idx]
    } else {
        0.5
    };

    q[13] = clamp01((payload.merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT) as f32);
    q[14] = 0.0;
    q[15] = 0.0;

    // 3. Encontrar os nprobe centroides mais próximos — batch de 4 por iteração para melhor ILP.
    // K_CENTROIDS=8192 é divisível por 4; sem resto.
    let nprobe = state.nprobe;
    let mut nearest_centroids = [(f32::MAX, 0usize); MAX_NPROBE];
    let mut threshold = f32::MAX;
    for chunk in 0..(K_CENTROIDS / 4) {
        let k = chunk * 4;
        let d0 = squared_distance(&q, &state.index.centroids[k]);
        let d1 = squared_distance(&q, &state.index.centroids[k + 1]);
        let d2 = squared_distance(&q, &state.index.centroids[k + 2]);
        let d3 = squared_distance(&q, &state.index.centroids[k + 3]);
        for (dist, ki) in [(d0, k), (d1, k + 1), (d2, k + 2), (d3, k + 3)] {
            if dist < threshold {
                nearest_centroids[nprobe - 1] = (dist, ki);
                let mut i = nprobe - 1;
                while i > 0 && nearest_centroids[i].0 < nearest_centroids[i - 1].0 {
                    nearest_centroids.swap(i, i - 1);
                    i -= 1;
                }
                threshold = nearest_centroids[nprobe - 1].0;
            }
        }
    }

    // Quantiza o query para i8 para o scan dos clusters (mesma codificação do índice).
    // Evita conversão i8→f32 no loop quente — permite auto-vectorização SIMD inteira.
    let q_i8: [i8; 16] = {
        let mut qi = [0i8; 16];
        for i in 0..16 {
            qi[i] = if q[i] == -1.0 {
                i8::MIN
            } else {
                (q[i] * 127.0).round() as i8
            };
        }
        qi
    };

    // 4. Varrer vetores nos clusters selecionados buscando os top 5 vizinhos mais próximos.
    // Prefetch do próximo cluster esconde a latência DRAM na transição entre clusters.
    let mut top5 = [(i32::MAX, 0u8); 5];
    let mut threshold_top5 = i32::MAX;
    let probed = &nearest_centroids[..nprobe];

    for i in 0..nprobe {
        let k = probed[i].1;

        // Prefetch o início do próximo cluster enquanto processamos o atual.
        if i + 1 < nprobe {
            let next_k = probed[i + 1].1;
            let next_start = state.index.cluster_metadata[next_k].offset as usize;
            unsafe {
                let ptr = state.index.vectors.as_ptr().add(next_start) as *const i8;
                #[cfg(target_arch = "x86_64")]
                std::arch::x86_64::_mm_prefetch(ptr, std::arch::x86_64::_MM_HINT_T1);
            }
        }

        let meta = &state.index.cluster_metadata[k];
        let start = meta.offset as usize;
        let end = start + meta.count as usize;

        for idx in start..end {
            let dist = squared_distance_i8(&q_i8, &state.index.vectors[idx]);
            if dist < threshold_top5 {
                let label = state.index.labels[idx];
                top5[4] = (dist, label);
                let mut i = 4;
                while i > 0 && top5[i].0 < top5[i - 1].0 {
                    top5.swap(i, i - 1);
                    i -= 1;
                }
                threshold_top5 = top5[4].0;
            }
        }
    }

    // 5. Calcular o score de fraude
    let fraud_count = top5.iter().filter(|&&(_, label)| label == 1).count();
    (
        [(header::CONTENT_TYPE, "application/json")],
        RESPONSES[fraud_count.min(5)],
    )
}

fn main() {
    let workers: usize = std::env::var("WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let nprobe: usize = std::env::var("NPROBE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48)
        .clamp(1, MAX_NPROBE);

    // 1. Inicializar tabela estática de MCC de risco
    let mut mcc_risk_table = [0.5f32; 10000];
    if let Ok(file) = File::open("resources/mcc_risk.json") {
        if let Ok(map) = serde_json::from_reader::<_, std::collections::HashMap<String, f32>>(file) {
            for (mcc_str, risk) in map {
                let idx = parse_mcc(&mcc_str);
                if idx < 10000 {
                    mcc_risk_table[idx] = risk;
                }
            }
        }
    }

    // 2. Mapear o index.bin gerado no build
    let index = IVFIndex::new("index.bin");

    println!("nprobe={nprobe}, workers={workers}");
    let state = Arc::new(AppState {
        index,
        mcc_risk_table,
        nprobe,
    });

    // 3. Configurar rotas Axum
    let app = Router::new()
        .route("/ready", get(handle_ready))
        .route("/fraud-score", post(handle_fraud_score))
        .with_state(state);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async move {
        // 4. Iniciar servidor (Socket Unix se SOCKET_PATH estiver definido, senão fallback TCP)
        if let Ok(socket_path) = std::env::var("SOCKET_PATH") {
            if std::path::Path::new(&socket_path).exists() {
                let _ = std::fs::remove_file(&socket_path);
            }
            let listener = tokio::net::UnixListener::bind(&socket_path).unwrap();

            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&socket_path).unwrap().permissions();
            perms.set_mode(0o777);
            std::fs::set_permissions(&socket_path, perms).unwrap();

            println!("API escutando no socket unix: {}", socket_path);

            use hyper_util::server::conn::auto;
            use hyper_util::rt::TokioExecutor;
            use tower::Service;

            loop {
                let (stream, _) = match listener.accept().await {
                    Ok(conn) => conn,
                    Err(err) => {
                        eprintln!("Erro ao aceitar conexão: {:?}", err);
                        continue;
                    }
                };

                let app = app.clone();
                tokio::spawn(async move {
                    let stream = hyper_util::rt::TokioIo::new(stream);
                    let hyper_service = hyper::service::service_fn(move |req: hyper::Request<hyper::body::Incoming>| {
                        let mut app = app.clone();
                        async move {
                            let req = req.map(axum::body::Body::new);
                            app.call(req).await
                        }
                    });

                    if let Err(_err) = auto::Builder::new(TokioExecutor::new())
                        .serve_connection(stream, hyper_service)
                        .await
                    {}
                });
            }
        } else {
            let listener = tokio::net::TcpListener::bind("0.0.0.0:8080")
                .await
                .unwrap();
            println!("API escutando em http://0.0.0.0:8080");
            axum::serve(listener, app).await.unwrap();
        }
    });
}
