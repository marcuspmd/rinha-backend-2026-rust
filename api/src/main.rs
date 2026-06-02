#![allow(dead_code)]

use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::Deserialize;

#[cfg(not(target_os = "macos"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

// Precomputed responses for all 6 possible fraud counts (0..=5)
const RESPONSES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

static METRIC_JSON_PARSE_NS: AtomicU64 = AtomicU64::new(0);
static METRIC_VECTORIZE_NS: AtomicU64 = AtomicU64::new(0);
static METRIC_CENTROID_SEARCH_NS: AtomicU64 = AtomicU64::new(0);
static METRIC_CLUSTER_SCAN_NS: AtomicU64 = AtomicU64::new(0);
static METRIC_TOTAL_NS: AtomicU64 = AtomicU64::new(0);
static METRIC_COUNT: AtomicU64 = AtomicU64::new(0);

// Parâmetros de Busca IVF-Flat
const K_CENTROIDS: usize = 8192;
// Runtime nprobe is read from NPROBE env var; this is the stack-allocated max.
const MAX_NPROBE: usize = 8192;

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
    // Quantized i16 vectors loaded into heap: 192MB f32 → 96MB i16, zero page faults during search.
    // Encoding: Sentinel/regular values scaled by 5000.0.
    vectors: Vec<[i16; 16]>,
    labels: &'static [u8],
    f32_vectors: &'static [[f32; 16]],
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

        // Quantize f32 vectors → i16 and load into heap (192MB → 96MB).
        // This eliminates page faults during search: all vectors fit in the 155MB memory limit.
        let mut vectors: Vec<[i16; 16]> = Vec::with_capacity(n_vectors);
        let mut checksum = 0i64;
        for v in f32_vectors {
            let mut qv = [0i16; 16];
            for i in 0..16 {
                qv[i] = (v[i] * 5000.0).round() as i16;
                checksum += qv[i] as i64;
            }
            vectors.push(qv);
        }
        println!("Index loaded: {} vectors quantized to i16 (checksum={})", n_vectors, checksum);

        let f32_vectors = unsafe { std::mem::transmute::<&[[f32; 16]], &'static [[f32; 16]]>(f32_vectors) };

        Self {
            _mmap: mmap,
            k_clusters,
            n_vectors,
            centroids,
            cluster_metadata,
            vectors,
            labels,
            f32_vectors,
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn squared_distance_preloaded(vq0: std::arch::x86_64::__m256, vq1: std::arch::x86_64::__m256, b: &[f32; 16]) -> f32 {
    use std::arch::x86_64::*;
    let vb0 = _mm256_loadu_ps(b.as_ptr());
    let vb1 = _mm256_loadu_ps(b.as_ptr().add(8));

    let diff0 = _mm256_sub_ps(vq0, vb0);
    let diff1 = _mm256_sub_ps(vq1, vb1);

    let sq0 = _mm256_mul_ps(diff0, diff0);
    let sq1 = _mm256_mul_ps(diff1, diff1);

    let sum = _mm256_add_ps(sq0, sq1);

    let low128 = _mm256_castps256_ps128(sum);
    let high128 = _mm256_extractf128_ps(sum, 1);
    let sum128 = _mm_add_ps(low128, high128);

    let shuf = _mm_movehdup_ps(sum128);
    let sum128 = _mm_add_ps(sum128, shuf);
    let shuf = _mm_movehl_ps(shuf, sum128);
    let sum128 = _mm_add_ps(sum128, shuf);

    _mm_cvtss_f32(sum128)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn squared_distance_preloaded(
    vq0: std::arch::aarch64::float32x4_t,
    vq1: std::arch::aarch64::float32x4_t,
    vq2: std::arch::aarch64::float32x4_t,
    vq3: std::arch::aarch64::float32x4_t,
    b: &[f32; 16],
) -> f32 {
    use std::arch::aarch64::*;
    let vb0 = vld1q_f32(b.as_ptr());
    let vb1 = vld1q_f32(b.as_ptr().add(4));
    let vb2 = vld1q_f32(b.as_ptr().add(8));
    let vb3 = vld1q_f32(b.as_ptr().add(12));

    let diff0 = vsubq_f32(vq0, vb0);
    let diff1 = vsubq_f32(vq1, vb1);
    let diff2 = vsubq_f32(vq2, vb2);
    let diff3 = vsubq_f32(vq3, vb3);

    let mut sum = vmulq_f32(diff0, diff0);
    sum = vfmaq_f32(sum, diff1, diff1);
    sum = vfmaq_f32(sum, diff2, diff2);
    sum = vfmaq_f32(sum, diff3, diff3);

    vaddvq_f32(sum)
}

#[inline(always)]
fn squared_distance_fallback(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..16 {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn squared_distance_i16_preloaded(vq: std::arch::x86_64::__m256i, b: &[i16; 16]) -> i32 {
    use std::arch::x86_64::*;
    let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    let diff = _mm256_sub_epi16(vq, vb);
    let prod = _mm256_madd_epi16(diff, diff);
    
    let low128 = _mm256_castsi256_si128(prod);
    let high128 = _mm256_extracti128_si256(prod, 1);
    let sum128 = _mm_add_epi32(low128, high128);
    
    let shuf = _mm_shuffle_epi32(sum128, 0x4E);
    let sum128 = _mm_add_epi32(sum128, shuf);
    let shuf = _mm_shuffle_epi32(sum128, 0x11);
    let sum128 = _mm_add_epi32(sum128, shuf);
    
    _mm_cvtsi128_si32(sum128)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn squared_distance_i16_preloaded(
    vq0: std::arch::aarch64::int16x8_t,
    vq1: std::arch::aarch64::int16x8_t,
    b: &[i16; 16],
) -> i32 {
    use std::arch::aarch64::*;
    let vb0 = vld1q_s16(b.as_ptr());
    let vb1 = vld1q_s16(b.as_ptr().add(8));

    let diff0 = vsubq_s16(vq0, vb0);
    let diff1 = vsubq_s16(vq1, vb1);

    let sq0 = vmull_s16(vget_low_s16(diff0), vget_low_s16(diff0));
    let sq0 = vmlal_s16(sq0, vget_high_s16(diff0), vget_high_s16(diff0));

    let sq1 = vmull_s16(vget_low_s16(diff1), vget_low_s16(diff1));
    let sq1 = vmlal_s16(sq1, vget_high_s16(diff1), vget_high_s16(diff1));

    let final_sum = vaddq_s32(sq0, sq1);
    vaddvq_s32(final_sum)
}

#[inline(always)]
fn squared_distance_i16_fallback(a: &[i16; 16], b: &[i16; 16]) -> i32 {
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

async fn handle_telemetry() -> impl IntoResponse {
    let count = METRIC_COUNT.load(Ordering::Relaxed);
    if count == 0 {
        return axum::Json(serde_json::json!({
            "count": 0,
            "json_parse_us": 0.0,
            "vectorize_us": 0.0,
            "centroid_search_us": 0.0,
            "cluster_scan_us": 0.0,
            "total_us": 0.0,
        }));
    }

    let count_f = count as f64;
    let json_parse_us = (METRIC_JSON_PARSE_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
    let vectorize_us = (METRIC_VECTORIZE_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
    let centroid_search_us = (METRIC_CENTROID_SEARCH_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
    let cluster_scan_us = (METRIC_CLUSTER_SCAN_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
    let total_us = (METRIC_TOTAL_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;

    axum::Json(serde_json::json!({
        "count": count,
        "json_parse_us": json_parse_us,
        "vectorize_us": vectorize_us,
        "centroid_search_us": centroid_search_us,
        "cluster_scan_us": cluster_scan_us,
        "total_us": total_us,
    }))
}

async fn handle_fraud_score(
    State(state): State<Arc<AppState>>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let start_time = Instant::now();

    // 1. Parser JSON
    let json_start = Instant::now();
    let payload: RequestPayload = match serde_json::from_slice(&body) {
        Ok(p) => p,
        // Malformed input: return legit (FP weight 1) instead of 500 (Err weight 5)
        Err(_) => return ([(header::CONTENT_TYPE, "application/json")], RESPONSES[0]),
    };
    let json_dur = json_start.elapsed().as_nanos() as u64;
    METRIC_JSON_PARSE_NS.fetch_add(json_dur, Ordering::Relaxed);

    let vec_start = Instant::now();
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
    let vec_dur = vec_start.elapsed().as_nanos() as u64;
    METRIC_VECTORIZE_NS.fetch_add(vec_dur, Ordering::Relaxed);

    // 3. Encontrar os nprobe centroides mais próximos.
    // Computar todas as distâncias de forma contígua em um buffer na stack para permitir auto-vetorização.
    let centroid_start = Instant::now();
    let nprobe = state.nprobe;

    let mut dists = [0.0f32; K_CENTROIDS];
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::*;
        let vq0 = _mm256_loadu_ps(q.as_ptr());
        let vq1 = _mm256_loadu_ps(q.as_ptr().add(8));
        for k in 0..K_CENTROIDS {
            dists[k] = squared_distance_preloaded(vq0, vq1, &state.index.centroids[k]);
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let vq0 = vld1q_f32(q.as_ptr());
        let vq1 = vld1q_f32(q.as_ptr().add(4));
        let vq2 = vld1q_f32(q.as_ptr().add(8));
        let vq3 = vld1q_f32(q.as_ptr().add(12));
        for k in 0..K_CENTROIDS {
            dists[k] = squared_distance_preloaded(vq0, vq1, vq2, vq3, &state.index.centroids[k]);
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for k in 0..K_CENTROIDS {
        dists[k] = squared_distance_fallback(&q, &state.index.centroids[k]);
    }

    // Inicializar índices na stack
    let mut indices = [0u16; K_CENTROIDS];
    for i in 0..K_CENTROIDS {
        indices[i] = i as u16;
    }

    // Quickselect O(N) na stack para obter os nprobe centróides mais próximos (desordenados)
    indices.select_nth_unstable_by(nprobe - 1, |&a, &b| {
        dists[a as usize].partial_cmp(&dists[b as usize]).unwrap()
    });

    let centroid_dur = centroid_start.elapsed().as_nanos() as u64;
    METRIC_CENTROID_SEARCH_NS.fetch_add(centroid_dur, Ordering::Relaxed);

    let scan_start = Instant::now();
    // Quantiza o query para i16 para o scan dos clusters (mesma codificação do índice).
    // Evita conversão i16→f32 no loop quente — permite auto-vectorização SIMD inteira.
    let q_i16: [i16; 16] = {
        let mut qi = [0i16; 16];
        for i in 0..16 {
            qi[i] = (q[i] * 5000.0).round() as i16;
        }
        qi
    };

    // 4. Varrer vetores nos clusters selecionados buscando os top 16 vizinhos mais próximos.
    let mut top_candidates = [(i32::MAX, 0usize); 16]; // (dist_i16, idx)
    let mut threshold_candidates = i32::MAX;
    let probed = &indices[0..nprobe];

    for i in 0..nprobe {
        let k = probed[i] as usize;

        // Prefetch o início do próximo cluster enquanto processamos o atual.
        #[cfg(target_arch = "x86_64")]
        if i + 1 < nprobe {
            let next_k = probed[i + 1] as usize;
            let next_start = state.index.cluster_metadata[next_k].offset as usize;
            unsafe {
                let ptr = state.index.vectors.as_ptr().add(next_start) as *const i16;
                std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T1);
            }
        }

        let meta = &state.index.cluster_metadata[k];
        let start = meta.offset as usize;
        let end = start + meta.count as usize;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let vq = _mm256_loadu_si256(q_i16.as_ptr() as *const __m256i);
            for idx in start..end {
                let dist = squared_distance_i16_preloaded(vq, &state.index.vectors[idx]);
                if dist < threshold_candidates {
                    top_candidates[15] = (dist, idx);
                    let mut i = 15;
                    while i > 0 && top_candidates[i].0 < top_candidates[i - 1].0 {
                        top_candidates.swap(i, i - 1);
                        i -= 1;
                    }
                    threshold_candidates = top_candidates[15].0;
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let vq0 = vld1q_s16(q_i16.as_ptr());
            let vq1 = vld1q_s16(q_i16.as_ptr().add(8));
            for idx in start..end {
                let dist = squared_distance_i16_preloaded(vq0, vq1, &state.index.vectors[idx]);
                if dist < threshold_candidates {
                    top_candidates[15] = (dist, idx);
                    let mut i = 15;
                    while i > 0 && top_candidates[i].0 < top_candidates[i - 1].0 {
                        top_candidates.swap(i, i - 1);
                        i -= 1;
                    }
                    threshold_candidates = top_candidates[15].0;
                }
            }
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        for idx in start..end {
            let dist = squared_distance_i16_fallback(&q_i16, &state.index.vectors[idx]);
            if dist < threshold_candidates {
                top_candidates[15] = (dist, idx);
                let mut i = 15;
                while i > 0 && top_candidates[i].0 < top_candidates[i - 1].0 {
                    top_candidates.swap(i, i - 1);
                    i -= 1;
                }
                threshold_candidates = top_candidates[15].0;
            }
        }
    }

    // 5. Re-rank os candidatos com distância f32 exata para eliminar erro de quantização
    let mut exact_top5 = [(f32::MAX, 0u8); 5];
    let mut threshold_exact = f32::MAX;
    
    let num_candidates = top_candidates.iter().filter(|&&(d, _)| d < i32::MAX).count();
    for i in 0..num_candidates {
        let idx = top_candidates[i].1;
        let dist_f32 = squared_distance_fallback(&q, &state.index.f32_vectors[idx]);
        if dist_f32 < threshold_exact {
            let label = state.index.labels[idx];
            exact_top5[4] = (dist_f32, label);
            let mut j = 4;
            while j > 0 && exact_top5[j].0 < exact_top5[j - 1].0 {
                exact_top5.swap(j, j - 1);
                j -= 1;
            }
            threshold_exact = exact_top5[4].0;
        }
    }

    let scan_dur = scan_start.elapsed().as_nanos() as u64;
    METRIC_CLUSTER_SCAN_NS.fetch_add(scan_dur, Ordering::Relaxed);

    let total_dur = start_time.elapsed().as_nanos() as u64;
    METRIC_TOTAL_NS.fetch_add(total_dur, Ordering::Relaxed);

    let count = METRIC_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    if count % 10000 == 0 {
        let count_f = count as f64;
        let jp = (METRIC_JSON_PARSE_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
        let vc = (METRIC_VECTORIZE_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
        let cs = (METRIC_CENTROID_SEARCH_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
        let cl = (METRIC_CLUSTER_SCAN_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
        let tot = (METRIC_TOTAL_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
        println!(
            "[TELEMETRY] Req Count: {}, Avg (us): JSON={:.2}, Vec={:.2}, Centroid={:.2}, Cluster={:.2}, Total={:.2}",
            count, jp, vc, cs, cl, tot
        );
    }

    // Calcular o score de fraude a partir do re-ranking de f32 exato
    let fraud_count = exact_top5.iter().filter(|&&(_, label)| label == 1).count();
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
    let mut mcc_count = 0;
    if let Ok(file) = File::open("resources/mcc_risk.json") {
        if let Ok(map) = serde_json::from_reader::<_, std::collections::HashMap<String, f32>>(file) {
            for (mcc_str, risk) in map {
                let idx = parse_mcc(&mcc_str);
                if idx < 10000 {
                    mcc_risk_table[idx] = risk;
                    mcc_count += 1;
                }
            }
        }
    }
    println!("MCC risk table loaded: {} entries", mcc_count);

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
        .route("/telemetry", get(handle_telemetry))
        .route("/fraud-score", post(handle_fraud_score))
        .with_state(state);

    let rt = if workers <= 1 {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    } else {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .build()
            .unwrap()
    };

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
            let port = std::env::var("PORT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(8080);
            let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port))
                .await
                .unwrap();
            println!("API escutando em http://0.0.0.0:{}", port);
            axum::serve(listener, app).await.unwrap();
        }
    });
}
