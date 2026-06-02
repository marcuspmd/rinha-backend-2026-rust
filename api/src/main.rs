#![allow(dead_code)]

use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use std::io::{Read, Write};
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

static IS_READY: AtomicBool = AtomicBool::new(false);

// Parâmetros de Busca IVF-Flat
const K_CENTROIDS: usize = 8192;
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
    radius: f32,
}

struct IVFIndex {
    _mmap: memmap2::Mmap,
    k_clusters: usize,
    n_vectors: usize,
    centroids: &'static [[f32; 16]],
    cluster_metadata: &'static [ClusterInfo],
    vectors: &'static [[f32; 16]],
    labels: &'static [u8],
    distances: &'static [f32],
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

        let distances_offset = labels_offset + labels_len;
        let distances_len = n_vectors * std::mem::size_of::<f32>();

        assert_eq!(mmap.len(), distances_offset + distances_len);

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

        let distances = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr().add(distances_offset) as *const f32,
                n_vectors,
            )
        };

        let centroids = unsafe { std::mem::transmute::<&[[f32; 16]], &'static [[f32; 16]]>(centroids) };
        let cluster_metadata = unsafe { std::mem::transmute::<&[ClusterInfo], &'static [ClusterInfo]>(cluster_metadata) };
        let f32_vectors = unsafe { std::mem::transmute::<&[[f32; 16]], &'static [[f32; 16]]>(f32_vectors) };
        let labels = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(labels) };
        let distances = unsafe { std::mem::transmute::<&[f32], &'static [f32]>(distances) };

        // Pretouch mmap to prevent page faults during benchmark
        let page_size = 4096;
        let mut dummy = 0u8;
        for offset in (0..mmap.len()).step_by(page_size) {
            dummy ^= mmap[offset];
        }
        println!("Mmap pretouch complete, dummy value = {}", dummy);
        println!("Index loaded: {} vectors mapped as f32 directly", n_vectors);

        Self {
            _mmap: mmap,
            k_clusters,
            n_vectors,
            centroids,
            cluster_metadata,
            vectors: f32_vectors,
            labels,
            distances,
        }
    }
}

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

#[inline]
fn days_from_civil(y: i64, m: i64, day: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

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
    let dow_sun0 = ((days % 7 + 4) % 7 + 7) % 7;
    let dow_mon0 = ((dow_sun0 + 6) % 7) as u8;
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

struct ParsedRequest {
    method: String,
    path: String,
    body_offset: usize,
    body_len: usize,
    keep_alive: bool,
}

fn parse_http(buf: &[u8]) -> Option<ParsedRequest> {
    let headers_end = buf.windows(4).position(|w| w == b"\r\n\r\n");
    let (header_len, body_start) = if let Some(pos) = headers_end {
        (pos, pos + 4)
    } else {
        if let Some(pos) = buf.windows(2).position(|w| w == b"\n\n") {
            (pos, pos + 2)
        } else {
            return None;
        }
    };

    let headers_part = &buf[..header_len];
    let mut lines = headers_part.split(|&b| b == b'\n');
    let first_line = lines.next()?;
    
    let mut parts = first_line.split(|&b| b == b' ');
    let method_bytes = parts.next()?;
    let path_bytes = parts.next()?;
    
    let method = std::str::from_utf8(method_bytes).ok()?.trim().to_string();
    let path = std::str::from_utf8(path_bytes).ok()?.trim().to_string();

    let mut content_length = 0;
    let mut keep_alive = true;

    for line in lines {
        let line_trimmed = if line.ends_with(b"\r") {
            &line[..line.len()-1]
        } else {
            line
        };
        if line_trimmed.is_empty() { continue; }

        if let Some(colon_pos) = line_trimmed.iter().position(|&b| b == b':') {
            let key = &line_trimmed[..colon_pos];
            let value = &line_trimmed[colon_pos+1..];
            
            let mut val_start = 0;
            while val_start < value.len() && (value[val_start] == b' ' || value[val_start] == b'\t') {
                val_start += 1;
            }
            let val_trimmed = &value[val_start..];

            if key.eq_ignore_ascii_case(b"content-length") {
                if let Ok(len_str) = std::str::from_utf8(val_trimmed) {
                    if let Ok(len) = len_str.parse::<usize>() {
                        content_length = len;
                    }
                }
            } else if key.eq_ignore_ascii_case(b"connection") {
                if val_trimmed.eq_ignore_ascii_case(b"close") {
                    keep_alive = false;
                }
            }
        }
    }

    if body_start + content_length > buf.len() {
        return None;
    }

    Some(ParsedRequest {
        method,
        path,
        body_offset: body_start,
        body_len: content_length,
        keep_alive,
    })
}

fn handle_connection<S: Read + Write>(
    mut stream: S,
    state: &Arc<AppState>,
    buf: &mut Vec<u8>
) -> std::io::Result<()> {
    buf.clear();
    let mut temp_buf = [0u8; 8192];
    
    loop {
        let n = match stream.read(&mut temp_buf) {
            Ok(0) => return Ok(()),
            Ok(size) => size,
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => {
                return Ok(());
            }
            Err(e) => return Err(e),
        };
        buf.extend_from_slice(&temp_buf[..n]);

        while let Some(req) = parse_http(buf) {
            let body_bytes = &buf[req.body_offset..req.body_offset + req.body_len];
            
            if req.path == "/fraud-score" && req.method == "POST" {
                let start_time = Instant::now();
                
                let json_start = Instant::now();
                let payload: RequestPayload = match serde_json::from_slice(body_bytes) {
                    Ok(p) => p,
                    Err(_) => {
                        let headers = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
                            RESPONSES[0].len(),
                            if req.keep_alive { "keep-alive" } else { "close" }
                        );
                        stream.write_all(headers.as_bytes())?;
                        stream.write_all(RESPONSES[0])?;
                        stream.flush()?;
                        
                        let consumed = req.body_offset + req.body_len;
                        buf.drain(0..consumed);
                        if !req.keep_alive { return Ok(()); }
                        continue;
                    }
                };
                let json_dur = json_start.elapsed().as_nanos() as u64;
                METRIC_JSON_PARSE_NS.fetch_add(json_dur, Ordering::Relaxed);

                let vec_start = Instant::now();
                let (req_epoch, hour, dow) = parse_datetime(payload.transaction.requested_at);

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

                let is_known = payload
                    .customer
                    .known_merchants
                    .iter()
                    .any(|m| *m == payload.merchant.id);
                q[11] = if is_known { 0.0 } else { 1.0 };

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

                let mut indices = [0u16; K_CENTROIDS];
                for i in 0..K_CENTROIDS {
                    indices[i] = i as u16;
                }
                indices.select_nth_unstable_by(nprobe - 1, |&a, &b| {
                    dists[a as usize].partial_cmp(&dists[b as usize]).unwrap()
                });

                let centroid_dur = centroid_start.elapsed().as_nanos() as u64;
                METRIC_CENTROID_SEARCH_NS.fetch_add(centroid_dur, Ordering::Relaxed);

                let scan_start = Instant::now();
                let mut top5 = [(f32::MAX, 0u8); 5];
                let mut threshold_top5 = f32::MAX;

                #[cfg(target_arch = "x86_64")]
                let vq0 = unsafe {
                    use std::arch::x86_64::*;
                    _mm256_loadu_ps(q.as_ptr())
                };
                #[cfg(target_arch = "x86_64")]
                let vq1 = unsafe {
                    use std::arch::x86_64::*;
                    _mm256_loadu_ps(q.as_ptr().add(8))
                };

                #[cfg(target_arch = "aarch64")]
                let (vq0, vq1, vq2, vq3) = unsafe {
                    use std::arch::aarch64::*;
                    (
                        vld1q_f32(q.as_ptr()),
                        vld1q_f32(q.as_ptr().add(4)),
                        vld1q_f32(q.as_ptr().add(8)),
                        vld1q_f32(q.as_ptr().add(12)),
                    )
                };

                let probed = &mut indices[0..nprobe];
                probed.sort_unstable_by(|&a, &b| {
                    dists[a as usize].partial_cmp(&dists[b as usize]).unwrap()
                });

                for &k_idx in probed.iter() {
                    let k = k_idx as usize;
                    let dist_q_c_sq = dists[k];
                    let dist_q_c = dist_q_c_sq.sqrt();
                    let meta = &state.index.cluster_metadata[k];

                    let threshold_top5_f32 = threshold_top5.sqrt();

                    if dist_q_c - meta.radius >= threshold_top5_f32 + 0.0002 {
                        continue;
                    }

                    let start = meta.offset as usize;
                    let end = start + meta.count as usize;

                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        for idx in start..end {
                            let dist_v_c = state.index.distances[idx];
                            if (dist_v_c - dist_q_c).abs() >= threshold_top5_f32 + 0.0002 {
                                continue;
                            }
                            let dist_sq = squared_distance_preloaded(vq0, vq1, &state.index.vectors[idx]);
                            if dist_sq < threshold_top5 {
                                top5[4] = (dist_sq, state.index.labels[idx]);
                                let mut x = 4;
                                while x > 0 && top5[x].0 < top5[x - 1].0 {
                                    top5.swap(x, x - 1);
                                    x -= 1;
                                }
                                threshold_top5 = top5[4].0;
                            }
                        }
                    }

                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        for idx in start..end {
                            let dist_v_c = state.index.distances[idx];
                            if (dist_v_c - dist_q_c).abs() >= threshold_top5_f32 + 0.0002 {
                                continue;
                            }
                            let dist_sq = squared_distance_preloaded(vq0, vq1, vq2, vq3, &state.index.vectors[idx]);
                            if dist_sq < threshold_top5 {
                                top5[4] = (dist_sq, state.index.labels[idx]);
                                let mut x = 4;
                                while x > 0 && top5[x].0 < top5[x - 1].0 {
                                    top5.swap(x, x - 1);
                                    x -= 1;
                                }
                                threshold_top5 = top5[4].0;
                            }
                        }
                    }

                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    for idx in start..end {
                        let dist_v_c = state.index.distances[idx];
                        if (dist_v_c - dist_q_c).abs() >= threshold_top5_f32 + 0.0002 {
                            continue;
                        }
                        let dist_sq = squared_distance_fallback(&q, &state.index.vectors[idx]);
                        if dist_sq < threshold_top5 {
                            top5[4] = (dist_sq, state.index.labels[idx]);
                            let mut x = 4;
                            while x > 0 && top5[x].0 < top5[x - 1].0 {
                                top5.swap(x, x - 1);
                                x -= 1;
                            }
                            threshold_top5 = top5[4].0;
                        }
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

                let fraud_count = top5.iter().filter(|&&(_, label)| label == 1).count();
                let response_body = RESPONSES[fraud_count.min(5)];

                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
                    response_body.len(),
                    if req.keep_alive { "keep-alive" } else { "close" }
                );
                stream.write_all(headers.as_bytes())?;
                stream.write_all(response_body)?;
                stream.flush()?;
            } else if req.path == "/ready" && req.method == "GET" {
                let (status, body) = if IS_READY.load(Ordering::Acquire) {
                    ("HTTP/1.1 200 OK", b"OK" as &[u8])
                } else {
                    ("HTTP/1.1 503 Service Unavailable", b"Service Unavailable" as &[u8])
                };
                let headers = format!(
                    "{}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
                    status,
                    body.len(),
                    if req.keep_alive { "keep-alive" } else { "close" }
                );
                stream.write_all(headers.as_bytes())?;
                stream.write_all(body)?;
                stream.flush()?;
            } else if req.path == "/telemetry" && req.method == "GET" {
                let count = METRIC_COUNT.load(Ordering::Relaxed);
                let json_res = if count == 0 {
                    "{\"count\":0,\"json_parse_us\":0.0,\"vectorize_us\":0.0,\"centroid_search_us\":0.0,\"cluster_scan_us\":0.0,\"total_us\":0.0}".to_string()
                } else {
                    let count_f = count as f64;
                    let jp = (METRIC_JSON_PARSE_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
                    let vc = (METRIC_VECTORIZE_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
                    let cs = (METRIC_CENTROID_SEARCH_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
                    let cl = (METRIC_CLUSTER_SCAN_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
                    let tot = (METRIC_TOTAL_NS.load(Ordering::Relaxed) as f64 / count_f) / 1000.0;
                    format!(
                        "{{\"count\":{},\"json_parse_us\":{:.2},\"vectorize_us\":{:.2},\"centroid_search_us\":{:.2},\"cluster_scan_us\":{:.2},\"total_us\":{:.2}}}",
                        count, jp, vc, cs, cl, tot
                    )
                };
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
                    json_res.len(),
                    if req.keep_alive { "keep-alive" } else { "close" }
                );
                stream.write_all(headers.as_bytes())?;
                stream.write_all(json_res.as_bytes())?;
                stream.flush()?;
            } else {
                let body = b"Not Found" as &[u8];
                let headers = format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: {}\r\n\r\n",
                    body.len(),
                    if req.keep_alive { "keep-alive" } else { "close" }
                );
                stream.write_all(headers.as_bytes())?;
                stream.write_all(body)?;
                stream.flush()?;
            }

            let consumed = req.body_offset + req.body_len;
            buf.drain(0..consumed);

            if !req.keep_alive {
                return Ok(());
            }
        }
    }
}

fn run_warmup(state: &Arc<AppState>) {
    let start = Instant::now();
    for i in 0..500 {
        let mut q = [0.0f32; 16];
        for j in 0..14 {
            q[j] = ((i * j) % 100) as f32 / 100.0;
        }

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

        let mut indices = [0u16; K_CENTROIDS];
        for x in 0..K_CENTROIDS {
            indices[x] = x as u16;
        }
        indices.select_nth_unstable_by(state.nprobe - 1, |&a, &b| {
            dists[a as usize].partial_cmp(&dists[b as usize]).unwrap()
        });

        let mut top5 = [(f32::MAX, 0u8); 5];
        let mut threshold_top5 = f32::MAX;

        #[cfg(target_arch = "x86_64")]
        let vq0 = unsafe {
            use std::arch::x86_64::*;
            _mm256_loadu_ps(q.as_ptr())
        };
        #[cfg(target_arch = "x86_64")]
        let vq1 = unsafe {
            use std::arch::x86_64::*;
            _mm256_loadu_ps(q.as_ptr().add(8))
        };

        #[cfg(target_arch = "aarch64")]
        let (vq0, vq1, vq2, vq3) = unsafe {
            use std::arch::aarch64::*;
            (
                vld1q_f32(q.as_ptr()),
                vld1q_f32(q.as_ptr().add(4)),
                vld1q_f32(q.as_ptr().add(8)),
                vld1q_f32(q.as_ptr().add(12)),
            )
        };

        let probed = &mut indices[0..state.nprobe];
        probed.sort_unstable_by(|&a, &b| {
            dists[a as usize].partial_cmp(&dists[b as usize]).unwrap()
        });

        for &k_idx in probed.iter() {
            let k = k_idx as usize;
            let dist_q_c_sq = dists[k];
            let dist_q_c = dist_q_c_sq.sqrt();
            let meta = &state.index.cluster_metadata[k];

            let threshold_top5_f32 = threshold_top5.sqrt();

            if dist_q_c - meta.radius >= threshold_top5_f32 + 0.0002 {
                continue;
            }

            let start = meta.offset as usize;
            let end = start + meta.count as usize;

            #[cfg(target_arch = "x86_64")]
            unsafe {
                for idx in start..end {
                    let dist_v_c = state.index.distances[idx];
                    if (dist_v_c - dist_q_c).abs() >= threshold_top5_f32 + 0.0002 {
                        continue;
                    }
                    let dist_sq = squared_distance_preloaded(vq0, vq1, &state.index.vectors[idx]);
                    if dist_sq < threshold_top5 {
                        top5[4] = (dist_sq, state.index.labels[idx]);
                        let mut x = 4;
                        while x > 0 && top5[x].0 < top5[x - 1].0 {
                            top5.swap(x, x - 1);
                            x -= 1;
                        }
                        threshold_top5 = top5[4].0;
                    }
                }
            }

            #[cfg(target_arch = "aarch64")]
            unsafe {
                for idx in start..end {
                    let dist_v_c = state.index.distances[idx];
                    if (dist_v_c - dist_q_c).abs() >= threshold_top5_f32 + 0.0002 {
                        continue;
                    }
                    let dist_sq = squared_distance_preloaded(vq0, vq1, vq2, vq3, &state.index.vectors[idx]);
                    if dist_sq < threshold_top5 {
                        top5[4] = (dist_sq, state.index.labels[idx]);
                        let mut x = 4;
                        while x > 0 && top5[x].0 < top5[x - 1].0 {
                            top5.swap(x, x - 1);
                            x -= 1;
                        }
                        threshold_top5 = top5[4].0;
                    }
                }
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            for idx in start..end {
                let dist_v_c = state.index.distances[idx];
                if (dist_v_c - dist_q_c).abs() >= threshold_top5_f32 + 0.0002 {
                    continue;
                }
                let dist_sq = squared_distance_fallback(&q, &state.index.vectors[idx]);
                if dist_sq < threshold_top5 {
                    top5[4] = (dist_sq, state.index.labels[idx]);
                    let mut x = 4;
                    while x > 0 && top5[x].0 < top5[x - 1].0 {
                        top5.swap(x, x - 1);
                        x -= 1;
                    }
                    threshold_top5 = top5[4].0;
                }
            }
        }
    }
    println!("Warmup completed in {:?}", start.elapsed());
}

fn main() {
    let workers: usize = std::env::var("WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let nprobe: usize = std::env::var("NPROBE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512)
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

    println!("Running synthetic warmup (500 queries)...");
    run_warmup(&state);
    IS_READY.store(true, Ordering::Release);
    println!("Warmup complete, ready to serve requests");

    // Determinar quantidade de threads para o pool síncrono.
    // Usamos 8 threads por padrão se WORKERS for menor que 8 para garantir alta taxa de aceitação sob keep-alive.
    let num_threads = std::cmp::max(workers * 4, 8);
    println!("Starting synchronous network thread pool with {} threads", num_threads);

    if let Ok(socket_path) = std::env::var("SOCKET_PATH") {
        use std::os::unix::net::UnixListener;
        if std::path::Path::new(&socket_path).exists() {
            let _ = std::fs::remove_file(&socket_path);
        }
        let listener = UnixListener::bind(&socket_path).expect("Failed to bind Unix socket");

        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&socket_path).unwrap().permissions();
        perms.set_mode(0o777);
        std::fs::set_permissions(&socket_path, perms).unwrap();

        println!("API escutando no socket unix: {}", socket_path);

        let listener = Arc::new(listener);
        let mut threads = Vec::new();
        for _ in 0..num_threads {
            let listener = listener.clone();
            let state = state.clone();
            let handle = std::thread::spawn(move || {
                let mut buf = Vec::with_capacity(16384);
                loop {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
                            let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
                            if let Err(_e) = handle_connection(stream, &state, &mut buf) {
                                // Erros normais de timeout ou fechamento são ignorados silenciosamente
                            }
                        }
                        Err(_e) => {}
                    }
                }
            });
            threads.push(handle);
        }
        for handle in threads {
            let _ = handle.join();
        }
    } else {
        let port = std::env::var("PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8080);
        let listener = std::net::TcpListener::bind(format!("0.0.0.0:{}", port)).expect("Failed to bind TCP port");
        println!("API escutando em http://0.0.0.0:{}", port);

        let listener = Arc::new(listener);
        let mut threads = Vec::new();
        for _ in 0..num_threads {
            let listener = listener.clone();
            let state = state.clone();
            let handle = std::thread::spawn(move || {
                let mut buf = Vec::with_capacity(16384);
                loop {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
                            let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(5)));
                            let _ = stream.set_nodelay(true);
                            if let Err(_e) = handle_connection(stream, &state, &mut buf) {
                                // Silencioso
                            }
                        }
                        Err(_e) => {}
                    }
                }
            });
            threads.push(handle);
        }
        for handle in threads {
            let _ = handle.join();
        }
    }
}
