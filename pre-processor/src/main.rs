use std::fs::File;
use std::io::{BufReader, Write};
use flate2::read::GzDecoder;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use serde::Deserialize;
use rayon::prelude::*;

const K: usize = 8192;
const SUB_SAMPLE_SIZE: usize = 300_000;
const KMEANS_ITERATIONS: usize = 25;

#[derive(Deserialize)]
struct RawEntry {
    vector: Vec<f32>,
    label: String,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(dead_code)]
struct Centroid {
    features: [f32; 16],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClusterInfo {
    offset: u32,
    count: u32,
    radius: f32,
}

fn squared_distance(a: &[f32; 16], b: &[f32; 16]) -> f32 {
    let mut sum = 0.0;
    for i in 0..16 {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

fn main() {
    println!("=== Iniciando Pré-Processador ===");

    // 1. Carregar references.json.gz
    let gz_path = "resources/references.json.gz";
    println!("Carregando {}...", gz_path);
    let file = File::open(gz_path).expect("Falha ao abrir references.json.gz");
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);

    println!("Fazendo parse do JSON...");
    let raw_entries: Vec<RawEntry> = serde_json::from_reader(reader)
        .expect("Falha ao ler JSON de referências");
    let n_vectors = raw_entries.len();
    println!("Total de vetores carregados: {}", n_vectors);

    // Converter para representação otimizada
    let mut features = vec![[0.0f32; 16]; n_vectors];
    let mut labels = vec![0u8; n_vectors];

    for (i, entry) in raw_entries.iter().enumerate() {
        // Pad de 14 para 16 floats
        for j in 0..14 {
            features[i][j] = entry.vector[j];
        }
        labels[i] = if entry.label == "fraud" { 1 } else { 0 };
    }

    // 2. Treinamento do K-Means usando sub-amostra
    println!("Selecionando sub-amostra de {} para K-Means...", SUB_SAMPLE_SIZE);
    let mut rng = thread_rng();
    let sample_features: Vec<[f32; 16]> = features
        .choose_multiple(&mut rng, SUB_SAMPLE_SIZE)
        .cloned()
        .collect();

    // KMeans++ initialization (O(K*N)): maintain per-point min distance to
    // any chosen centroid; update incrementally after each new centroid.
    println!("Inicializando centroides com KMeans++ ({} centroids)...", K);
    let mut centroids = Vec::with_capacity(K);
    centroids.push(*sample_features.choose(&mut rng).unwrap());
    let mut min_dists = vec![f32::MAX; SUB_SAMPLE_SIZE];
    while centroids.len() < K {
        // Update min_dists with the last added centroid
        let last = centroids.last().unwrap();
        for (j, v) in sample_features.iter().enumerate() {
            let d = squared_distance(v, last);
            if d < min_dists[j] {
                min_dists[j] = d;
            }
        }
        // Sample next centroid proportional to min_dist^2
        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        let mut pick = rng.gen::<f64>() * total;
        let mut chosen = sample_features.last().unwrap();
        for (v, &d) in sample_features.iter().zip(min_dists.iter()) {
            pick -= d as f64;
            if pick <= 0.0 {
                chosen = v;
                break;
            }
        }
        centroids.push(*chosen);
        if centroids.len() % 1024 == 0 {
            println!("  KMeans++ init: {}/{}", centroids.len(), K);
        }
    }
    let mut centroids: Vec<[f32; 16]> = centroids;

    println!("Rodando K-Means ({} iterações)...", KMEANS_ITERATIONS);
    for iter in 0..KMEANS_ITERATIONS {
        let (new_centroids, counts) = sample_features
            .par_iter()
            .map(|vec| {
                let mut min_dist = f32::MAX;
                let mut best_k = 0;
                for k in 0..K {
                    let dist = squared_distance(vec, &centroids[k]);
                    if dist < min_dist {
                        min_dist = dist;
                        best_k = k;
                    }
                }
                (best_k, vec)
            })
            .fold(
                || (vec![[0.0f32; 16]; K], vec![0u32; K]),
                |mut acc, (best_k, vec)| {
                    for j in 0..16 {
                        acc.0[best_k][j] += vec[j];
                    }
                    acc.1[best_k] += 1;
                    acc
                },
            )
            .reduce(
                || (vec![[0.0f32; 16]; K], vec![0u32; K]),
                |mut a, b| {
                    for k in 0..K {
                        for j in 0..16 {
                            a.0[k][j] += b.0[k][j];
                        }
                        a.1[k] += b.1[k];
                    }
                    a
                },
            );

        for k in 0..K {
            if counts[k] > 0 {
                for j in 0..16 {
                    centroids[k][j] = new_centroids[k][j] / (counts[k] as f32);
                }
            } else {
                // Reinicializa com um ponto aleatório se o cluster esvaziar
                centroids[k] = *sample_features.choose(&mut rng).unwrap();
            }
        }
        println!("  Iteração {}/{} concluída.", iter + 1, KMEANS_ITERATIONS);
    }

    // 3. Atribuição de todos os 3.000.000 vetores
    println!("Atribuindo todos os vetores aos centroides...");
    let assignments: Vec<(usize, f32)> = features
        .par_iter()
        .map(|vec| {
            let mut min_dist = f32::MAX;
            let mut best_k = 0;
            for k in 0..K {
                let dist = squared_distance(vec, &centroids[k]);
                if dist < min_dist {
                    min_dist = dist;
                    best_k = k;
                }
            }
            (best_k, min_dist)
        })
        .collect();

    let mut cluster_assignments = vec![Vec::new(); K];
    for (idx, &(best_k, _)) in assignments.iter().enumerate() {
        cluster_assignments[best_k].push(idx);
    }

    // 4. Agrupar dados no disco
    println!("Ordenando vetores por cluster para gravação contígua...");
    let mut ordered_features = vec![[0.0f32; 16]; n_vectors];
    let mut ordered_labels = vec![0u8; n_vectors];
    let mut ordered_distances = vec![0.0f32; n_vectors];
    let mut cluster_metadata = vec![ClusterInfo { offset: 0, count: 0, radius: 0.0 }; K];

    let mut current_offset = 0u32;
    for k in 0..K {
        let indices = &cluster_assignments[k];
        let count = indices.len() as u32;
        let start = current_offset as usize;

        for &idx in indices {
            ordered_features[current_offset as usize] = features[idx];
            ordered_labels[current_offset as usize] = labels[idx];
            ordered_distances[current_offset as usize] = assignments[idx].1.sqrt();
            current_offset += 1;
        }

        let end = current_offset as usize;
        let mut max_radius = 0.0f32;
        for idx in start..end {
            if ordered_distances[idx] > max_radius {
                max_radius = ordered_distances[idx];
            }
        }

        cluster_metadata[k] = ClusterInfo {
            offset: start as u32,
            count,
            radius: max_radius,
        };
    }

    // 5. Escrever arquivo binário index.bin
    let output_path = "my-solution/index.bin";
    println!("Escrevendo índice binário para {}...", output_path);
    let mut out_file = File::create(output_path).expect("Falha ao criar index.bin");

    // Header (16 bytes)
    out_file.write_all(b"IVFF").unwrap();
    out_file.write_all(&(K as u32).to_le_bytes()).unwrap();
    out_file.write_all(&(n_vectors as u32).to_le_bytes()).unwrap();
    out_file.write_all(&[0u8; 4]).unwrap(); // Padding para 16 bytes

    // Centroids (K * 64 bytes)
    let centroids_bytes = unsafe {
        std::slice::from_raw_parts(
            centroids.as_ptr() as *const u8,
            K * std::mem::size_of::<[f32; 16]>(),
        )
    };
    out_file.write_all(centroids_bytes).unwrap();

    // Cluster Metadata (K * 8 bytes)
    let metadata_bytes = unsafe {
        std::slice::from_raw_parts(
            cluster_metadata.as_ptr() as *const u8,
            K * std::mem::size_of::<ClusterInfo>(),
        )
    };
    out_file.write_all(metadata_bytes).unwrap();

    // Vectors Features (N * 64 bytes)
    let features_bytes = unsafe {
        std::slice::from_raw_parts(
            ordered_features.as_ptr() as *const u8,
            n_vectors * std::mem::size_of::<[f32; 16]>(),
        )
    };
    out_file.write_all(features_bytes).unwrap();

    // Labels (N * 1 byte)
    out_file.write_all(&ordered_labels).unwrap();

    // Distances (N * 4 bytes)
    let distances_bytes = unsafe {
        std::slice::from_raw_parts(
            ordered_distances.as_ptr() as *const u8,
            n_vectors * std::mem::size_of::<f32>(),
        )
    };
    out_file.write_all(distances_bytes).unwrap();

    println!("Gravação concluída com sucesso! index.bin pronto.");
}
