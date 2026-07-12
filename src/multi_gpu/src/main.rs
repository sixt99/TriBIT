use lender::prelude::*;
use std::env;
use std::time::Instant;
use webgraph::prelude::*;
//use sysinfo::System;
use mpi::collective::SystemOperation;
use mpi::traits::*;
use std::io::{self, Write};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::fs;
use std::collections::HashMap;

extern "C" {
    fn count_partial_triangles_diagonal(
        cols_rows_zipped: *const *const u64,
        nnz_ptr: *const u64,
        num_ptrs_cols: u32,
        num_nodes: u64,
    ) -> u64;

    fn count_partial_triangles_off_diagonal(
        cols_rows_zipped_cols: *const *const u64,
        nnz_cols_ptr: *const u64,
        num_ptrs_cols: u32,
        cols_rows_zipped_idxs: *const *const u64,
        nnz_idxs_ptr: *const u64,
        num_ptrs_idxs: u32,
        num_nodes: u64,
    ) -> u64;

    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn init_gpu_kernel5(gpu_id: i32);
    fn init_gpu_kernel6(gpu_id: i32);
}

fn read_properties(basename: &str) -> HashMap<String, String> {
    let path = format!("{}.properties", basename);
    let contents = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read properties file {}: {}", path, e));

    contents
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(2, '=');
            let key = parts.next()?.trim().to_string();
            let value = parts.next()?.trim().to_string();
            Some((key, value))
        })
        .collect()
}

fn get_available_memory_gib() -> f64 {
    let meminfo = fs::read_to_string("/proc/meminfo").expect("Failed to read /proc/meminfo");
    for line in meminfo.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: u64 = line
                .split_whitespace()
                .nth(1)
                .expect("Failed to parse MemAvailable")
                .parse()
                .expect("Failed to convert to u64");
            return kb as f64 / (1024.0 * 1024.0);
        }
    }
    panic!("MemAvailable not found in /proc/meminfo");
}

fn main() -> anyhow::Result<()> {
    // MPI managing
    let universe = mpi::initialize().unwrap();
    let world = universe.world();
    let rank = world.rank(); // Which node am I?
    let size = world.size(); // How many nodes are available?

    // Get number of GPUs
    //let num_devices = cuda_get_device_count().expect("Error in reading number of devices");
    let mut num_devices: i32 = 0;
    unsafe { cudaGetDeviceCount(&mut num_devices); }
    println!("Number of GPUs on node {}: {}", rank, num_devices);
    // let total_num_devices = size * num_devices;

    // Read arguments (graph and number of partitions)
    let basename = env::args()
        .nth(1)
        .expect("Please provide graph basename as argument");
    let graph = BvGraphSeq::with_basename(basename.clone()).load()?;
    let mut iter = graph.iter();
    // Number of partitions needed to cover the whole graph
    let num_partitions: usize = env::args()
        .nth(2)
        .expect("Please provide a partition size")
        .parse()
        .expect("Partition size must be a valid usize");
    // Important variables
    let num_nodes = graph.num_nodes();
    let block_size = (num_nodes / num_partitions).div_ceil(32) * 32; // Closest multiple of 32 rounding up
    let num_partitions = (num_nodes + block_size - 1) / block_size; // All partitions needed to cover this
    let meta_size = num_partitions / size as usize;

    let mut cols_rows_zipped_vec: Vec<Vec<Vec<u64>>> =
        vec![vec![Vec::new(); num_partitions]; num_partitions];
    // Create a matrix of nnz for each partition
    let total = iter.size_hint().0 as u64;
    let step_ratio = 10;
    assert!(0 < step_ratio && step_ratio < 100);


    // ------------ Generate graph ---------------
    // Iterate over the graph and distribute coordinates into cols_rows_zipped_vec
    let mut counter = 0;
    let mut nnz_matrix: Vec<Vec<usize>> = vec![vec![0; num_partitions]; num_partitions];
    let start = meta_size * rank as usize;
    let end = if rank < size - 1 {
        meta_size * (rank as usize + 1)
    } else {
        num_partitions as usize
    };

    let print_progress = |progress: u64, edges_loaded: usize, load_start: &Instant| {
        let mem_gib = (edges_loaded * 8) as f64 / (1u64 << 30) as f64;
        let host_free_gib = get_available_memory_gib();
        let elapsed = load_start.elapsed().as_secs_f64();
        let rate = if elapsed > 0.0 { edges_loaded as f64 / elapsed } else { 0.0 };
        println!(
            "Loading {:3}% | graph mem: {:6.2} GiB | host free: {:7.2} GiB | rate: {:>9.0} edges/s",
            progress, mem_gib, host_free_gib, rate
        );
        io::stdout().flush().unwrap();
    };

    // Load data
    let load_start = Instant::now();
    while let Some((src, successors)) = iter.next() {
        // Custom progress bar
        if rank == 0 && (counter % (total / step_ratio) == 0 || counter == total - 1) {
            let progress = (counter * 100 + total / 2) / total;
            let edges_loaded: usize = cols_rows_zipped_vec
                .iter()
                .flat_map(|row| row.iter())
                .map(|block| block.len())
                .sum();
            print_progress(progress, edges_loaded, &load_start);
            io::stdout().flush().unwrap();
        }

        for dst in successors {
            // Flip variables if src < dst. Skip cases where src == dst
            let (row, col) = if dst < src {
                (src, dst)
            } else if src < dst {
                (dst, src)
            } else {
                continue;
            };
            // LOAD A DIFFERENT PORTION OF THE GRAPH TO EACH NODE
            if (col/block_size) < end && start <= (row/block_size) as usize {
                let packed = ((col as u64) << 32) | (row as u64);
                cols_rows_zipped_vec[row / block_size][col / block_size].push(packed);
                nnz_matrix[row / block_size][col / block_size] += 1;
            }
        }
        counter += 1;
    }
    // Synchronize all ranks
    world.barrier();
    let load_duration = load_start.elapsed().as_secs_f64();


    if rank == 0 {
        println!("Counting triangles...")
    };
    // Cumsum of nnz_matrix
    //println!("{nnz_matrix:?}");

    for i in 0..num_partitions {
        for j in 1..i + 1 {
            nnz_matrix[i][j] += nnz_matrix[i][j - 1];
        }
    }
    // -------------------------------------------

    // ---------- Create list of tasks -----------
    // Each GPU within each node will consume tasks from here
    let mut list_of_tasks: Vec<(usize, Vec<usize>)> = Vec::new();
    let threshold: usize = 1_000_000_000; // Maximum number of nnz permitted per task
    // Diagonal cases
    let start_p = meta_size * (rank as usize);
    let end_p = if rank == size - 1 {
        num_partitions
    } else {
        meta_size * (rank as usize + 1)
    };
    for partition in (start_p..end_p).rev() {
        list_of_tasks.push((partition, vec![partition]));
    }
    // Off-diagonal cases
    for partition_col in (start_p..end_p).rev() {
        let mut idx_list: Vec<usize> = Vec::new();
        let mut nnz_sum: usize = 0;
        for partition_idx in partition_col + 1..num_partitions {
            nnz_sum += nnz_matrix[partition_idx][partition_col];
            if nnz_sum < threshold {
                idx_list.push(partition_idx);
            } else {
                if !idx_list.is_empty() {
                    // Each rank will have a different list of tasks
                    list_of_tasks.push((partition_col, idx_list));
                }
                // Start new group WITH current partition
                idx_list = vec![partition_idx];
                nnz_sum = nnz_matrix[partition_idx][partition_col];
            }
        }
        // Flush remaining at end of column
        if !idx_list.is_empty() {
            list_of_tasks.push((partition_col, idx_list));
        }
    }
    let task_counter = Arc::new(AtomicU32::new(0));
    // -------------------------------------------

    // Synchronize all ranks
    world.barrier();

    // Measure total time
    let start_processing = Instant::now();

    // Feed GPUs
    let mut handles = vec![];
    let cols_rows_zipped_vec = Arc::new(cols_rows_zipped_vec);
    let tc = Arc::new(AtomicU64::new(0));
    let list_of_tasks = Arc::new(list_of_tasks);
    // Run first experiments for equal cols and idxs (usually the hardest experiments)
    // Each node will create num_device threads, each managing ONE device
    for gpu_id in 0..num_devices {
        // Create references
        let cols_rows_zipped_vec = Arc::clone(&cols_rows_zipped_vec);
        let tc = Arc::clone(&tc);
        let task_counter = Arc::clone(&task_counter);
        let list_of_tasks = Arc::clone(&list_of_tasks);

        // Create as many handles as num_devices
        let handle = thread::spawn(move || {
            let mut partial_tc: u64;
            unsafe { init_gpu_kernel5(gpu_id) }
            unsafe { init_gpu_kernel6(gpu_id) }

            loop {
                // Atomically grab index AND increment in one step
                let idx = task_counter.fetch_add(1, Ordering::Relaxed) as usize;
                if idx >= list_of_tasks.len() {
                    break;
                }
                let (col_partition, ref idx_list) = list_of_tasks[idx]; // Consume task from list

                // Get ptrs and lens of each data chunk for each task
                // Cols
                let num_ptrs_cols = col_partition + 1;
                let (cols_ptr_list, cols_len_list): (Vec<*const u64>, Vec<u64>) =
                    cols_rows_zipped_vec[col_partition]
                        .iter()
                        .take(num_ptrs_cols) // Cut right on the diagonal
                        .map(|v| (v.as_ptr(), v.len() as u64))
                        .unzip();
                // Idxs
                let mut idxs_ptr_list: Vec<*const u64> = Vec::new();
                let mut idxs_len_list: Vec<u64> = Vec::new();
                for &idx_partition in idx_list {
                    let (ptrs, lens): (Vec<*const u64>, Vec<u64>) = cols_rows_zipped_vec
                        [idx_partition]
                        .iter()
                        .take(num_ptrs_cols) // Don't go further down than cols
                        .map(|v: &Vec<u64>| (v.as_ptr(), v.len() as u64))
                        .unzip();
                    idxs_ptr_list.extend(ptrs);
                    idxs_len_list.extend(lens);
                }
                let num_ptrs_idxs = num_ptrs_cols * idx_list.len();

                // Accumulate lens
                let cols_len_list: Vec<u64> = std::iter::once(0)
                    .chain(cols_len_list.iter().scan(0, |acc, &x| {
                        *acc += x;
                        Some(*acc)
                    }))
                    .collect();
                let nnz_cols = *cols_len_list.last().unwrap();
                let idxs_len_list: Vec<u64> = std::iter::once(0)
                    .chain(idxs_len_list.iter().scan(0, |acc, &x| {
                        *acc += x;
                        Some(*acc)
                    }))
                    .collect();
                let nnz_idxs = *idxs_len_list.last().unwrap();

                // Launch C++ kernels
                partial_tc = 0;
                if nnz_cols != 0 && nnz_idxs != 0 {
                    // Diagonal kernel
                    if idx_list.len() == 1 && col_partition == idx_list[0] {
                        unsafe {
                            partial_tc = count_partial_triangles_diagonal(
                                cols_ptr_list.as_ptr(),
                                cols_len_list.as_ptr(),
                                num_ptrs_cols as u32,
                                num_nodes as u64,
                            );
                        }
                    }
                    // Off-diagonal kernel
                    else if idx_list.iter().all(|&idx| col_partition < idx) {
                        // col_partition is strictly smaller than every idx_partition
                        unsafe {
                            partial_tc = count_partial_triangles_off_diagonal(
                                cols_ptr_list.as_ptr(),
                                cols_len_list.as_ptr(),
                                num_ptrs_cols as u32,
                                idxs_ptr_list.as_ptr(),
                                idxs_len_list.as_ptr(),
                                num_ptrs_idxs as u32,
                                num_nodes as u64,
                            );
                        }
                    } else {
                        panic!("Strange list of idxs partitions was given: {:?}", idx_list);
                    }
                }
                tc.fetch_add(partial_tc, Ordering::Relaxed);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Get accumulated result
    let local_tc = tc.load(Ordering::Relaxed);
    let mut global_tc: u64 = 0;
    world.all_reduce_into(&local_tc, &mut global_tc, SystemOperation::sum());
    let counting_time = start_processing.elapsed().as_secs_f64();

    // Sum up total number of devices
    let local_devices = num_devices as u64;
    let mut total_devices: u64 = 0;
    world.all_reduce_into(&local_devices, &mut total_devices, SystemOperation::sum());

    // Sum up total number of tasks
    let local_tasks = list_of_tasks.len() as u64;
    let mut global_tasks: u64 = 0;
    world.all_reduce_into(&local_tasks, &mut global_tasks, SystemOperation::sum());

    if rank == 0 {
        let triangles_per_sec = if counting_time > 0.0 {
            global_tc as f64 / counting_time
        } else {
            0.0
        };

        let props = read_properties(&basename);
        let declared_nodes: u64 = props.get("nodes").and_then(|s| s.parse().ok()).unwrap_or(0);
        let declared_arcs: u64 = props.get("arcs").and_then(|s| s.parse().ok()).unwrap_or(0);
        let graph_mem_gib = (declared_arcs * 8) as f64 / (1u64 << 30) as f64;

        println!(
            "============================================================\n\
             {:^60}\n\
             ============================================================\n\
             Graph               : {}\n\
             Nodes (declared)    : {}\n\
             Arcs (declared)     : {}\n\
             Graph memory (GiB)  : {:.2}\n\
             Load time (s)       : {:.2}\n\
             Partition           : {}\n\
             Tasks processed     : {}\n\
             Triangles           : {}\n\
             Counting time (s)   : {:.2}\n\
             GPUs                : {}\n\
             Triangles/sec       : {:.2e}\n\
             ============================================================",
            "Triangle Counting Summary",
            basename,
            declared_nodes,
            declared_arcs,
            graph_mem_gib,
            load_duration,
            num_partitions,
            global_tasks,
            global_tc,
            counting_time,
            total_devices,
            triangles_per_sec
        );
    }

    Ok(())
}
