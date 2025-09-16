//! Utility functions for PicoTron Tch

use tch::{Device, Tensor, Kind};
use anyhow::Result;
use log::info;

/// Utility functions for PicoTron Tch
pub struct Utils;

impl Utils {
    /// Create new utils instance
    pub fn new() -> Self {
        Self
    }
    
    /// Create random input tensor for testing
    pub fn create_random_input(batch_size: usize, seq_len: usize, vocab_size: usize, device: Device) -> Tensor {
        Tensor::randint(vocab_size as i64, &[batch_size as i64, seq_len as i64], (Kind::Int64, device))
    }
    
    /// Create random labels for testing
    pub fn create_random_labels(batch_size: usize, seq_len: usize, vocab_size: usize, device: Device) -> Tensor {
        Tensor::randint(vocab_size as i64, &[batch_size as i64, seq_len as i64], (Kind::Int64, device))
    }
    
    /// Print tensor information
    pub fn print_tensor_info(tensor: &Tensor, name: &str) {
        info!("{}: shape={:?}, dtype={:?}, device={:?}", 
              name, tensor.size(), tensor.kind(), tensor.device());
    }
    
    /// Calculate model size in MB
    pub fn calculate_model_size_mb(num_parameters: i64) -> f64 {
        // Assuming float32 (4 bytes per parameter)
        (num_parameters * 4) as f64 / (1024.0 * 1024.0)
    }
}
