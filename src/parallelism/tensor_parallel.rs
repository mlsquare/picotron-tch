//! Tensor parallelism implementation using Tch

use tch::{Device, Tensor};
use anyhow::Result;
use log::info;

/// Tensor parallelism manager using Tch
pub struct TensorParallel {
    world_size: usize,
    rank: usize,
    device: Device,
}

impl TensorParallel {
    /// Create new tensor parallel manager
    pub fn new(world_size: usize, rank: usize, device: Device) -> Self {
        info!("Creating tensor parallel manager: world_size={}, rank={}", world_size, rank);
        Self { world_size, rank, device }
    }
    
    /// Get world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }
    
    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }
    
    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }
    
    /// Split tensor along last dimension
    pub fn split_tensor(&self, tensor: &Tensor) -> Result<Tensor> {
        let dim = tensor.size().len() - 1;
        let size = tensor.size()[dim];
        let chunk_size = size / self.world_size as i64;
        let start = self.rank * chunk_size as usize;
        let end = if self.rank == self.world_size - 1 {
            size as usize
        } else {
            (self.rank + 1) * chunk_size as usize
        };
        
        Ok(tensor.narrow(dim as i64, start as i64, (end - start) as i64))
    }
    
    /// Concatenate tensors along last dimension
    pub fn concat_tensors(&self, tensors: &[Tensor]) -> Result<Tensor> {
        Ok(Tensor::cat(tensors, -1))
    }
}
