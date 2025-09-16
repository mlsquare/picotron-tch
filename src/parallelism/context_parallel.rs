//! Context parallelism implementation using Tch

use tch::{Device, Tensor};
use anyhow::Result;
use log::info;

/// Context parallelism manager using Tch
pub struct ContextParallel {
    world_size: usize,
    rank: usize,
    device: Device,
}

impl ContextParallel {
    /// Create new context parallel manager
    pub fn new(world_size: usize, rank: usize, device: Device) -> Self {
        info!("Creating context parallel manager: world_size={}, rank={}", world_size, rank);
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
    
    /// Split sequence along sequence dimension
    pub fn split_sequence(&self, tensor: &Tensor) -> Result<Tensor> {
        let seq_len = tensor.size()[1];
        let chunk_size = seq_len / self.world_size as i64;
        let start = self.rank * chunk_size as usize;
        let end = if self.rank == self.world_size - 1 {
            seq_len as usize
        } else {
            (self.rank + 1) * chunk_size as usize
        };
        
        Ok(tensor.narrow(1, start as i64, (end - start) as i64))
    }
    
    /// Concatenate sequences along sequence dimension
    pub fn concat_sequences(&self, tensors: &[Tensor]) -> Result<Tensor> {
        Ok(Tensor::cat(tensors, 1))
    }
}
