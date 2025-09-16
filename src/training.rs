//! Training loop for PicoTron Tch

use tch::{nn, Device, Tensor, Kind};
use crate::config::TrainingConfig;
use crate::model::PicoTronModel;
use anyhow::Result;
use log::{info, warn};

/// PicoTron trainer using Tch
pub struct PicoTronTrainer {
    config: TrainingConfig,
    device: Device,
    optimizer: nn::Optimizer,
}

impl PicoTronTrainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig, model: &PicoTronModel) -> Result<Self> {
        info!("Creating PicoTron trainer with config: {:?}", config);
        
        let device = model.device();
        let vs = model.var_store();
        
        // Create optimizer
        let optimizer = nn::Adam::default()
            .lr(config.learning_rate)
            .betas(&[config.adam_beta1, config.adam_beta2])
            .eps(config.adam_epsilon)
            .weight_decay(config.weight_decay)
            .build(vs, config.learning_rate)?;
        
        Ok(Self {
            config,
            device,
            optimizer,
        })
    }
    
    /// Get training configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
    
    /// Get device
    pub fn device(&self) -> Device {
        self.device
    }
    
    /// Training step
    pub fn train_step(&mut self, model: &PicoTronModel, input_ids: &Tensor, labels: Option<&Tensor>) -> Result<f64> {
        // Forward pass
        let logits = model.forward(input_ids, None)?;
        
        // Compute loss
        let loss = if let Some(labels) = labels {
            // Cross-entropy loss for language modeling
            let shift_logits = logits.narrow(1, 0, logits.size()[1] - 1);
            let shift_labels = labels.narrow(1, 1, labels.size()[1] - 1);
            shift_logits.cross_entropy_for_logits(&shift_labels)
        } else {
            // If no labels, return dummy loss
            Tensor::zeros(&[], (Kind::Float, self.device))
        };
        
        // Backward pass
        self.optimizer.zero_grad();
        loss.backward();
        
        // Gradient clipping
        if self.config.max_grad_norm > 0.0 {
            nn::clip_grad_norm(&self.optimizer, self.config.max_grad_norm);
        }
        
        // Optimizer step
        self.optimizer.step();
        
        Ok(loss.double_value(&[]))
    }
    
    /// Evaluation step
    pub fn eval_step(&self, model: &PicoTronModel, input_ids: &Tensor, labels: Option<&Tensor>) -> Result<f64> {
        // Forward pass (no gradients)
        let logits = model.forward(input_ids, None)?;
        
        // Compute loss
        let loss = if let Some(labels) = labels {
            let shift_logits = logits.narrow(1, 0, logits.size()[1] - 1);
            let shift_labels = labels.narrow(1, 1, labels.size()[1] - 1);
            shift_logits.cross_entropy_for_logits(&shift_labels)
        } else {
            Tensor::zeros(&[], (Kind::Float, self.device))
        };
        
        Ok(loss.double_value(&[]))
    }
    
    /// Save model checkpoint
    pub fn save_checkpoint(&self, model: &PicoTronModel, path: &str) -> Result<()> {
        info!("Saving checkpoint to: {}", path);
        model.var_store().save(path)?;
        Ok(())
    }
    
    /// Load model checkpoint
    pub fn load_checkpoint(&self, model: &PicoTronModel, path: &str) -> Result<()> {
        info!("Loading checkpoint from: {}", path);
        model.var_store().load(path)?;
        Ok(())
    }
}
