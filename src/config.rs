//! Configuration management for PicoTron Tch

use serde::{Deserialize, Serialize};
use std::path::Path;
use anyhow::Result;

/// PicoTron Tch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PicoTronConfig {
    /// Model configuration
    pub model: ModelConfig,
    
    /// Training configuration
    pub training: TrainingConfig,
    
    /// Parallelism configuration
    pub parallelism: ParallelismConfig,
    
    /// Data configuration
    pub data: DataConfig,
    
    /// Output configuration
    pub output: OutputConfig,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Model name (e.g., "llama-7b")
    pub name: String,
    
    /// Vocabulary size
    pub vocab_size: usize,
    
    /// Hidden dimension
    pub hidden_size: usize,
    
    /// Number of attention heads
    pub num_attention_heads: usize,
    
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    
    /// Intermediate size (FFN)
    pub intermediate_size: usize,
    
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    
    /// Hidden dropout probability
    pub hidden_dropout_prob: f64,
    
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f64,
    
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    
    /// Initializer range
    pub initializer_range: f64,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Batch size per device
    pub per_device_train_batch_size: usize,
    
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    
    /// Number of training epochs
    pub num_train_epochs: usize,
    
    /// Maximum training steps
    pub max_steps: Option<usize>,
    
    /// Warmup steps
    pub warmup_steps: usize,
    
    /// Weight decay
    pub weight_decay: f64,
    
    /// Adam epsilon
    pub adam_epsilon: f64,
    
    /// Adam beta1
    pub adam_beta1: f64,
    
    /// Adam beta2
    pub adam_beta2: f64,
    
    /// Maximum gradient norm
    pub max_grad_norm: f64,
    
    /// Logging steps
    pub logging_steps: usize,
    
    /// Save steps
    pub save_steps: usize,
    
    /// Evaluation steps
    pub eval_steps: Option<usize>,
}

/// Parallelism configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismConfig {
    /// Data parallelism degree
    pub data_parallel_size: usize,
    
    /// Tensor parallelism degree
    pub tensor_parallel_size: usize,
    
    /// Pipeline parallelism degree
    pub pipeline_parallel_size: usize,
    
    /// Context parallelism degree
    pub context_parallel_size: usize,
    
    /// Pipeline engine type
    pub pipeline_engine: String,
}

/// Data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    /// Training data path
    pub train_data_path: String,
    
    /// Evaluation data path
    pub eval_data_path: Option<String>,
    
    /// Data preprocessing workers
    pub num_workers: usize,
    
    /// Data cache directory
    pub cache_dir: Option<String>,
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory
    pub output_dir: String,
    
    /// Experiment name
    pub experiment_name: String,
    
    /// Save total limit
    pub save_total_limit: usize,
    
    /// Load best model at end
    pub load_best_model_at_end: bool,
    
    /// Metric for best model
    pub metric_for_best_model: String,
}

impl Default for PicoTronConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            parallelism: ParallelismConfig::default(),
            data: DataConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            name: "llama-7b".to_string(),
            vocab_size: 32000,
            hidden_size: 4096,
            num_attention_heads: 32,
            num_hidden_layers: 32,
            intermediate_size: 11008,
            max_position_embeddings: 2048,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            per_device_train_batch_size: 4,
            gradient_accumulation_steps: 32,
            num_train_epochs: 3,
            max_steps: None,
            warmup_steps: 1000,
            weight_decay: 0.01,
            adam_epsilon: 1e-8,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            max_grad_norm: 1.0,
            logging_steps: 10,
            save_steps: 1000,
            eval_steps: Some(500),
        }
    }
}

impl Default for ParallelismConfig {
    fn default() -> Self {
        Self {
            data_parallel_size: 1,
            tensor_parallel_size: 1,
            pipeline_parallel_size: 1,
            context_parallel_size: 1,
            pipeline_engine: "1f1b".to_string(),
        }
    }
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            train_data_path: "data/train".to_string(),
            eval_data_path: None,
            num_workers: 4,
            cache_dir: None,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            output_dir: "outputs".to_string(),
            experiment_name: "picotron-experiment".to_string(),
            save_total_limit: 3,
            load_best_model_at_end: true,
            metric_for_best_model: "loss".to_string(),
        }
    }
}

impl PicoTronConfig {
    /// Load configuration from JSON file
    pub fn from_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: PicoTronConfig = serde_json::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to JSON file
    pub fn to_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Load configuration from TOML file
    pub fn from_toml<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: PicoTronConfig = toml::from_str(&content)?;
        Ok(config)
    }
    
    /// Save configuration to TOML file
    pub fn to_toml<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate model config
        if self.model.vocab_size == 0 {
            return Err(anyhow::anyhow!("vocab_size must be > 0"));
        }
        if self.model.hidden_size == 0 {
            return Err(anyhow::anyhow!("hidden_size must be > 0"));
        }
        if self.model.num_attention_heads == 0 {
            return Err(anyhow::anyhow!("num_attention_heads must be > 0"));
        }
        if self.model.hidden_size % self.model.num_attention_heads != 0 {
            return Err(anyhow::anyhow!("hidden_size must be divisible by num_attention_heads"));
        }
        
        // Validate training config
        if self.training.learning_rate <= 0.0 {
            return Err(anyhow::anyhow!("learning_rate must be > 0"));
        }
        if self.training.per_device_train_batch_size == 0 {
            return Err(anyhow::anyhow!("per_device_train_batch_size must be > 0"));
        }
        
        // Validate parallelism config
        let total_parallelism = self.parallelism.data_parallel_size * 
                               self.parallelism.tensor_parallel_size * 
                               self.parallelism.pipeline_parallel_size * 
                               self.parallelism.context_parallel_size;
        if total_parallelism == 0 {
            return Err(anyhow::anyhow!("Total parallelism must be > 0"));
        }
        
        Ok(())
    }
}
