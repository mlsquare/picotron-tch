//! Basic PicoTron Tch example

use picotron_tch::*;
use tch::{Device, Tensor, Kind};
use anyhow::Result;

fn main() -> Result<()> {
    // Initialize PicoTron
    init()?;
    
    println!("PicoTron Tch Version: {}", version());
    
    // Check if CUDA is available
    let device = if tch::Cuda::is_available() {
        println!("CUDA is available, using GPU");
        Device::Cuda(0)
    } else {
        println!("CUDA not available, using CPU");
        Device::Cpu
    };
    
    // Create a simple configuration
    let mut config = PicoTronConfig::default();
    config.model.hidden_size = 512;  // Smaller model for demo
    config.model.num_hidden_layers = 4;
    config.model.num_attention_heads = 8;
    config.model.vocab_size = 1000;
    config.validate()?;
    
    println!("Configuration validated successfully");
    println!("Model: {}", config.model.name);
    println!("Hidden Size: {}", config.model.hidden_size);
    println!("Attention Heads: {}", config.model.num_attention_heads);
    println!("Hidden Layers: {}", config.model.num_hidden_layers);
    
    // Create model
    let model = PicoTronModel::new(config.model.clone(), device)?;
    let num_params = model.num_parameters();
    let model_size_mb = Utils::calculate_model_size_mb(num_params);
    
    println!("Model created successfully");
    println!("Number of parameters: {}", num_params);
    println!("Model size: {:.2} MB", model_size_mb);
    
    // Create trainer
    let mut trainer = PicoTronTrainer::new(config.training.clone(), &model)?;
    
    // Create sample data
    let batch_size = 2;
    let seq_len = 10;
    let input_ids = Utils::create_random_input(batch_size, seq_len, config.model.vocab_size, device);
    let labels = Utils::create_random_labels(batch_size, seq_len, config.model.vocab_size, device);
    
    Utils::print_tensor_info(&input_ids, "input_ids");
    Utils::print_tensor_info(&labels, "labels");
    
    // Training step
    let loss = trainer.train_step(&model, &input_ids, Some(&labels))?;
    println!("Training loss: {:.4}", loss);
    
    // Evaluation step
    let eval_loss = trainer.eval_step(&model, &input_ids, Some(&labels))?;
    println!("Evaluation loss: {:.4}", eval_loss);
    
    // Test forward pass
    let logits = model.forward(&input_ids, None)?;
    Utils::print_tensor_info(&logits, "logits");
    
    // Save configuration
    config.to_json("config.json")?;
    println!("Configuration saved to config.json");
    
    // Test parallelism
    let data_parallel = DataParallel::new(1, 0, device);
    let tensor_parallel = TensorParallel::new(1, 0, device);
    let pipeline_parallel = PipelineParallel::new(1, 0, device);
    let context_parallel = ContextParallel::new(1, 0, device);
    
    println!("Parallelism managers created successfully");
    println!("Data parallel world size: {}", data_parallel.world_size());
    println!("Tensor parallel world size: {}", tensor_parallel.world_size());
    println!("Pipeline parallel world size: {}", pipeline_parallel.world_size());
    println!("Context parallel world size: {}", context_parallel.world_size());
    
    Ok(())
}
