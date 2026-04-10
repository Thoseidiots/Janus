use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use anyhow::{Result, Context};
use uuid::Uuid;
use crate::jce::JanusCoherencyEngine;

#[derive(Debug, Clone)]
pub struct NodeMetrics {
    pub node_id: u8,
    pub cpu_load: f32,
    pub gpu_load: f32,
    pub active_wasm_tasks: usize,
}

#[derive(Debug, Clone)]
pub struct TaskDescriptor {
    pub task_id: Uuid,
    pub required_memory_regions: Vec<u64>,
    pub estimated_compute_cost: u64,
}

/// The Locality-Aware Scheduler (LAS) optimizes task placement based on memory and compute.
pub struct LocalityAwareScheduler {
    jce: Arc<JanusCoherencyEngine>,
    node_metrics: Arc<Mutex<HashMap<u8, NodeMetrics>>>,
}

impl LocalityAwareScheduler {
    pub fn new(jce: Arc<JanusCoherencyEngine>) -> Self {
        LocalityAwareScheduler {
            jce,
            node_metrics: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn update_metrics(&self, metrics: NodeMetrics) {
        let mut node_metrics = self.node_metrics.lock().unwrap();
        node_metrics.insert(metrics.node_id, metrics);
    }

    /// Determine the optimal node for a task based on memory locality and compute load.
    pub fn schedule_task(&self, task: &TaskDescriptor) -> Result<u8> {
        let metrics = self.node_metrics.lock().unwrap();
        
        // Calculate locality score for each node (Node 1 and Node 2).
        let mut best_node = 0;
        let mut best_score = -1.0;

        for node_id in 1..=2 {
            let mut score = 0.0;

            // 1. Memory Locality: How many required regions are already owned by this node?
            let mut local_regions = 0;
            for &region_id in &task.required_memory_regions {
                if let Ok(owner) = self.jce.current_owner(region_id) {
                    if owner == node_id {
                        local_regions += 1;
                    }
                }
            }
            score += (local_regions as f32 / task.required_memory_regions.len().max(1) as f32) * 10.0;

            // 2. Compute Load: Penalize nodes with high CPU/GPU load.
            if let Some(m) = metrics.get(&node_id) {
                score -= m.cpu_load * 5.0;
                score -= m.gpu_load * 3.0;
                score -= (m.active_wasm_tasks as f32) * 0.5;
            }

            if score > best_score {
                best_score = score;
                best_node = node_id;
            }
        }

        if best_node == 0 {
            return Err(anyhow::anyhow!("No suitable node found for scheduling"));
        }

        Ok(best_node)
    }
}

// Add a helper to JanusCoherencyEngine to check ownership without acquiring.
impl JanusCoherencyEngine {
    pub fn current_owner(&self, region_id: u64) -> Result<u8> {
        let region = self.regions.get(region_id as usize)
            .context("Region ID out of bounds")?;
        Ok(region.current_owner())
    }
}
