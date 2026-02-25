use janus_wasm::{JanusWasmHost, WasmSnapshot};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use anyhow::Result;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct WasmTask {
    pub id: Uuid,
    pub wasm_binary: Vec<u8>,
    pub initial_snapshot: Option<WasmSnapshot>,
}

pub struct DistributedWasmDispatcher {
    hosts: Arc<Mutex<HashMap<Uuid, JanusWasmHost>>>,
}

impl DistributedWasmDispatcher {
    pub fn new() -> Self {
        DistributedWasmDispatcher {
            hosts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn spawn_task(&self, task: WasmTask) -> Result<Uuid> {
        let mut host = JanusWasmHost::new(&task.wasm_binary)?;
        if let Some(snapshot) = task.initial_snapshot {
            host.restore_snapshot(snapshot);
        }
        
        let mut hosts = self.hosts.lock().unwrap();
        hosts.insert(task.id, host);
        Ok(task.id)
    }

    pub fn tick_task(&self, task_id: Uuid) -> Result<i32> {
        let mut hosts = self.hosts.lock().unwrap();
        let host = hosts.get_mut(&task_id).ok_or_else(|| anyhow::anyhow!("Task not found"))?;
        host.tick()
    }

    pub fn get_task_snapshot(&self, task_id: Uuid) -> Result<WasmSnapshot> {
        let hosts = self.hosts.lock().unwrap();
        let host = hosts.get(&task_id).ok_or_else(|| anyhow::anyhow!("Task not found"))?;
        Ok(host.get_snapshot())
    }

    pub fn remove_task(&self, task_id: Uuid) {
        let mut hosts = self.hosts.lock().unwrap();
        hosts.remove(&task_id);
    }

    /// Transfers a task by returning its snapshot and removing it from the current dispatcher.
    pub fn transfer_task_snapshot(&self, task_id: Uuid) -> Result<WasmSnapshot> {
        let mut hosts = self.hosts.lock().unwrap();
        let host = hosts.remove(&task_id).ok_or_else(|| anyhow::anyhow!("Task not found for transfer"))?;
        Ok(host.get_snapshot())
    }

    /// Resumes a task from a snapshot, effectively migrating it to this dispatcher.
    pub fn resume_task_from_snapshot(&self, task_id: Uuid, wasm_binary: Vec<u8>, snapshot: WasmSnapshot) -> Result<Uuid> {
        let mut host = JanusWasmHost::new(&wasm_binary)?;
        host.restore_snapshot(snapshot);
        let mut hosts = self.hosts.lock().unwrap();
        hosts.insert(task_id, host);
        Ok(task_id)
    }

