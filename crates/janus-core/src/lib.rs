use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct IdentityContract {
    pub id: String,
    pub creation_timestamp: DateTime<Utc>,
    pub design_constraints: Vec<String>,
    pub role_definition: String,
    pub non_negotiable_boundaries: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Aborted,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Task {
    pub id: Uuid,
    pub description: String,
    pub status: TaskStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Event {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub event_type: String,
    pub data: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct JanusState {
    pub identity: IdentityContract,
    pub active_tasks: Vec<Task>,
    pub event_log: Vec<Event>,
    pub metadata: HashMap<String, String>,
}

impl JanusState {
    pub fn new(identity: IdentityContract) -> Self {
        Self {
            identity,
            active_tasks: Vec::new(),
            event_log: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_event(&mut self, event_type: &str, data: serde_json::Value) {
        let event = Event {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            event_type: event_type.to_string(),
            data,
        };
        self.event_log.push(event);
    }

    pub fn add_task(&mut self, description: &str) -> Uuid {
        let task = Task {
            id: Uuid::new_v4(),
            description: description.to_string(),
            status: TaskStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let id = task.id;
        self.active_tasks.push(task);
        id
    }
}

use std::process::{Command, Stdio, Child};
use std::io::{Write, BufReader, BufRead};

pub struct BrainBridge {
    child: Child,
}

impl BrainBridge {
    pub fn new() -> anyhow::Result<Self> {
        let child = Command::new("python3")
            .arg("-m")
            .arg("janus_brain.bridge")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        Ok(Self { child })
    }

    pub fn send_command(&mut self, cmd: serde_json::Value) -> anyhow::Result<serde_json::Value> {
        let stdin = self.child.stdin.as_mut().ok_or_else(|| anyhow::anyhow!("Failed to open stdin"))?;
        let stdout = self.child.stdout.as_mut().ok_or_else(|| anyhow::anyhow!("Failed to open stdout"))?;
        
        let cmd_str = serde_json::to_string(&cmd)?;
        writeln!(stdin, "{}", cmd_str)?;
        
        let mut reader = BufReader::new(stdout);
        let mut response_str = String::new();
        reader.read_line(&mut response_str)?;
        
        let response: serde_json::Value = serde_json::from_str(&response_str)?;
        Ok(response)
    }
}
