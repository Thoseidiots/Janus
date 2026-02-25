use tonic::{transport::Server, Request, Response, Status};
use nexus_service::{nexus_service_server::{NexusService, NexusServiceServer}, NexusCommand, NexusResponse, NexusStateUpdate, Empty};
use janus_core::{JanusState, IdentityContract, Event, Task};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use chrono::Utc;
use openraft::{Config, Raft};
use openraft::error::RaftError;
use log::{info, error};
use env_logger;

pub mod nexus_service {
    tonic::include_proto!("nexus");
}

mod raft_app;
use raft_app::{TypeConfig, JanusRaftStorage, JanusRaftNetwork, AppData, AppDataResponse, NodeId, LogEntry};

mod dispatcher;
use dispatcher::{DistributedWasmDispatcher, WasmTask};
use uuid::Uuid;

#[derive(Debug)]
pub struct MyNexusService {
    janus_state: Arc<Mutex<JanusState>>,
    state_update_senders: Arc<Mutex<Vec<mpsc::Sender<NexusStateUpdate>>>>,
    raft: Arc<Raft<TypeConfig>>,
    wasm_dispatcher: Arc<DistributedWasmDispatcher>,
}

impl MyNexusService {
    pub fn new(janus_state: JanusState, raft: Arc<Raft<TypeConfig>>, wasm_dispatcher: Arc<DistributedWasmDispatcher>) -> Self {
        MyNexusService {
            janus_state: Arc::new(Mutex::new(janus_state)),
            state_update_senders: Arc::new(Mutex::new(Vec::new())),
            raft,
            wasm_dispatcher,
        }
    }

    // Helper to broadcast state updates
    pub fn broadcast_state_update(&self, update: NexusStateUpdate) {
        let mut senders = self.state_update_senders.lock().unwrap();
        senders.retain(|sender| {
            match sender.try_send(update.clone()) {
                Ok(_) => true,
                Err(e) => {
                    eprintln!("Failed to send state update: {:?}", e);
                    false // Remove disconnected sender
                }
            }
        });
    }
}

#[tonic::async_trait]
impl NexusService for MyNexusService {
    async fn execute_command(&self, request: Request<NexusCommand>) -> Result<Response<NexusResponse>, Status> {
        let command = request.into_inner();
        info!("Received command: {:?}", command);

        // Simulate a state change and apply it via Raft
        let app_data = match command.command_type.as_str() {
            "add_event" => {
                let event: Event = serde_json::from_str(&command.payload).map_err(|e| Status::invalid_argument(format!("Invalid event payload: {}", e)))?;
                AppData::AddEvent { event }
            },
            "add_task" => {
                let task: Task = serde_json::from_str(&command.payload).map_err(|e| Status::invalid_argument(format!("Invalid task payload: {}", e)))?;
                AppData::AddTask { task }
            },
            "update_identity" => {
                let identity: IdentityContract = serde_json::from_str(&command.payload).map_err(|e| Status::invalid_argument(format!("Invalid identity payload: {}", e)))?;
                AppData::UpdateIdentity { identity }
            },
            "spawn_wasm_task" => {
                // Special handling for WASM tasks (could also be a Raft command)
                let task_id = Uuid::new_v4();
                // For demonstration, we'll assume the payload contains the WASM binary (base64 encoded)
                // In a real scenario, this would be more complex.
                let wasm_binary = vec![]; // Placeholder
                let task = WasmTask { id: task_id, wasm_binary, initial_snapshot: None };
                self.wasm_dispatcher.spawn_task(task).map_err(|e| Status::internal(format!("Failed to spawn WASM task: {}", e)))?;
                
                let reply = NexusResponse {
                    success: true,
                    message: format!("WASM task spawned with ID: {}", task_id),
                    result_payload: serde_json::json!({ "task_id": task_id.to_string() }).to_string(),
                };
                return Ok(Response::new(reply));
            },
            _ => return Err(Status::unimplemented(format!("Unknown command type: {}", command.command_type))),
        };

        let log_entry = LogEntry::new_payload(app_data);
        match self.raft.client_write(log_entry).await {
            Ok(response) => {
                info!("Raft client_write successful: {:?}", response);
                let mut janus_state = self.janus_state.lock().unwrap();
                // In a real scenario, the state machine would update janus_state.
                // For now, we'll manually update for demonstration.
                // This part needs to be properly integrated with Raft's state machine application.
                let response_message = format!("Command \'{}\' processed via Raft.", command.command_type);
                let result_payload = serde_json::json!({ "status": "success", "command_type": command.command_type, "raft_log_id": response.log_id }).to_string();

                // Broadcast state update (this should come from the state machine applying the log)
                self.broadcast_state_update(NexusStateUpdate {
                    update_type: "command_applied".to_string(),
                    state_payload: serde_json::to_string(&*janus_state).unwrap_or_default(),
                });

                let reply = NexusResponse {
                    success: true,
                    message: response_message,
                    result_payload,
                };
                Ok(Response::new(reply))
            },
            Err(e) => {
                error!("Raft client_write failed: {:?}", e);
                Err(Status::internal(format!("Failed to apply command via Raft: {}", e)))
            }
        }
    }

    type StreamStateUpdatesStream = ReceiverStream<NexusStateUpdate>;

    async fn stream_state_updates(&self, _request: Request<Empty>) -> Result<Response<Self::StreamStateUpdatesStream>, Status> {
        let (tx, rx) = mpsc::channel(4);
        self.state_update_senders.lock().unwrap().push(tx);
        info!("Client connected for state updates.");
        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let node_id = 0; // This node's ID. In a real cluster, this would be configured.
    let addr = "[::1]:50051".parse()?;

    // Initialize a dummy JanusState for the Nexus Core
    let identity = IdentityContract {
        id: format!("NEXUS-CORE-{}", node_id).to_string(),
        creation_timestamp: Utc::now(),
        design_constraints: vec!["Distributed operation".to_string()],
        role_definition: "Nexus Core for Janus Distributed System".to_string(),
        non_negotiable_boundaries: vec![],
    };
    let janus_state = JanusState::new(identity);

    // Raft setup
    let config = Config::build("default_config").validate()?;
    let config = Arc::new(config);

    let janus_state_machine = Arc::new(Mutex::new(raft_app::JanusRaftStateMachine {
        janus_state: janus_state.clone(),
        ..Default::default()
    }));

    let storage = Arc::new(JanusRaftStorage {
        id: node_id,
        janus_state_machine: janus_state_machine.clone(),
    });
    let network = Arc::new(JanusRaftNetwork {});

    let raft = Arc::new(Raft::new(node_id, config, network, storage));
    let wasm_dispatcher = Arc::new(DistributedWasmDispatcher::new());

    let nexus_service = MyNexusService::new(janus_state, raft.clone(), wasm_dispatcher);

    info!("NexusService listening on {}", addr);

    Server::builder()
        .add_service(NexusServiceServer::new(nexus_service))
        .serve(addr)
        .await?;

    Ok(())
}
