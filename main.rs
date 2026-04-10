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

mod jumf;
use jumf::{JumfMemoryWindow, JumfRpcHandler, JumfUnifiedAllocator};

mod jce;
use jce::JanusCoherencyEngine;

mod las;
use las::{LocalityAwareScheduler, TaskDescriptor, NodeMetrics};

mod soft_ntb;
use soft_ntb::{SoftNtbWindow, SoftNtbServer, JanusTeleporter};

#[derive(Debug)]
pub struct MyNexusService {
    janus_state: Arc<Mutex<JanusState>>,
    state_update_senders: Arc<Mutex<Vec<mpsc::Sender<NexusStateUpdate>>>>,
    raft: Arc<Raft<TypeConfig>>,
    wasm_dispatcher: Arc<DistributedWasmDispatcher>,
    jce: Arc<JanusCoherencyEngine>,
    las: Arc<LocalityAwareScheduler>,
}

impl MyNexusService {
    pub fn new(
        janus_state: JanusState, 
        raft: Arc<Raft<TypeConfig>>, 
        wasm_dispatcher: Arc<DistributedWasmDispatcher>,
        jce: Arc<JanusCoherencyEngine>,
        las: Arc<LocalityAwareScheduler>
    ) -> Self {
        MyNexusService {
            janus_state: Arc::new(Mutex::new(janus_state)),
            state_update_senders: Arc::new(Mutex::new(Vec::new())),
            raft,
            wasm_dispatcher,
            jce,
            las,
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
                let task_id = Uuid::new_v4();
                let wasm_binary = vec![]; // Placeholder
                
                // Invoke Locality-Aware Scheduler (LAS) to find the best node.
                let task_desc = TaskDescriptor {
                    task_id,
                    required_memory_regions: vec![0, 1], // Example regions
                    estimated_compute_cost: 1000,
                };
                let target_node = self.las.schedule_task(&task_desc).map_err(|e| Status::internal(format!("Scheduling failed: {}", e)))?;
                
                if target_node == self.jce.node_id() {
                    let task = WasmTask { id: task_id, wasm_binary: wasm_binary.clone(), initial_snapshot: None };
                    self.wasm_dispatcher.spawn_task(task).map_err(|e| Status::internal(format!("Failed to spawn WASM task: {}", e)))?;
                } else {
                    // Simulate migration: take a snapshot, send it, and resume on target.
                    // In a real system, this would involve JUMF RPC to the target node.
                    info!("Simulating WASM task migration: {} from Node {} to Node {}", task_id, self.jce.node_id(), target_node);
                    // For demonstration, we'll just simulate the transfer and resume locally.
                    // In a real scenario, the snapshot would be sent over JUMF RPC.
                    let snapshot = self.wasm_dispatcher.transfer_task_snapshot(task_id).unwrap_or_else(|_| {
                        // If task wasn't found (e.g., first spawn), create a dummy snapshot.
                        janus_wasm::WasmSnapshot { memory: vec![], registers: HashMap::new() }
                    });
                    // This would be called on the *target* node's dispatcher.
                    self.wasm_dispatcher.resume_task_from_snapshot(task_id, wasm_binary.clone(), snapshot).map_err(|e| Status::internal(format!("Failed to resume WASM task on target: {}", e)))?;
                }
                
                let reply = NexusResponse {
                    success: true,
                    message: format!("WASM task {} scheduled on Node {}", task_id, target_node),
                    result_payload: serde_json::json!({ "task_id": task_id.to_string(), "node": target_node }).to_string(),
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
    
    // Initialize JCE and LAS.
    // Try hardware NTB first, fall back to software NTB over TCP.
    let ntb_size = 1024 * 1024 * 1024; // 1GB shared window
    let soft_window = Arc::new(SoftNtbWindow::local(ntb_size));

    // Start soft NTB server so other nodes can connect
    let ntb_addr = "0.0.0.0:19840".parse().unwrap();
    let server = SoftNtbServer::new(soft_window.clone(), ntb_addr);
    if let Err(e) = server.start() {
        eprintln!("[SoftNTB] Server failed to start: {}", e);
    }

    // Teleporter for migrating Janus state to other nodes
    let teleporter = Arc::new(JanusTeleporter::new(soft_window.clone()));

    // JCE uses a dummy JumfMemoryWindow for now — coherency runs over soft_ntb
    let jce_window = Arc::new(Mutex::new(
        JumfMemoryWindow::map("/dev/ntb0", ntb_size)
            .unwrap_or_else(|_| {
                info!("Hardware NTB not found — using software NTB on port 19840");
                // Return a minimal dummy — JCE will use soft_ntb for actual transfers
                JumfMemoryWindow::map("/dev/zero", 4096)
                    .unwrap_or_else(|_| unsafe { std::mem::zeroed() })
            })
    ));

    let jce = Arc::new(JanusCoherencyEngine::new(
        jce_window,
        node_id as u8,
        ntb_size,
        4096,
        500,
    ));
    let las = Arc::new(LocalityAwareScheduler::new(jce.clone()));

    let nexus_service = MyNexusService::new(janus_state, raft.clone(), wasm_dispatcher, jce, las);

    info!("NexusService listening on {}", addr);

    Server::builder()
        .add_service(NexusServiceServer::new(nexus_service))
        .serve(addr)
        .await?;

    Ok(())
}
