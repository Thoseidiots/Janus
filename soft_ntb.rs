/// soft_ntb.rs
/// ===========
/// Software-defined Non-Transparent Bridge (NTB) for Janus.
///
/// Replaces PCIe NTB hardware with a TCP-based distributed shared memory layer.
/// Presents the same API as JumfMemoryWindow so the rest of the system
/// (JCE, LAS, dispatcher) works without modification.
///
/// Architecture:
///   - Each node runs a SoftNtbServer listening on a TCP port
///   - SoftNtbClient connects to remote nodes
///   - Shared memory is a local Vec<u8> kept in sync via delta messages
///   - Doorbell interrupts become TCP signals (< 50μs on LAN)
///   - Coherency protocol unchanged — JCE runs on top as before
///
/// Latency vs hardware NTB:
///   Hardware PCIe NTB:  1-2 μs
///   This (LAN TCP):     200-500 μs
///   This (localhost):   20-50 μs
///
/// For Janus state migration (WASM snapshot + HBM vector):
///   ~50-100 MB at 1Gbps = 400-800ms — fast enough to feel like teleportation.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, SocketAddr};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

// ── Protocol messages ─────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum NtbMessage {
    /// Write bytes at offset into remote memory window
    Write { offset: usize, data: Vec<u8> },
    /// Read bytes from remote memory window
    Read { offset: usize, len: usize },
    /// Response to a Read
    ReadResponse { data: Vec<u8> },
    /// Doorbell interrupt — equivalent to PCIe doorbell register
    Doorbell { channel: u8, value: u32 },
    /// Sync entire memory window (used on connect)
    FullSync { data: Vec<u8> },
    /// Acknowledge a message
    Ack,
    /// Error response
    Error { message: String },
}

fn encode_msg(msg: &NtbMessage) -> Vec<u8> {
    let payload = bincode::serialize(msg).unwrap_or_default();
    let len = payload.len() as u32;
    let mut buf = len.to_be_bytes().to_vec();
    buf.extend(payload);
    buf
}

fn decode_msg(stream: &mut TcpStream) -> Option<NtbMessage> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).ok()?;
    let len = u32::from_be_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).ok()?;
    bincode::deserialize(&payload).ok()
}

fn send_msg(stream: &mut TcpStream, msg: &NtbMessage) -> bool {
    let encoded = encode_msg(msg);
    stream.write_all(&encoded).is_ok()
}

// ── Doorbell handler ──────────────────────────────────────────────────────────

pub type DoorbellHandler = Box<dyn Fn(u8, u32) + Send + Sync>;

// ── Software NTB Memory Window ────────────────────────────────────────────────

/// Drop-in replacement for JumfMemoryWindow.
/// Backed by a local Vec<u8> that stays in sync with the remote node.
pub struct SoftNtbWindow {
    /// Local copy of the shared memory
    memory: Arc<RwLock<Vec<u8>>>,
    size:   usize,
    /// Connection to the remote node (None = local-only mode)
    remote: Arc<Mutex<Option<TcpStream>>>,
    /// Registered doorbell handlers
    doorbell_handlers: Arc<Mutex<HashMap<u8, DoorbellHandler>>>,
    /// Latency stats
    last_rtt_us: Arc<Mutex<u64>>,
}

impl SoftNtbWindow {
    /// Create a local-only window (no remote connection).
    /// Useful for single-machine testing.
    pub fn local(size: usize) -> Self {
        SoftNtbWindow {
            memory:            Arc::new(RwLock::new(vec![0u8; size])),
            size,
            remote:            Arc::new(Mutex::new(None)),
            doorbell_handlers: Arc::new(Mutex::new(HashMap::new())),
            last_rtt_us:       Arc::new(Mutex::new(0)),
        }
    }

    /// Connect to a remote NTB server.
    pub fn connect(addr: &str, size: usize) -> anyhow::Result<Self> {
        let mut stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true)?; // Disable Nagle — critical for low latency
        stream.set_read_timeout(Some(Duration::from_secs(5)))?;

        // Request full sync on connect
        send_msg(&mut stream, &NtbMessage::FullSync { data: vec![] });
        let response = decode_msg(&mut stream)
            .ok_or_else(|| anyhow::anyhow!("No response on connect"))?;

        let initial_memory = match response {
            NtbMessage::FullSync { data } => {
                let mut mem = data;
                mem.resize(size, 0);
                mem
            }
            _ => vec![0u8; size],
        };

        println!("[SoftNTB] Connected to {} ({} bytes synced)", addr, size);

        Ok(SoftNtbWindow {
            memory:            Arc::new(RwLock::new(initial_memory)),
            size,
            remote:            Arc::new(Mutex::new(Some(stream))),
            doorbell_handlers: Arc::new(Mutex::new(HashMap::new())),
            last_rtt_us:       Arc::new(Mutex::new(0)),
        })
    }

    /// Read bytes from the local memory window.
    /// Same signature as JumfMemoryWindow::read.
    pub fn read<T: Copy>(&self, offset: usize) -> anyhow::Result<T> {
        let size = std::mem::size_of::<T>();
        if offset + size > self.size {
            return Err(anyhow::anyhow!("Offset out of bounds"));
        }
        let mem = self.memory.read().unwrap();
        let bytes = &mem[offset..offset + size];
        Ok(unsafe { std::ptr::read(bytes.as_ptr() as *const T) })
    }

    /// Write bytes to local memory and propagate to remote.
    /// Same signature as JumfMemoryWindow::write.
    pub fn write<T: Copy>(&self, offset: usize, value: T) -> anyhow::Result<()> {
        let size = std::mem::size_of::<T>();
        if offset + size > self.size {
            return Err(anyhow::anyhow!("Offset out of bounds"));
        }

        // Write locally
        {
            let mut mem = self.memory.write().unwrap();
            let bytes = unsafe {
                std::slice::from_raw_parts(&value as *const T as *const u8, size)
            };
            mem[offset..offset + size].copy_from_slice(bytes);
        }

        // Propagate to remote
        self.propagate_write(offset, size);
        Ok(())
    }

    /// Write a raw byte slice — more efficient for large transfers.
    pub fn write_bytes(&self, offset: usize, data: &[u8]) -> anyhow::Result<()> {
        if offset + data.len() > self.size {
            return Err(anyhow::anyhow!("Offset out of bounds"));
        }
        {
            let mut mem = self.memory.write().unwrap();
            mem[offset..offset + data.len()].copy_from_slice(data);
        }
        self.propagate_write(offset, data.len());
        Ok(())
    }

    /// Read a raw byte slice.
    pub fn read_bytes(&self, offset: usize, len: usize) -> anyhow::Result<Vec<u8>> {
        if offset + len > self.size {
            return Err(anyhow::anyhow!("Offset out of bounds"));
        }
        let mem = self.memory.read().unwrap();
        Ok(mem[offset..offset + len].to_vec())
    }

    /// Ring the doorbell on the remote node.
    /// Equivalent to writing to a PCIe doorbell register.
    pub fn ring_doorbell(&self, channel: u8, value: u32) {
        let t0 = Instant::now();
        let mut remote = self.remote.lock().unwrap();
        if let Some(ref mut stream) = *remote {
            send_msg(stream, &NtbMessage::Doorbell { channel, value });
            // Wait for ack to measure RTT
            if let Some(NtbMessage::Ack) = decode_msg(stream) {
                let rtt = t0.elapsed().as_micros() as u64;
                *self.last_rtt_us.lock().unwrap() = rtt;
            }
        }
    }

    /// Register a handler for incoming doorbell interrupts.
    pub fn on_doorbell<F>(&self, channel: u8, handler: F)
    where
        F: Fn(u8, u32) + Send + Sync + 'static,
    {
        self.doorbell_handlers
            .lock()
            .unwrap()
            .insert(channel, Box::new(handler));
    }

    /// Get last measured round-trip time in microseconds.
    pub fn rtt_us(&self) -> u64 {
        *self.last_rtt_us.lock().unwrap()
    }

    /// Snapshot the entire memory window for migration.
    pub fn snapshot(&self) -> Vec<u8> {
        self.memory.read().unwrap().clone()
    }

    /// Restore from a snapshot (used when migrating to this node).
    pub fn restore(&self, data: Vec<u8>) {
        let mut mem = self.memory.write().unwrap();
        let len = data.len().min(self.size);
        mem[..len].copy_from_slice(&data[..len]);
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn propagate_write(&self, offset: usize, len: usize) {
        let mut remote = self.remote.lock().unwrap();
        if let Some(ref mut stream) = *remote {
            let data = {
                let mem = self.memory.read().unwrap();
                mem[offset..offset + len].to_vec()
            };
            send_msg(stream, &NtbMessage::Write { offset, data });
            // Fire-and-forget for writes — don't wait for ack (async propagation)
        }
    }
}

// ── Software NTB Server ───────────────────────────────────────────────────────

/// Runs on each node. Accepts connections from remote nodes.
pub struct SoftNtbServer {
    window:   Arc<SoftNtbWindow>,
    addr:     SocketAddr,
}

impl SoftNtbServer {
    pub fn new(window: Arc<SoftNtbWindow>, addr: SocketAddr) -> Self {
        SoftNtbServer { window, addr }
    }

    /// Start listening. Spawns a thread per connection.
    pub fn start(self) -> anyhow::Result<()> {
        let listener = TcpListener::bind(self.addr)?;
        println!("[SoftNTB] Server listening on {}", self.addr);

        let window = self.window.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(mut conn) => {
                        let _ = conn.set_nodelay(true);
                        let w = window.clone();
                        thread::spawn(move || {
                            Self::handle_connection(conn, w);
                        });
                    }
                    Err(e) => eprintln!("[SoftNTB] Accept error: {}", e),
                }
            }
        });

        Ok(())
    }

    fn handle_connection(mut stream: TcpStream, window: Arc<SoftNtbWindow>) {
        let peer = stream.peer_addr().unwrap_or("unknown".parse().unwrap());
        println!("[SoftNTB] Connection from {}", peer);

        loop {
            let msg = match decode_msg(&mut stream) {
                Some(m) => m,
                None    => break,
            };

            match msg {
                NtbMessage::Write { offset, data } => {
                    // Remote wrote to shared memory — update local copy
                    let mut mem = window.memory.write().unwrap();
                    let end = (offset + data.len()).min(mem.len());
                    if offset < mem.len() {
                        mem[offset..end].copy_from_slice(&data[..end - offset]);
                    }
                    // No ack for writes — fire and forget
                }

                NtbMessage::Read { offset, len } => {
                    let data = window.read_bytes(offset, len)
                        .unwrap_or_default();
                    send_msg(&mut stream, &NtbMessage::ReadResponse { data });
                }

                NtbMessage::FullSync { .. } => {
                    // Send our full memory to the connecting node
                    let data = window.snapshot();
                    send_msg(&mut stream, &NtbMessage::FullSync { data });
                }

                NtbMessage::Doorbell { channel, value } => {
                    // Fire registered handler
                    let handlers = window.doorbell_handlers.lock().unwrap();
                    if let Some(handler) = handlers.get(&channel) {
                        handler(channel, value);
                    }
                    send_msg(&mut stream, &NtbMessage::Ack);
                }

                _ => {}
            }
        }

        println!("[SoftNTB] Connection from {} closed", peer);
    }
}

// ── Janus Teleporter ─────────────────────────────────────────────────────────

/// High-level API for migrating Janus state between nodes.
/// Wraps the low-level SoftNtbWindow with Janus-specific semantics.
pub struct JanusTeleporter {
    window: Arc<SoftNtbWindow>,
}

/// Everything needed to reconstruct Janus on another machine.
#[derive(Serialize, Deserialize, Debug)]
pub struct JanusStateSnapshot {
    /// WASM task snapshots (task_id -> binary snapshot)
    pub wasm_tasks:    HashMap<String, Vec<u8>>,
    /// HBM memory vector (serialized tensor bytes)
    pub hbm_memory:    Vec<u8>,
    /// Skill tree state (JSON)
    pub skill_state:   String,
    /// Working context (JSON)
    pub context:       String,
    /// Timestamp
    pub captured_at:   u64,
    /// Source node identifier
    pub source_node:   String,
}

impl JanusTeleporter {
    pub fn new(window: Arc<SoftNtbWindow>) -> Self {
        JanusTeleporter { window }
    }

    /// Serialize and write a Janus state snapshot into the shared window.
    /// The remote node can then read and restore it.
    pub fn send_state(&self, snapshot: &JanusStateSnapshot) -> anyhow::Result<()> {
        let t0 = Instant::now();
        let data = bincode::serialize(snapshot)?;
        let size = data.len();

        // Write size header (8 bytes) then data
        self.window.write::<u64>(0, size as u64)?;
        self.window.write_bytes(8, &data)?;

        // Ring doorbell channel 1 = "state ready"
        self.window.ring_doorbell(1, size as u32);

        let elapsed = t0.elapsed();
        println!(
            "[Teleporter] Sent {:.1} MB in {:.0}ms (RTT: {}μs)",
            size as f64 / 1e6,
            elapsed.as_millis(),
            self.window.rtt_us()
        );
        Ok(())
    }

    /// Read a Janus state snapshot from the shared window.
    pub fn receive_state(&self) -> anyhow::Result<JanusStateSnapshot> {
        let size = self.window.read::<u64>(0)? as usize;
        let data = self.window.read_bytes(8, size)?;
        let snapshot = bincode::deserialize(&data)?;
        Ok(snapshot)
    }

    /// Register a callback for when a state snapshot arrives.
    pub fn on_state_received<F>(&self, callback: F)
    where
        F: Fn(u8, u32) + Send + Sync + 'static,
    {
        self.window.on_doorbell(1, callback);
    }
}

// ── Quick test ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::SocketAddr;

    #[test]
    fn test_local_window_read_write() {
        let window = SoftNtbWindow::local(1024);
        window.write::<u32>(0, 42).unwrap();
        let val: u32 = window.read(0).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn test_snapshot_restore() {
        let w1 = SoftNtbWindow::local(256);
        w1.write::<u64>(0, 0xDEADBEEF).unwrap();
        let snap = w1.snapshot();

        let w2 = SoftNtbWindow::local(256);
        w2.restore(snap);
        let val: u64 = w2.read(0).unwrap();
        assert_eq!(val, 0xDEADBEEF);
    }

    #[test]
    fn test_loopback_server_client() {
        let addr: SocketAddr = "127.0.0.1:19876".parse().unwrap();

        // Server
        let server_window = Arc::new(SoftNtbWindow::local(1024));
        server_window.write::<u32>(0, 99).unwrap();
        let server = SoftNtbServer::new(server_window.clone(), addr);
        server.start().unwrap();

        std::thread::sleep(std::time::Duration::from_millis(50));

        // Client connects and gets full sync
        let client_window = SoftNtbWindow::connect("127.0.0.1:19876", 1024).unwrap();
        let val: u32 = client_window.read(0).unwrap();
        assert_eq!(val, 99, "Client should see server's initial value");

        // Client writes, server should see it
        client_window.write::<u32>(4, 77).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let server_val: u32 = server_window.read(4).unwrap();
        assert_eq!(server_val, 77, "Server should see client's write");

        println!("RTT: {}μs", client_window.rtt_us());
    }

    #[test]
    fn test_teleporter_snapshot() {
        let window = Arc::new(SoftNtbWindow::local(1024 * 1024));
        let teleporter = JanusTeleporter::new(window);

        let snapshot = JanusStateSnapshot {
            wasm_tasks:  HashMap::new(),
            hbm_memory:  vec![1, 2, 3, 4],
            skill_state: r#"{"language": 0.5}"#.to_string(),
            context:     "{}".to_string(),
            captured_at: 0,
            source_node: "node-a".to_string(),
        };

        teleporter.send_state(&snapshot).unwrap();
        let received = teleporter.receive_state().unwrap();
        assert_eq!(received.hbm_memory, vec![1, 2, 3, 4]);
        assert_eq!(received.source_node, "node-a");
    }
}
