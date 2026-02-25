use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use anyhow::{Result, Context};
use crate::jumf::JumfMemoryWindow;

/// Represents a logically divided region of the JUMF shared memory.
pub struct MemoryRegion {
    pub id: u64,
    pub start_offset: usize,
    pub size: usize,
    /// Atomic flag for ownership: 0 = unowned, 1 = Node A, 2 = Node B.
    owner: AtomicU8,
}

impl MemoryRegion {
    pub fn new(id: u64, start_offset: usize, size: usize) -> Self {
        MemoryRegion {
            id,
            start_offset,
            size,
            owner: AtomicU8::new(0),
        }
    }

    /// Try to acquire ownership of this region using Compare-and-Swap (CAS).
    pub fn try_acquire(&self, node_id: u8) -> bool {
        self.owner.compare_exchange(
            0,
            node_id,
            Ordering::Acquire,
            Ordering::Relaxed
        ).is_ok()
    }

    /// Release ownership of this region.
    pub fn release(&self, node_id: u8) -> bool {
        self.owner.compare_exchange(
            node_id,
            0,
            Ordering::Release,
            Ordering::Relaxed
        ).is_ok()
    }

    pub fn current_owner(&self) -> u8 {
        self.owner.load(Ordering::Acquire)
    }
}

/// The Janus Coherency Engine (JCE) manages memory region ownership across nodes.
pub struct JanusCoherencyEngine {
    jumf_window: Arc<Mutex<JumfMemoryWindow>>,
    regions: Vec<MemoryRegion>,
    node_id: u8,
}

impl JanusCoherencyEngine {
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    pub fn new(jumf_window: Arc<Mutex<JumfMemoryWindow>>, node_id: u8, total_size: usize, region_size: usize) -> Self {
        let num_regions = total_size / region_size;
        let mut regions = Vec::with_capacity(num_regions);
        for i in 0..num_regions {
            regions.push(MemoryRegion::new(i as u64, i * region_size, region_size));
        }

        JanusCoherencyEngine {
            jumf_window,
            regions,
            node_id,
        }
    }

    /// Acquire a region, ensuring no other node is writing to it.
    pub fn acquire(&self, region_id: u64) -> Result<()> {
        let region = self.regions.get(region_id as usize)
            .context("Region ID out of bounds")?;

        // Spin-lock for demonstration; in production, this would use a doorbell signal/wait.
        while !region.try_acquire(self.node_id) {
            if region.current_owner() == self.node_id {
                return Ok(()); // Already owned
            }
            // std::thread::yield_now(); // In a real system, we'd wait for a signal
        }
        Ok(())
    }

    pub fn release(&self, region_id: u64) -> Result<()> {
        let region = self.regions.get(region_id as usize)
            .context("Region ID out of bounds")?;
        
        if !region.release(self.node_id) {
            return Err(anyhow::anyhow!("Failed to release region: not the owner"));
        }
        Ok(())
    }

    /// Perform a coherent write to the shared memory.
    pub fn write_coherent<T: Copy>(&self, region_id: u64, offset: usize, value: T) -> Result<()> {
        self.acquire(region_id)?;
        let mut window = self.jumf_window.lock().unwrap();
        let region = &self.regions[region_id as usize];
        window.write(region.start_offset + offset, value)?;
        // We don't necessarily release immediately to allow batch writes
        Ok(())
    }

    /// Perform a coherent read from the shared memory.
    pub fn read_coherent<T: Copy>(&self, region_id: u64, offset: usize) -> Result<T> {
        self.acquire(region_id)?;
        let window = self.jumf_window.lock().unwrap();
        let region = &self.regions[region_id as usize];
        window.read(region.start_offset + offset)
    }
}
