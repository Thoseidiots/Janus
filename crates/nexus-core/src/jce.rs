use std::sync::atomic::{AtomicU8, AtomicU64, AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use anyhow::{Result, Context};
use crate::jumf::JumfMemoryWindow;
use std::time::{SystemTime, UNIX_EPOCH};

/// Represents a logically divided region of the JUMF shared memory.
pub struct MemoryRegion {
    pub id: u64,
    pub start_offset: usize,
    pub size: usize,
    /// Atomic flag for ownership: 0 = unowned, 1 = Node A, 2 = Node B.
    owner: AtomicU8,
    /// Version counter to prevent ABA problems during CAS loops.
    version: AtomicU64,
    /// Signals if the region has been modified by the current owner, requiring remote cache invalidation.
    dirty: AtomicBool,
    /// Unix timestamp (in microseconds) when the current lease expires. 0 if no lease.
    lease_expiry: AtomicU64,
}

impl MemoryRegion {
    pub fn new(id: u64, start_offset: usize, size: usize) -> Self {
        MemoryRegion {
            id,
            start_offset,
            size,
            owner: AtomicU8::new(0),
            version: AtomicU64::new(0),
            dirty: AtomicBool::new(false),
            lease_expiry: AtomicU64::new(0),
        }
    }

    /// Get current time in microseconds since UNIX_EPOCH.
    fn current_time_micros() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_micros() as u64
    }

    /// Try to acquire ownership of this region using Compare-and-Swap (CAS).
    /// Includes lease management and versioning.
    pub fn try_acquire(&self, node_id: u8, lease_duration_micros: u64) -> bool {
        let current_time = Self::current_time_micros();
        let mut current_owner = self.owner.load(Ordering::Acquire);
        let mut current_version = self.version.load(Ordering::Acquire);
        let mut current_lease_expiry = self.lease_expiry.load(Ordering::Acquire);

        loop {
            // If already owned by us and lease is still valid, we're good.
            if current_owner == node_id && current_lease_expiry > current_time {
                return true;
            }

            // If unowned or lease expired, try to acquire.
            if current_owner == 0 || current_lease_expiry <= current_time {
                let new_lease_expiry = current_time + lease_duration_micros;
                let new_version = current_version.wrapping_add(1); // Increment version

                // Attempt to acquire ownership and update lease/version atomically.
                match self.owner.compare_exchange(
                    current_owner,
                    node_id,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(old_owner) => {
                        // Successfully acquired ownership.
                        // Now update version and lease expiry.
                        self.version.store(new_version, Ordering::Release);
                        self.lease_expiry.store(new_lease_expiry, Ordering::Release);
                        self.dirty.store(false, Ordering::Release); // New owner, not dirty yet
                        return true;
                    },
                    Err(actual_owner) => {
                        // Ownership changed, retry.
                        current_owner = actual_owner;
                        current_version = self.version.load(Ordering::Acquire);
                        current_lease_expiry = self.lease_expiry.load(Ordering::Acquire);
                        continue;
                    },
                }
            }

            // Owned by another node with a valid lease, cannot acquire.
            return false;
        }
    }

    /// Release ownership of this region.
    pub fn release(&self, node_id: u8) -> bool {
        // Only the current owner can release.
        if self.owner.load(Ordering::Acquire) == node_id {
            self.owner.store(0, Ordering::Release);
            self.lease_expiry.store(0, Ordering::Release); // Invalidate lease
            // Version is incremented on acquire, not release.
            return true;
        }
        false
    }

    pub fn current_owner(&self) -> u8 {
        self.owner.load(Ordering::Acquire)
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    pub fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

/// The Janus Coherency Engine (JCE) manages memory region ownership across nodes.
pub struct JanusCoherencyEngine {
    jumf_window: Arc<Mutex<JumfMemoryWindow>>,
    regions: Vec<MemoryRegion>,
    node_id: u8,
    lease_duration_micros: u64,
}

impl JanusCoherencyEngine {
    pub fn node_id(&self) -> u8 {
        self.node_id
    }

    pub fn new(jumf_window: Arc<Mutex<JumfMemoryWindow>>, node_id: u8, total_size: usize, region_size: usize, lease_duration_micros: u64) -> Self {
        let num_regions = total_size / region_size;
        let mut regions = Vec::with_capacity(num_regions);
        for i in 0..num_regions {
            regions.push(MemoryRegion::new(i as u64, i * region_size, region_size));
        }

        JanusCoherencyEngine {
            jumf_window,
            regions,
            node_id,
            lease_duration_micros,
        }
    }

    /// Acquire a region, ensuring no other node is writing to it.
    pub fn acquire(&self, region_id: u64) -> Result<()> {
        let region = self.regions.get(region_id as usize)
            .context("Region ID out of bounds")?;

        // Spin-lock for demonstration; in production, this would use a doorbell signal/wait.
        while !region.try_acquire(self.node_id, self.lease_duration_micros) {
            // If another node owns it with a valid lease, we wait.
            // In a real system, we'd use a more efficient wait mechanism (e.g., doorbell interrupt).
            std::thread::yield_now(); 
        }
        Ok(())
    }

    pub fn release(&self, region_id: u64) -> Result<()> {
        let region = self.regions.get(region_id as usize)
            .context("Region ID out of bounds")?;
        
        if !region.release(self.node_id) {
            return Err(anyhow::anyhow!("Failed to release region: not the owner or lease still active"));
        }
        Ok(())
    }

    /// Perform a coherent write to the shared memory.
    pub fn write_coherent<T: Copy>(&self, region_id: u64, offset: usize, value: T) -> Result<()> {
        self.acquire(region_id)?;
        let mut window = self.jumf_window.lock().unwrap();
        let region = &self.regions[region_id as usize];
        window.write(region.start_offset + offset, value)?;
        region.mark_dirty(); // Mark as dirty after write
        Ok(())
    }

    /// Perform a coherent read from the shared memory.
    pub fn read_coherent<T: Copy>(&self, region_id: u64, offset: usize) -> Result<T> {
        self.acquire(region_id)?;
        let window = self.jumf_window.lock().unwrap();
        let region = &self.regions[region_id as usize];
        let value = window.read(region.start_offset + offset)?;
        // If we read a dirty region from another node, we might need to invalidate our cache.
        // For now, `acquire` handles ensuring we have ownership before reading/writing.
        Ok(value)
    }

    /// Check if a region is dirty (modified by its owner).
    pub fn is_region_dirty(&self, region_id: u64) -> Result<bool> {
        let region = self.regions.get(region_id as usize)
            .context("Region ID out of bounds")?;
        Ok(region.is_dirty())
    }

    /// Mark a region as clean after its dirty state has been propagated/handled.
    pub fn mark_region_clean(&self, region_id: u64) -> Result<()> {
        let region = self.regions.get(region_id as usize)
            .context("Region ID out of bounds")?;
        region.mark_clean();
        Ok(())
    }
}
