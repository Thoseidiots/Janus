/// Generation-tagged entity identifier.
/// Upper 32 bits: generation counter (detects use-after-free).
/// Lower 32 bits: index into sparse array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(pub u64);

impl EntityId {
    #[inline]
    pub fn index(self) -> u32 {
        self.0 as u32
    }

    #[inline]
    pub fn generation(self) -> u32 {
        (self.0 >> 32) as u32
    }

    fn new(index: u32, generation: u32) -> Self {
        EntityId(((generation as u64) << 32) | (index as u64))
    }
}

/// Allocates and recycles entity indices with generation tracking.
pub struct EntityAllocator {
    /// Generation counter per slot index.
    generations: Vec<u32>,
    /// Free list of recycled indices.
    free_list: Vec<u32>,
}

impl EntityAllocator {
    pub fn new() -> Self {
        EntityAllocator {
            generations: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Allocate a new EntityId, reusing a recycled index if available.
    pub fn allocate(&mut self) -> EntityId {
        if let Some(index) = self.free_list.pop() {
            let gen = self.generations[index as usize];
            EntityId::new(index, gen)
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            EntityId::new(index, 0)
        }
    }

    /// Free an entity. Increments the generation so old IDs become stale.
    pub fn free(&mut self, id: EntityId) {
        let index = id.index() as usize;
        if index < self.generations.len() && self.generations[index] == id.generation() {
            self.generations[index] = self.generations[index].wrapping_add(1);
            self.free_list.push(index as u32);
        }
    }

    /// Returns true if the entity is still alive (generation matches).
    pub fn is_alive(&self, id: EntityId) -> bool {
        let index = id.index() as usize;
        index < self.generations.len() && self.generations[index] == id.generation()
    }
}

impl Default for EntityAllocator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_returns_unique_ids() {
        let mut alloc = EntityAllocator::new();
        let a = alloc.allocate();
        let b = alloc.allocate();
        assert_ne!(a, b);
    }

    #[test]
    fn is_alive_after_allocate() {
        let mut alloc = EntityAllocator::new();
        let id = alloc.allocate();
        assert!(alloc.is_alive(id));
    }

    #[test]
    fn not_alive_after_free() {
        let mut alloc = EntityAllocator::new();
        let id = alloc.allocate();
        alloc.free(id);
        assert!(!alloc.is_alive(id));
    }

    #[test]
    fn recycled_index_has_new_generation() {
        let mut alloc = EntityAllocator::new();
        let id1 = alloc.allocate();
        alloc.free(id1);
        let id2 = alloc.allocate();
        // Same index, different generation
        assert_eq!(id1.index(), id2.index());
        assert_ne!(id1.generation(), id2.generation());
        assert!(!alloc.is_alive(id1));
        assert!(alloc.is_alive(id2));
    }

    #[test]
    fn double_free_is_ignored() {
        let mut alloc = EntityAllocator::new();
        let id = alloc.allocate();
        alloc.free(id);
        alloc.free(id); // should not panic or corrupt state
        let id2 = alloc.allocate();
        assert!(alloc.is_alive(id2));
    }
}
