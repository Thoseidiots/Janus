use crate::entity::EntityId;

/// Marker trait for all component types.
pub trait Component: 'static + Send + Sync {}

/// Sparse-set component storage providing O(1) insert/remove and contiguous iteration.
pub struct ComponentStorage<T: Component> {
    /// Dense array of component values.
    dense: Vec<T>,
    /// Parallel dense array of entity IDs (same order as `dense`).
    dense_ids: Vec<EntityId>,
    /// Sparse array: indexed by entity index, holds the dense index (if present).
    sparse: Vec<Option<usize>>,
}

impl<T: Component> ComponentStorage<T> {
    pub fn new() -> Self {
        ComponentStorage {
            dense: Vec::new(),
            dense_ids: Vec::new(),
            sparse: Vec::new(),
        }
    }

    /// Insert or overwrite a component for the given entity. O(1) amortized.
    pub fn insert(&mut self, id: EntityId, component: T) {
        let idx = id.index() as usize;
        // Grow sparse array if needed.
        if idx >= self.sparse.len() {
            self.sparse.resize(idx + 1, None);
        }
        if let Some(dense_idx) = self.sparse[idx] {
            // Overwrite existing.
            self.dense[dense_idx] = component;
        } else {
            let dense_idx = self.dense.len();
            self.dense.push(component);
            self.dense_ids.push(id);
            self.sparse[idx] = Some(dense_idx);
        }
    }

    /// Remove the component for the given entity. O(1) via swap-remove.
    pub fn remove(&mut self, id: EntityId) -> Option<T> {
        let idx = id.index() as usize;
        if idx >= self.sparse.len() {
            return None;
        }
        let dense_idx = self.sparse[idx].take()?;
        let last = self.dense.len() - 1;
        if dense_idx != last {
            // Swap with last element and update sparse pointer for the swapped entity.
            self.dense.swap(dense_idx, last);
            self.dense_ids.swap(dense_idx, last);
            let swapped_entity_idx = self.dense_ids[dense_idx].index() as usize;
            self.sparse[swapped_entity_idx] = Some(dense_idx);
        }
        self.dense_ids.pop();
        Some(self.dense.pop().unwrap())
    }

    /// Get a shared reference to the component for the given entity.
    pub fn get(&self, id: EntityId) -> Option<&T> {
        let idx = id.index() as usize;
        let dense_idx = *self.sparse.get(idx)?.as_ref()?;
        Some(&self.dense[dense_idx])
    }

    /// Get a mutable reference to the component for the given entity.
    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut T> {
        let idx = id.index() as usize;
        let dense_idx = *self.sparse.get(idx)?.as_ref()?;
        Some(&mut self.dense[dense_idx])
    }

    /// Iterate over all (EntityId, &T) pairs in contiguous dense order.
    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &T)> {
        self.dense_ids.iter().copied().zip(self.dense.iter())
    }

    /// Number of components stored.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }
}

impl<T: Component> Default for ComponentStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::EntityAllocator;

    #[derive(Debug, PartialEq)]
    struct Health(i32);
    impl Component for Health {}

    #[test]
    fn insert_and_get() {
        let mut alloc = EntityAllocator::new();
        let mut storage: ComponentStorage<Health> = ComponentStorage::new();
        let e = alloc.allocate();
        storage.insert(e, Health(100));
        assert_eq!(storage.get(e), Some(&Health(100)));
    }

    #[test]
    fn overwrite_component() {
        let mut alloc = EntityAllocator::new();
        let mut storage: ComponentStorage<Health> = ComponentStorage::new();
        let e = alloc.allocate();
        storage.insert(e, Health(50));
        storage.insert(e, Health(99));
        assert_eq!(storage.get(e), Some(&Health(99)));
        assert_eq!(storage.len(), 1);
    }

    #[test]
    fn remove_returns_component() {
        let mut alloc = EntityAllocator::new();
        let mut storage: ComponentStorage<Health> = ComponentStorage::new();
        let e = alloc.allocate();
        storage.insert(e, Health(42));
        let removed = storage.remove(e);
        assert_eq!(removed, Some(Health(42)));
        assert!(storage.get(e).is_none());
    }

    #[test]
    fn remove_absent_returns_none() {
        let mut alloc = EntityAllocator::new();
        let mut storage: ComponentStorage<Health> = ComponentStorage::new();
        let e = alloc.allocate();
        assert!(storage.remove(e).is_none());
    }

    #[test]
    fn swap_remove_preserves_other_components() {
        let mut alloc = EntityAllocator::new();
        let mut storage: ComponentStorage<Health> = ComponentStorage::new();
        let e1 = alloc.allocate();
        let e2 = alloc.allocate();
        let e3 = alloc.allocate();
        storage.insert(e1, Health(1));
        storage.insert(e2, Health(2));
        storage.insert(e3, Health(3));
        storage.remove(e1);
        assert!(storage.get(e1).is_none());
        assert_eq!(storage.get(e2), Some(&Health(2)));
        assert_eq!(storage.get(e3), Some(&Health(3)));
        assert_eq!(storage.len(), 2);
    }

    #[test]
    fn iter_returns_all_components() {
        let mut alloc = EntityAllocator::new();
        let mut storage: ComponentStorage<Health> = ComponentStorage::new();
        let e1 = alloc.allocate();
        let e2 = alloc.allocate();
        storage.insert(e1, Health(10));
        storage.insert(e2, Health(20));
        let mut collected: Vec<_> = storage.iter().map(|(id, h)| (id, h.0)).collect();
        collected.sort_by_key(|(_, v)| *v);
        assert_eq!(collected, vec![(e1, 10), (e2, 20)]);
    }

    #[test]
    fn get_mut_modifies_in_place() {
        let mut alloc = EntityAllocator::new();
        let mut storage: ComponentStorage<Health> = ComponentStorage::new();
        let e = alloc.allocate();
        storage.insert(e, Health(5));
        storage.get_mut(e).unwrap().0 = 99;
        assert_eq!(storage.get(e), Some(&Health(99)));
    }

    // Property 2: Component Query Completeness
    // Validates: Requirements 1.3
    #[test]
    fn property_component_query_completeness() {
        let mut alloc = EntityAllocator::new();
        let mut storage: ComponentStorage<Health> = ComponentStorage::new();
        
        // Create multiple entities with components
        let entities: Vec<_> = (0..100).map(|_| alloc.allocate()).collect();
        let mut expected_values = Vec::new();
        
        // Insert components for most entities (but not all)
        for (i, &entity) in entities.iter().enumerate() {
            if i % 3 != 0 { // Skip every third entity
                let health = Health(i as i32 * 10);
                expected_values.push((entity, health.0));
                storage.insert(entity, health);
            }
        }
        
        // Query all components
        let mut queried: Vec<_> = storage.iter().map(|(id, h)| (id, h.0)).collect();
        queried.sort_by_key(|&(id, _)| id.index());
        expected_values.sort_by_key(|&(id, _)| id.index());
        
        // Verify query returns exactly the inserted components
        assert_eq!(queried.len(), expected_values.len());
        assert_eq!(queried, expected_values);
        
        // Verify no extra components are returned
        for &entity in &entities {
            let has_component = expected_values.iter().any(|(id, _)| *id == entity);
            let queried_has = queried.iter().any(|(id, _)| *id == entity);
            assert_eq!(queried_has, has_component);
        }
    }
}
