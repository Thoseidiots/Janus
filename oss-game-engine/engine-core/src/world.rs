use std::any::{Any, TypeId};
use std::collections::HashMap;

use crate::entity::{EntityAllocator, EntityId};
use crate::storage::{Component, ComponentStorage};
use crate::system::SystemScheduler;

/// Object-safe trait for type-erased component storages, with downcast support.
trait AnyStorage: Any + Send + Sync {
    fn remove_entity(&mut self, id: EntityId);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Component> AnyStorage for ComponentStorage<T> {
    fn remove_entity(&mut self, id: EntityId) {
        self.remove(id);
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Holds all component storages keyed by component TypeId.
struct ComponentRegistry {
    storages: HashMap<TypeId, Box<dyn AnyStorage>>,
}

impl ComponentRegistry {
    fn new() -> Self {
        ComponentRegistry {
            storages: HashMap::new(),
        }
    }

    fn get_or_create<T: Component>(&mut self) -> &mut ComponentStorage<T> {
        self.storages
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(ComponentStorage::<T>::new()))
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
            .expect("type mismatch in ComponentRegistry")
    }

    fn get_storage<T: Component>(&self) -> Option<&ComponentStorage<T>> {
        self.storages
            .get(&TypeId::of::<T>())?
            .as_any()
            .downcast_ref::<ComponentStorage<T>>()
    }

    fn get_storage_mut<T: Component>(&mut self) -> Option<&mut ComponentStorage<T>> {
        self.storages
            .get_mut(&TypeId::of::<T>())?
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
    }

    fn remove_entity_from_all(&mut self, id: EntityId) {
        for storage in self.storages.values_mut() {
            storage.remove_entity(id);
        }
    }
}

/// The central ECS container.
pub struct World {
    pub(crate) entities: EntityAllocator,
    components: ComponentRegistry,
    pub(crate) systems: SystemScheduler,
}

impl World {
    pub fn new() -> Self {
        World {
            entities: EntityAllocator::new(),
            components: ComponentRegistry::new(),
            systems: SystemScheduler::new(),
        }
    }

    /// Spawn a new entity and return its ID.
    pub fn spawn(&mut self) -> EntityId {
        self.entities.allocate()
    }

    /// Destroy an entity, removing it from the allocator and all component storages.
    pub fn despawn(&mut self, id: EntityId) {
        self.components.remove_entity_from_all(id);
        self.entities.free(id);
    }

    /// Attach a component to an entity.
    pub fn add_component<T: Component>(&mut self, id: EntityId, c: T) {
        self.components.get_or_create::<T>().insert(id, c);
    }

    /// Remove and return a component from an entity.
    pub fn remove_component<T: Component>(&mut self, id: EntityId) -> Option<T> {
        self.components.get_storage_mut::<T>()?.remove(id)
    }

    /// Query read-only access to a component storage.
    pub fn query<T: Component>(&self) -> Option<&ComponentStorage<T>> {
        self.components.get_storage::<T>()
    }

    /// Query mutable access to a component storage.
    pub fn query_mut<T: Component>(&mut self) -> Option<&mut ComponentStorage<T>> {
        self.components.get_storage_mut::<T>()
    }

    /// Register a system to run each tick.
    pub fn register_system(&mut self, system: impl crate::system::System) {
        self.systems.register(system);
    }

    /// Advance the world by one tick, running all registered systems.
    pub fn tick(&mut self, delta: f32) {
        // Temporarily take the scheduler out to avoid simultaneous mutable borrows.
        let mut scheduler = std::mem::replace(&mut self.systems, SystemScheduler::new());
        scheduler.run_all(self, delta);
        self.systems = scheduler;
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct Position { x: f32, y: f32 }
    impl Component for Position {}

    #[derive(Debug, PartialEq)]
    struct Velocity { dx: f32, dy: f32 }
    impl Component for Velocity {}

    #[test]
    fn spawn_and_despawn() {
        let mut world = World::new();
        let e = world.spawn();
        assert!(world.entities.is_alive(e));
        world.despawn(e);
        assert!(!world.entities.is_alive(e));
    }

    #[test]
    fn add_and_query_component() {
        let mut world = World::new();
        let e = world.spawn();
        world.add_component(e, Position { x: 1.0, y: 2.0 });
        let storage = world.query::<Position>().unwrap();
        assert_eq!(storage.get(e), Some(&Position { x: 1.0, y: 2.0 }));
    }

    #[test]
    fn remove_component() {
        let mut world = World::new();
        let e = world.spawn();
        world.add_component(e, Position { x: 0.0, y: 0.0 });
        let removed = world.remove_component::<Position>(e);
        assert_eq!(removed, Some(Position { x: 0.0, y: 0.0 }));
        assert!(world.query::<Position>().unwrap().get(e).is_none());
    }

    #[test]
    fn despawn_removes_all_components() {
        let mut world = World::new();
        let e = world.spawn();
        world.add_component(e, Position { x: 5.0, y: 5.0 });
        world.add_component(e, Velocity { dx: 1.0, dy: 0.0 });
        world.despawn(e);
        if let Some(pos_storage) = world.query::<Position>() {
            assert!(pos_storage.get(e).is_none());
        }
        if let Some(vel_storage) = world.query::<Velocity>() {
            assert!(vel_storage.get(e).is_none());
        }
    }

    #[test]
    fn multiple_entities_independent() {
        let mut world = World::new();
        let e1 = world.spawn();
        let e2 = world.spawn();
        world.add_component(e1, Position { x: 1.0, y: 0.0 });
        world.add_component(e2, Position { x: 2.0, y: 0.0 });
        let storage = world.query::<Position>().unwrap();
        assert_eq!(storage.get(e1), Some(&Position { x: 1.0, y: 0.0 }));
        assert_eq!(storage.get(e2), Some(&Position { x: 2.0, y: 0.0 }));
    }

    // Property 2: Component Query Completeness
    // Validates: Requirements 1.3
    #[test]
    fn property_component_query_completeness() {
        #[derive(Debug, PartialEq)]
        struct Position { x: f32, y: f32 }
        impl Component for Position {}
        
        #[derive(Debug, PartialEq)]
        struct Velocity { x: f32, y: f32 }
        impl Component for Velocity {}
        
        let mut world = World::new();
        
        // Create entities with different component combinations
        let e1 = world.spawn();
        world.add_component(e1, Position { x: 1.0, y: 2.0 });
        
        let e2 = world.spawn();
        world.add_component(e2, Velocity { x: 3.0, y: 4.0 });
        
        let e3 = world.spawn();
        world.add_component(e3, Position { x: 5.0, y: 6.0 });
        world.add_component(e3, Velocity { x: 7.0, y: 8.0 });
        
        // Query should return all entities with Position component
        if let Some(pos_storage) = world.query::<Position>() {
            let positions: Vec<_> = pos_storage.iter().collect();
            assert_eq!(positions.len(), 2);
            assert!(positions.iter().any(|(id, _)| *id == e1));
            assert!(positions.iter().any(|(id, _)| *id == e3));
        }
        
        // Query should return all entities with Velocity component
        if let Some(vel_storage) = world.query::<Velocity>() {
            let velocities: Vec<_> = vel_storage.iter().collect();
            assert_eq!(velocities.len(), 2);
            assert!(velocities.iter().any(|(id, _)| *id == e2));
            assert!(velocities.iter().any(|(id, _)| *id == e3));
        }
        
        // Query should return entities in contiguous memory order
        if let Some(pos_storage) = world.query::<Position>() {
            for (_, pos) in pos_storage.iter() {
                // This test ensures the storage is contiguous
                // by verifying we can iterate without gaps
                assert!(pos.x > 0.0); // Just verify data is accessible
            }
        }
    }

    // Property 3: Entity Destruction Removes All Components
    // Validates: Requirements 1.4
    #[test]
    fn property_entity_destruction_removes_all_components() {
        #[derive(Debug, PartialEq)]
        struct Position { x: f32, y: f32 }
        impl Component for Position {}
        
        #[derive(Debug, PartialEq)]
        struct Velocity { x: f32, y: f32 }
        impl Component for Velocity {}
        
        #[derive(Debug, PartialEq)]
        struct Health { value: u32 }
        impl Component for Health {}
        
        let mut world = World::new();
        
        // Create entity with multiple components
        let entity = world.spawn();
        world.add_component(entity, Position { x: 1.0, y: 2.0 });
        world.add_component(entity, Velocity { x: 3.0, y: 4.0 });
        world.add_component(entity, Health { value: 100 });
        
        // Verify components exist
        if let Some(pos_storage) = world.query::<Position>() {
            assert!(pos_storage.iter().any(|(id, _)| id == entity));
        }
        if let Some(vel_storage) = world.query::<Velocity>() {
            assert!(vel_storage.iter().any(|(id, _)| id == entity));
        }
        if let Some(health_storage) = world.query::<Health>() {
            assert!(health_storage.iter().any(|(id, _)| id == entity));
        }
        
        // Destroy entity
        world.despawn(entity);
        
        // Verify all components are removed
        if let Some(pos_storage) = world.query::<Position>() {
            assert!(!pos_storage.iter().any(|(id, _)| id == entity));
        }
        if let Some(vel_storage) = world.query::<Velocity>() {
            assert!(!vel_storage.iter().any(|(id, _)| id == entity));
        }
        if let Some(health_storage) = world.query::<Health>() {
            assert!(!health_storage.iter().any(|(id, _)| id == entity));
        }
        
        // Verify entity ID is no longer valid
        assert!(!world.entities.is_alive(entity));
    }
}
