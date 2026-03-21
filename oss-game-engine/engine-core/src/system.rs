use crate::world::World;

/// A system that operates on the World each frame.
pub trait System: 'static + Send + Sync {
    fn run(&mut self, world: &mut World, delta: f32);
}

/// Holds and executes registered systems in registration order.
pub struct SystemScheduler {
    systems: Vec<Box<dyn System>>,
}

impl SystemScheduler {
    pub fn new() -> Self {
        SystemScheduler {
            systems: Vec::new(),
        }
    }

    /// Register a system; it will run after all previously registered systems.
    pub fn register(&mut self, system: impl System) {
        self.systems.push(Box::new(system));
    }

    /// Execute all systems in registration order (serial).
    pub fn run_all(&mut self, world: &mut World, delta: f32) {
        for system in &mut self.systems {
            system.run(world, delta);
        }
    }
}

impl Default for SystemScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct RecordingSystem {
        log: Arc<Mutex<Vec<usize>>>,
        id: usize,
    }

    impl System for RecordingSystem {
        fn run(&mut self, _world: &mut World, _delta: f32) {
            self.log.lock().unwrap().push(self.id);
        }
    }

    #[test]
    fn systems_run_in_registration_order() {
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = SystemScheduler::new();
        for i in 0..5 {
            scheduler.register(RecordingSystem { log: Arc::clone(&log), id: i });
        }
        let mut world = World::new();
        scheduler.run_all(&mut world, 0.016);
        assert_eq!(*log.lock().unwrap(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn empty_scheduler_runs_without_panic() {
        let mut scheduler = SystemScheduler::new();
        let mut world = World::new();
        scheduler.run_all(&mut world, 0.016); // should not panic
    }

    // Property 1: System Execution Order
    // Validates: Requirements 1.2
    #[test]
    fn property_system_execution_order() {
        use std::sync::{Arc, Mutex};
        
        struct OrderSystem {
            log: Arc<Mutex<Vec<usize>>>,
            id: usize,
        }
        
        impl System for OrderSystem {
            fn run(&mut self, _world: &mut World, _delta: f32) {
                self.log.lock().unwrap().push(self.id);
            }
        }
        
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = SystemScheduler::new();
        
        // Register systems in specific order
        for i in 0..10 {
            scheduler.register(OrderSystem { 
                log: Arc::clone(&log), 
                id: i 
            });
        }
        
        let mut world = World::new();
        scheduler.run_all(&mut world, 0.016);
        
        let execution_order = log.lock().unwrap().clone();
        assert_eq!(execution_order, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        
        // Run multiple times to ensure consistent order
        scheduler.run_all(&mut world, 0.016);
        let second_run = log.lock().unwrap().clone();
        assert_eq!(second_run, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    // Property 4: Concurrent Write Serialization
    // Validates: Requirements 1.5
    #[test]
    fn property_concurrent_write_serialization() {
        use crate::storage::Component;
        use std::sync::{Arc, Mutex};
        
        #[derive(Debug)]
        struct Counter(usize);
        impl Component for Counter {}
        
        let counter = Arc::new(Mutex::new(0));
        let mut scheduler = SystemScheduler::new();

        // Create two systems that both write to Counter component
        struct CounterSystem {
            counter: Arc<Mutex<usize>>,
        }
        
        impl System for CounterSystem {
            fn run(&mut self, world: &mut World, _delta: f32) {
                // Collect entities and their current counter values first
                let mut updates = Vec::new();
                if let Some(storage) = world.query_mut::<Counter>() {
                    for (entity, counter_comp) in storage.iter() {
                        updates.push((entity, counter_comp.0 + 1));
                    }
                }
                
                // Then apply updates
                for (entity, new_value) in updates {
                    world.add_component(entity, Counter(new_value));
                    *self.counter.lock().unwrap() += 1;
                }
            }
        }

        scheduler.register(CounterSystem { counter: Arc::clone(&counter) });
        scheduler.register(CounterSystem { counter: Arc::clone(&counter) });

        let mut world = World::new();
        let entity = world.spawn();
        world.add_component(entity, Counter(0));
        
        // Run systems twice to verify consistent serialization
        scheduler.run_all(&mut world, 0.016);
        scheduler.run_all(&mut world, 0.016);
        
        // Check that the counter has been incremented consistently
        if let Some(storage) = world.query::<Counter>() {
            let final_value = storage.iter().next().unwrap().1;
            assert_eq!(final_value.0, 4); // Each of 2 systems runs twice: 2 * 2 = 4 increments
        }
        
        // Verify atomicity test: run count should equal component value
        let run_count = *counter.lock().unwrap();
        assert_eq!(run_count, 4);
    }
}
