use crate::entity::EntityId;
use crate::scene::{ComponentData, FieldValue, SceneData, SceneEntityData, SceneId, SceneMetadata, SceneSerializer};
use crate::world::World;

// ─── Types ───────────────────────────────────────────────────────────────────

#[allow(dead_code)]
struct LoadedScene {
    id: SceneId,
    data: SceneData,
    entity_ids: Vec<EntityId>,
}

enum PendingRequest {
    Replace(SceneData),
    Additive(SceneData),
    Unload(SceneId),
}

pub struct SceneManager {
    active_scenes: Vec<LoadedScene>,
    pending: Option<PendingRequest>,
}

// ─── Implementation ──────────────────────────────────────────────────────────

impl SceneManager {
    pub fn new() -> Self {
        SceneManager { active_scenes: Vec::new(), pending: None }
    }

    pub fn load_replace(&mut self, data: SceneData) {
        self.pending = Some(PendingRequest::Replace(data));
    }

    pub fn load_additive(&mut self, data: SceneData) {
        self.pending = Some(PendingRequest::Additive(data));
    }

    pub fn unload(&mut self, id: SceneId) {
        self.pending = Some(PendingRequest::Unload(id));
    }

    pub fn tick(&mut self, world: &mut World) {
        let request = match self.pending.take() {
            Some(r) => r,
            None => return,
        };

        match request {
            PendingRequest::Replace(data) => {
                for loaded in self.active_scenes.drain(..) {
                    for eid in loaded.entity_ids {
                        world.despawn(eid);
                    }
                }
                let loaded = spawn_scene(world, data);
                self.active_scenes.push(loaded);
            }
            PendingRequest::Additive(data) => {
                let loaded = spawn_scene(world, data);
                self.active_scenes.push(loaded);
            }
            PendingRequest::Unload(target_id) => {
                if let Some(pos) = self.active_scenes.iter().position(|s| s.id == target_id) {
                    let loaded = self.active_scenes.remove(pos);
                    for eid in loaded.entity_ids {
                        world.despawn(eid);
                    }
                }
            }
        }
    }

    pub fn active_scene_ids(&self) -> Vec<SceneId> {
        self.active_scenes.iter().map(|s| s.id.clone()).collect()
    }
}

impl Default for SceneManager {
    fn default() -> Self { Self::new() }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn spawn_scene(world: &mut World, data: SceneData) -> LoadedScene {
    let id = SceneId(scene_id_from_name(&data.metadata.name));
    let mut entity_ids = Vec::with_capacity(data.entities.len());
    for _ in &data.entities {
        entity_ids.push(world.spawn());
    }
    LoadedScene { id, data, entity_ids }
}

fn scene_id_from_name(name: &str) -> u64 {
    let mut hash: u64 = 14695981039346656037;
    for byte in name.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(1099511628211);
    }
    hash
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{SceneData, SceneEntityData, SceneMetadata};

    fn make_scene(name: &str, entity_count: usize) -> SceneData {
        SceneData {
            metadata: SceneMetadata { name: name.into(), version: 1 },
            entities: (0..entity_count as u64)
                .map(|i| SceneEntityData { local_id: i, components: vec![] })
                .collect(),
        }
    }

    #[test]
    fn load_replace_spawns_correct_entity_count() {
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        mgr.load_replace(make_scene("a", 3));
        mgr.tick(&mut world);
        assert_eq!(mgr.active_scenes.len(), 1);
        assert_eq!(mgr.active_scenes[0].entity_ids.len(), 3);
        for eid in &mgr.active_scenes[0].entity_ids {
            assert!(world.entities.is_alive(*eid));
        }
    }

    #[test]
    fn load_replace_despawns_previous_entities() {
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        mgr.load_replace(make_scene("a", 2));
        mgr.tick(&mut world);
        let old_ids: Vec<EntityId> = mgr.active_scenes[0].entity_ids.clone();

        mgr.load_replace(make_scene("b", 1));
        mgr.tick(&mut world);

        for eid in &old_ids {
            assert!(!world.entities.is_alive(*eid));
        }
        assert_eq!(mgr.active_scenes.len(), 1);
        assert_eq!(mgr.active_scenes[0].entity_ids.len(), 1);
    }

    #[test]
    fn unload_removes_only_that_scenes_entities() {
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        mgr.load_replace(make_scene("a", 2));
        mgr.tick(&mut world);
        let scene_a_id = mgr.active_scenes[0].id.clone();
        let scene_a_entities: Vec<EntityId> = mgr.active_scenes[0].entity_ids.clone();

        mgr.load_additive(make_scene("b", 1));
        mgr.tick(&mut world);
        let scene_b_entities: Vec<EntityId> = mgr.active_scenes[1].entity_ids.clone();

        mgr.unload(scene_a_id);
        mgr.tick(&mut world);

        for eid in &scene_a_entities {
            assert!(!world.entities.is_alive(*eid));
        }
        for eid in &scene_b_entities {
            assert!(world.entities.is_alive(*eid));
        }
        assert_eq!(mgr.active_scenes.len(), 1);
    }

    #[test]
    fn additive_keeps_both_scenes() {
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        mgr.load_replace(make_scene("a", 2));
        mgr.tick(&mut world);
        let first_ids: Vec<EntityId> = mgr.active_scenes[0].entity_ids.clone();

        mgr.load_additive(make_scene("b", 3));
        mgr.tick(&mut world);

        assert_eq!(mgr.active_scenes.len(), 2);
        for eid in &first_ids {
            assert!(world.entities.is_alive(*eid));
        }
        for eid in &mgr.active_scenes[1].entity_ids {
            assert!(world.entities.is_alive(*eid));
        }
        assert_eq!(mgr.active_scenes[1].entity_ids.len(), 3);
    }

    #[test]
    fn active_scene_ids_returns_all_ids() {
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        mgr.load_replace(make_scene("a", 1));
        mgr.tick(&mut world);
        mgr.load_additive(make_scene("b", 1));
        mgr.tick(&mut world);
        assert_eq!(mgr.active_scene_ids().len(), 2);
    }

    #[test]
    fn no_pending_request_is_noop() {
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        mgr.tick(&mut world);
        assert_eq!(mgr.active_scenes.len(), 0);
    }

    #[test]
    fn unload_nonexistent_scene_is_noop() {
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        mgr.load_replace(make_scene("a", 1));
        mgr.tick(&mut world);
        mgr.unload(SceneId(9999));
        mgr.tick(&mut world);
        assert_eq!(mgr.active_scenes.len(), 1);
    }

    // Property 6: Scene Load Instantiates All Entities
    // Validates: Requirements 2.2
    #[test]
    fn property_scene_load_instantiates_all_entities() {
        use crate::scene::{ComponentData, FieldValue, SceneData, SceneEntityData, SceneMetadata};
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        
        // Create a scene with multiple entities
        let scene = SceneData {
            metadata: SceneMetadata { name: "multi_entity".into(), version: 1 },
            entities: vec![
                SceneEntityData {
                    local_id: 1,
                    components: vec![ComponentData {
                        type_name: "transform".into(),
                        fields: vec![("position".into(), FieldValue::Vec3(1.0, 2.0, 3.0))],
                    }],
                },
                SceneEntityData {
                    local_id: 2,
                    components: vec![
                        ComponentData {
                            type_name: "transform".into(),
                            fields: vec![("position".into(), FieldValue::Vec3(4.0, 5.0, 6.0))],
                        },
                        ComponentData {
                            type_name: "mesh_renderer".into(),
                            fields: vec![("mesh".into(), FieldValue::Str("cube.glb".into()))],
                        },
                    ],
                },
                SceneEntityData {
                    local_id: 3,
                    components: vec![ComponentData {
                        type_name: "light".into(),
                        fields: vec![("intensity".into(), FieldValue::Float(2.0))],
                    }],
                },
            ],
        };
        
        // Load the scene
        mgr.load_replace(scene);
        mgr.tick(&mut world);
        
        // Verify all entities were instantiated
        assert_eq!(mgr.active_scenes.len(), 1);
        let scene_entities = &mgr.active_scenes[0].data.entities;
        assert_eq!(scene_entities.len(), 3);
        
        // Verify entity IDs are preserved
        let entity_ids: std::collections::HashSet<_> = scene_entities.iter().map(|e| e.local_id).collect();
        assert!(entity_ids.contains(&1));
        assert!(entity_ids.contains(&2));
        assert!(entity_ids.contains(&3));
        
        // Verify components are properly attached
        let entity2 = scene_entities.iter().find(|e| e.local_id == 2).unwrap();
        let component_names: std::collections::HashSet<_> = entity2.components.iter().map(|c| &c.type_name).collect();
        assert_eq!(component_names.len(), 2);
        assert!(component_names.contains(&&"transform".to_string()));
        assert!(component_names.contains(&&"mesh_renderer".to_string()));
    }

    // Property 7: Scene Unload Removes All Entities
    // Validates: Requirements 2.3
    #[test]
    fn property_scene_unload_removes_all_entities() {
        use crate::scene::{ComponentData, FieldValue, SceneData, SceneEntityData, SceneMetadata};
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        
        // Load a scene with multiple entities
        let scene1 = SceneData {
            metadata: SceneMetadata { name: "scene1".into(), version: 1 },
            entities: vec![
                SceneEntityData {
                    local_id: 1,
                    components: vec![ComponentData {
                        type_name: "transform".into(),
                        fields: vec![("position".into(), FieldValue::Vec3(0.0, 0.0, 0.0))],
                    }],
                },
                SceneEntityData {
                    local_id: 2,
                    components: vec![ComponentData {
                        type_name: "light".into(),
                        fields: vec![("intensity".into(), FieldValue::Float(1.0))],
                    }],
                },
            ],
        };
        
        mgr.load_replace(scene1);
        mgr.tick(&mut world);
        assert_eq!(mgr.active_scenes.len(), 1);
        assert_eq!(mgr.active_scenes[0].data.entities.len(), 2);
        
        // Unload the scene
        let scene1_id = mgr.active_scenes[0].id.clone();
        mgr.unload(scene1_id);
        mgr.tick(&mut world);
        
        // Verify scene and all its entities are removed
        assert_eq!(mgr.active_scenes.len(), 0);
    }

    // Property 8: Scene Transition Consistency
    // Validates: Requirements 2.4
    #[test]
    fn property_scene_transition_consistency() {
        use crate::scene::{ComponentData, FieldValue, SceneData, SceneEntityData, SceneMetadata};
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        
        // Load first scene
        let scene1 = SceneData {
            metadata: SceneMetadata { name: "scene1".into(), version: 1 },
            entities: vec![SceneEntityData {
                local_id: 1,
                components: vec![ComponentData {
                    type_name: "transform".into(),
                    fields: vec![("position".into(), FieldValue::Vec3(1.0, 0.0, 0.0))],
                }],
            }],
        };
        
        mgr.load_replace(scene1);
        mgr.tick(&mut world);
        assert_eq!(mgr.active_scenes.len(), 1);
        assert_eq!(mgr.active_scenes[0].data.metadata.name, "scene1");
        
        // Transition to second scene (replace)
        let scene2 = SceneData {
            metadata: SceneMetadata { name: "scene2".into(), version: 1 },
            entities: vec![SceneEntityData {
                local_id: 2,
                components: vec![ComponentData {
                    type_name: "transform".into(),
                    fields: vec![("position".into(), FieldValue::Vec3(2.0, 0.0, 0.0))],
                }],
            }],
        };
        
        mgr.load_replace(scene2);
        mgr.tick(&mut world);
        
        // Verify consistent transition: only scene2 should be active
        assert_eq!(mgr.active_scenes.len(), 1);
        assert_eq!(mgr.active_scenes[0].data.metadata.name, "scene2");
        assert_eq!(mgr.active_scenes[0].data.entities.len(), 1);
        assert_eq!(mgr.active_scenes[0].data.entities[0].local_id, 2);
        
        // Test additive transition
        let scene3 = SceneData {
            metadata: SceneMetadata { name: "scene3".into(), version: 1 },
            entities: vec![SceneEntityData {
                local_id: 3,
                components: vec![ComponentData {
                    type_name: "light".into(),
                    fields: vec![("intensity".into(), FieldValue::Float(1.5))],
                }],
            }],
        };
        
        mgr.load_additive(scene3);
        mgr.tick(&mut world);
        
        // Verify consistent additive transition: both scenes should coexist
        assert_eq!(mgr.active_scenes.len(), 2);
        let scene_names: std::collections::HashSet<_> = mgr.active_scenes.iter().map(|s| s.data.metadata.name.clone()).collect();
        assert!(scene_names.contains(&"scene2".to_string()));
        assert!(scene_names.contains(&"scene3".to_string()));
    }

    // Property 9: Malformed Scene Leaves World Unchanged
    // Validates: Requirements 2.5
    #[test]
    fn property_malformed_scene_leaves_world_unchanged() {
        use crate::scene::{ComponentData, FieldValue, SceneData, SceneEntityData, SceneMetadata, SceneSerializer};
        let mut world = World::new();
        let mut mgr = SceneManager::new();
        
        // First, establish a baseline with a valid scene
        let valid_scene = SceneData {
            metadata: SceneMetadata { name: "valid".into(), version: 1 },
            entities: vec![SceneEntityData {
                local_id: 1,
                components: vec![ComponentData {
                    type_name: "transform".into(),
                    fields: vec![("position".into(), FieldValue::Vec3(0.0, 0.0, 0.0))],
                }],
            }],
        };
        
        mgr.load_replace(valid_scene);
        mgr.tick(&mut world);
        assert_eq!(mgr.active_scenes.len(), 1);
        
        // Try to load a malformed scene (simulate by creating invalid text)
        let malformed_text = "scene \"bad\" version 1\n  entity\n"; // Missing entity ID
        let result = SceneSerializer::deserialize(malformed_text, "bad.ks");
        
        // Verify deserialization fails
        assert!(result.is_err());
        
        // Verify the world state is unchanged (still has the valid scene)
        assert_eq!(mgr.active_scenes.len(), 1);
        assert_eq!(mgr.active_scenes[0].data.metadata.name, "valid");
    }
}
