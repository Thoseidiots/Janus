# Janus

**Autonomous AI system for AAA game asset generation, self-directed learning, and economic autonomy.**

Built by Ishmael Sears. GitHub: [github.com/Thoseidiots/Janus](https://github.com/Thoseidiots/Janus)

-----

## What Janus Is

Janus is an AI system that runs on local hardware, generates game assets, manages commissions, learns from videos and the internet, and operates autonomously toward financial goals. It has its own trained model (Avus), a complete game engine written in Rust, a humanization layer that makes it sound human, and a visual avatar named Arania.

The system has three major layers:

- **Avus** — a 142M parameter transformer trained to generate 3D game asset parameters
- **Python brain** — orchestrator, CEO agent, screen vision, memory, speech, payment handling
- **OSS Game Engine** — a from-scratch Rust engine (ECS, physics, audio, scripting, renderer)

-----

## Quick Start

```bash
git clone https://github.com/Thoseidiots/Janus.git
cd Janus
python pull_weights.py --setup    # one-time Kaggle credential setup
python pull_weights.py            # download Avus weights (584MB)
pip install torch tiktoken numpy imageio
python janus_system_orchestrator.py
```

To verify the game engine:

```bash
cd oss-game-engine
cargo test --workspace            # should show 0 failures across all 9 crates
```

To run the debugger on the whole repo:

```bash
python tools/universal_oxpecker/cli.py scan .
```

-----

## Hardware

- **Machine:** HP EliteDesk 705 G4 SFF
- **CPU:** AMD Ryzen 5 Pro 2400G
- **RAM:** 16GB DDR4
- **Storage:** 256GB NVMe
- **OS:** Windows 11 Pro
- **Training:** Kaggle dual T4 GPU (remote)
- **Note:** No local GPU needed — Avus runs on CPU for inference

-----

## File Map

### Entry Points

|File                          |Purpose                                                                 |
|------------------------------|------------------------------------------------------------------------|
|`janus_system_orchestrator.py`|**Main entry point.** Runs the full OBSERVE→PLAN→ACT→EXECUTE→REVIEW loop|
|`janus_main.py`               |Alternate entry point for simpler runs                                  |
|`janus_ceo_main.py`           |CEO agent entry point for strategic goal execution                      |
|`pull_weights.py`             |Downloads Avus weights from Kaggle. Run this first after cloning        |
|`rustup-init.exe`             |Rust installer for Windows (EliteDesk setup)                            |

-----

### Avus — The AI Brain

|File                                    |Purpose                                                                                                                        |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|`avus_inference.py`                     |Loads Avus weights, exposes `generate()`, `generate_3d_params()`, `generate_action()`, `generate_lighting()`, `generate_pose()`|
|`avus.py`                               |Avus model architecture definition                                                                                             |
|`avus_tokenizer .py`                    |Tokenizer (note: filename has a space — known issue)                                                                           |
|`config_avus_1b .json`                  |Avus config: vocab=50304, dim=768, 12 layers, 12 heads (note: filename has a space)                                            |
|`train_avus_kaggle.py`                  |Kaggle training script — runs on dual T4, loads from dataset, saves after every epoch                                          |
|`train_avus_lightning.py`               |Lightning AI training version                                                                                                  |
|`avus_cognitive_dataset_generator_v3.py`|Generates cognitive training data for Avus                                                                                     |
|`avus_phase2_spec.md`                   |Phase 2 training specification                                                                                                 |

**Phase 2 dataset generators (AAA rendering quality):**

|File                             |Fixes                                       |
|---------------------------------|--------------------------------------------|
|`temporal_consistency_dataset.py`|Squibbling — frame-to-frame instability     |
|`spatial_detail_dataset.py`      |Yosification — over-smoothing of fine detail|
|`geometric_constraint_dataset.py`|Uncanny Valley — face geometry drift        |
|`optical_flow_dataset.py`        |Ghosting — disocclusion background fill     |
|`semantic_lighting_dataset.py`   |Imaginary Lighting — wrong scene lighting   |

-----

### Orchestration & Autonomy

|File                          |Purpose                                              |
|------------------------------|-----------------------------------------------------|
|`janus_system_orchestrator.py`|Master coordinator — runs the full autonomous loop   |
|`ceo_agent.py`                |Strategic goals and high-level decision making       |
|`agent_loop.py`               |Core agent execution loop                            |
|`goal_planner.py`             |Derives goals from valence state, proposes actions   |
|`goal_generator.py`           |Generates new goals based on context                 |
|`replanner.py`                |Replans when goals fail or context changes           |
|`tree_planner.py`             |Tree-based planning for multi-step tasks             |
|`task_execution_engine.py`    |Executes planned tasks                               |
|`autonomous_task_selector.py` |Selects next task autonomously                       |
|`actions_and_feedback.py`     |Action definitions and feedback processing           |
|`skill_executor.py`           |Executes skills via JanusOS desktop control (Windows)|
|`os_human_interface.py`       |JanusOS — Windows desktop control via ctypes         |
|`autonomous_ide.py`           |Autonomous IDE interaction and code editing          |
|`autonomy_capability.py`      |Core autonomy capability definitions                 |
|`bootstrap_autonomy.py`       |Bootstraps the autonomy system on first run          |

-----

### Memory

|File                         |Purpose                                                                                                      |
|-----------------------------|-------------------------------------------------------------------------------------------------------------|
|`janus_memory_integration.py`|**Main memory interface.** `JanusMemory` class — auto-uses holographic mode with torch, dict fallback without|
|`holographic_memory.py`      |`InfiniteJanusMemory` — fixed ~16KB footprint, unlimited logical capacity via HRR/FFT                        |
|`structured_memory.py`       |Structured memory layer                                                                                      |
|`long_term_memory.py`        |Long-term memory persistence                                                                                 |
|`memory_manager.py`          |Memory management utilities                                                                                  |
|`learning.py`                |Learning and knowledge update system                                                                         |

**Holographic Brain Memory library** (`holographic_brain_memory/` — if present):

|File              |Purpose                                                         |
|------------------|----------------------------------------------------------------|
|`core.py`         |`HolographicBrainMemory`, `PhaseBrainLayer` — complex-valued HRR|
|`real_valued.py`  |`RealHolographicMemory` — hardware-friendly alternative         |
|`spawning.py`     |`SpawningBrain` — dynamic logical neuron growth                 |
|`visualization.py`|Memory trace visualization                                      |

-----

### Speech & Humanization

|File                         |Purpose                                                                                                 |
|-----------------------------|--------------------------------------------------------------------------------------------------------|
|`janus_humanization_layer.py`|**Full humanization stack.** Fillers, SSML, breath model, imperfections, discourse, late-binding pivot  |
|`janus_speech_arbiter.rs`    |Rust real-time speech arbiter — interruption classification, backchannel, turn-taking, echo cancellation|
|`demo_humanized_janus.py`    |Demo showing humanization in action                                                                     |
|`voice_io_enhanced.py`       |Voice input/output handling                                                                             |
|`emotional_intelligence.py`  |Emotional intelligence layer                                                                            |

**Key humanization components in `janus_humanization_layer.py`:**

- `NaturalSpeechGenerator` — fillers by arousal level
- `EmotionalVoiceGenerator` — SSML with valence-driven prosody, near-field EQ
- `RespiratoryModel` — breath cycles, trailing-off, pre-speech inhale
- `ImperfectionEngine` — restarts, jitter/shimmer
- `ReflectionEngine` — “I’m curious about…”
- `DiscourseEngine` — pragmatic markers, uses “Ish” nickname casually
- `LateBindingPivot` — graceful interruption reconstruction

-----

### Vision & Screen Perception

|File                         |Purpose                                      |
|-----------------------------|---------------------------------------------|
|`screen_interpreter.py`      |Raw pixels → text description for Avus       |
|`enhanced_vision.py`         |Enhanced vision processing                   |
|`updated_enhanced_vision.py` |Temporal video analysis methods              |
|`video_observer.py`          |Watches videos autonomously for self-learning|
|`video_capability.py`        |Registers video tools into capability hub    |
|`video_learner.py`           |Processes video content into memory          |
|`bootstrap_video_observer.py`|Enables autonomous video watching            |
|`local_vision.py`            |Local vision without cloud APIs              |
|`vision_perception.py`       |Core perception pipeline                     |
|`vision_automation.py`       |Automates vision-based tasks                 |
|`unified_perception.py`      |Unifies all perception sources               |
|`multimodal_fusion.py`       |Fuses text, vision, and audio inputs         |

-----

### Game Generation Pipeline

|File                           |Purpose                                                                             |
|-------------------------------|------------------------------------------------------------------------------------|
|`game_generation_pipeline.py`  |Text → Avus params → geometry + PBR textures → `GameAsset`. Exports `.obj` and `.ks`|
|`kiro_scene_exporter.py`       |`GameAsset` → KiroScene `.ks` format for direct engine loading                      |
|`advanced_3d_face_generator.py`|AAA-quality 3D face generation with blend shapes                                    |
|`game_dev_example.py`          |Example of the full generation pipeline                                             |

**Game AI Database** (`game_ai_database/`):

|File                                  |Purpose                        |
|--------------------------------------|-------------------------------|
|`generators/character_generator.py`   |Procedural character generation|
|`generators/level_design_generator.py`|Level design generation        |
|`generators/narrative_generator.py`   |Narrative and story generation |
|`generators/quest_generator.py`       |Quest generation               |
|`generators/world_builder.py`         |World building                 |
|`generators/dialogue_generator.py`    |Dialogue generation            |
|`generators/asset_descriptor.py`      |Asset description generation   |
|`generators/audio_descriptor.py`      |Audio asset descriptions       |
|`generators/economy_generator.py`     |In-game economy generation     |
|`generators/mechanics_generator.py`   |Game mechanics generation      |
|`database/db_manager.py`              |Database management            |

-----

### Revenue & Payments

|File                   |Purpose                                                                                                                                |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------|
|`payment_pipeline.py`  |**Payment system.** Generates Revolut/PayPal links, tracks orders, maintains revenue ledger. Revolut: `@i_sears`, PayPal: Ishmael Sears|
|`revenue_execution.py` |Asset pack generation (dungeon/forest/castle/sci_fi/ruins), commission fulfillment, revenue tracking                                   |
|`autonomous_finance.py`|Autonomous financial planning and execution                                                                                            |

-----

### Arania — Visual Avatar

|File        |Purpose                                                                                                                                                                                                |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|`Arania.mat`|PBR material definition — 7 passes: skin (warm tan, SSS 0.28), hair (dark, anisotropic 0.6), robe (gold-yellow), gold (metallic 0.95), eye_iris (amber, emissive 0.9), eye_white, lip. Three-light rig.|

Arania is the visual avatar of Janus — a character that walks across the screen and performs tasks as a visual representation of Janus operating. The 3D body mesh is pending; materials and shaders are complete.

-----

### OSS Game Engine (`oss-game-engine/`)

A complete game engine written from scratch in Rust. Zero external crate dependencies. All 9 crates pass `cargo test --workspace`.

|Crate             |Contents                                                                                                           |
|------------------|-------------------------------------------------------------------------------------------------------------------|
|`engine-core`     |ECS World, Entity, ComponentStorage (sparse-set), SystemScheduler, Scene serializer, SceneManager                  |
|`engine-renderer` |GfxBackend trait, PBR Cook-Torrance shaders (GLSL complete, WGSL/HLSL/MSL stubs), Vulkan/Metal/DX12/WebGPU backends|
|`engine-physics`  |RigidBody, BVH broadphase, GJK+EPA narrowphase, fixed 60Hz timestep, triggers, raycasting                          |
|`engine-audio`    |WAV decoder, Mixer (64 sources), 3D spatialization, inverse-square attenuation                                     |
|`engine-scripting`|Loom language — lexer, parser, AST, type checker, bytecode compiler, VM, ECS bridge                                |
|`engine-assets`   |Asset pipeline, Blake3 cache, FileWatcher, hot-reload, KiroMeta format                                             |
|`engine-editor`   |Immediate-mode UI, HierarchyPanel, InspectorPanel, UndoStack, AutoSave                                             |
|`engine-build`    |Incremental build, platform targets (Windows/macOS/Linux/Web/iOS/Android)                                          |
|`engine-runtime`  |Main loop, plugin system, network guard, full integration of all subsystems                                        |

**KiroScene format** — assets export as `.ks` files readable directly by the engine:

```
scene "asset_name" version 1
  entity 1
    transform position 0.0 0.0 0.0 rotation 0.0 0.0 0.0 1.0 scale 1.0 1.0 1.0
    mesh_renderer mesh "assets/name.glb" material "assets/name.mat"
```

-----

### Distributed Infrastructure (`crates/`)

|Crate/File                           |Purpose                                                                      |
|-------------------------------------|-----------------------------------------------------------------------------|
|`crates/nexus-core/`                 |Multi-node Nexus system — gRPC communication between Janus nodes             |
|`crates/nexus-core/src/raft_app.rs`  |Raft consensus algorithm for distributed state agreement                     |
|`crates/nexus-core/src/dispatcher.rs`|WASM task dispatcher — spawns, ticks, snapshots, migrates tasks between nodes|
|`crates/nexus-core/src/jumf.rs`      |PCIe NTB memory mapping between motherboards (requires NTB hardware)         |
|`crates/nexus-core/src/jce.rs`       |Janus Compute Engine                                                         |
|`crates/nexus-core/src/las.rs`       |Local autonomy server                                                        |
|`crates/nexus-core/proto/nexus.proto`|gRPC protocol definition                                                     |
|`crates/janus-core/`                 |Core Rust library (PyO3 bridge to Python brain)                              |
|`crates/janus-llm/`                  |LLM integration crate                                                        |
|`crates/janus-cli/`                  |CLI entry point                                                              |
|`crates/janus-wasm/`                 |WASM compilation target                                                      |
|`nexus_client.py`                    |Python gRPC client for Nexus node communication                              |
|`nexus_pb2.py` / `nexus_pb2_grpc.py` |Generated protobuf bindings                                                  |

-----

### Universal Oxpecker (`tools/universal_oxpecker/`)

Multi-language static analyzer, debugger, and auto-repair engine. Supports 13 language families.

```bash
python tools/universal_oxpecker/cli.py scan .          # scan entire repo
python tools/universal_oxpecker/cli.py scan model.py   # single file
python tools/universal_oxpecker/cli.py repair model.py # auto-repair with rollback
```

|Directory                    |Contents                                                                                                      |
|-----------------------------|--------------------------------------------------------------------------------------------------------------|
|`adapters/`                  |Language adapters: Python, JS/TS, Java/Kotlin, C/C++, Rust, Go, C#, Ruby, PHP, Lua, Swift, Zig, functional/DSL|
|`core/engine.py`             |Central orchestrator — detects language, routes to adapter                                                    |
|`core/scanner.py`            |Project-wide multi-language scanner                                                                           |
|`core/orchestrator.py`       |Full repair workflow orchestrator                                                                             |
|`analysis/fix_suggester.py`  |Suggests fixes by complexity tier                                                                             |
|`analysis/repair_workflow.py`|Automated Program Repair with rollback history                                                                |
|`analysis/repair_plugins/`   |Language-specific repair plugins (Python, JS/TS)                                                              |
|`tests/`                     |Test suite with fixture files                                                                                 |

-----

### Janus Brain Package (`janus-brain/src/janus_brain/`)

|File             |Purpose                                                                                                         |
|-----------------|----------------------------------------------------------------------------------------------------------------|
|`core.py`        |`AutonomousCore` — main cognitive core with homeostasis, valence, memory                                        |
|`homeostasis.py` |`ValenceVector` — 6-dimensional emotional state (pleasure, arousal, curiosity, autonomy, connection, competence)|
|`llm.py`         |LLM interface and streaming                                                                                     |
|`memory.py`      |Core memory implementation                                                                                      |
|`bridge.py`      |Python↔Rust bridge                                                                                              |
|`nexus_client.py`|Nexus node client                                                                                               |

-----

### Training Data (`training_data/`)

Reference documents used for Avus training:

|File                                          |Topic                        |
|----------------------------------------------|-----------------------------|
|`aaa_game_design_&_systems_architecture.md`   |AAA game design patterns     |
|`advanced_shader_&_rendering_techniques.md`   |PBR, ray tracing, DLSS       |
|`ai_&_pathfinding_architecture.md`            |Game AI and pathfinding      |
|`asset_pipeline_&_tooling.md`                 |Asset pipeline best practices|
|`audio_engineering_&_spatial_sound.md`        |3D audio engineering         |
|`c++_performance_&_memory_management.md`      |Performance optimization     |
|`memory_layouts_&_cache_optimization_(dod).md`|Data-oriented design         |
|`multiplayer_networking_&_replication.md`     |Multiplayer architecture     |
|`physics_&_collision_systems.md`              |Physics engine design        |

-----

### Configuration Files

|File                   |Purpose                                                                            |
|-----------------------|-----------------------------------------------------------------------------------|
|`config_avus_1b .json` |Avus 1B config — vocab=50304, dim=768, 12 layers, 12 heads, 4 KV heads, max_seq=512|
|`config_100m.json`     |100M parameter config                                                              |
|`config_65m.json`      |65M parameter config                                                               |
|`config_33m.json`      |33M parameter config                                                               |
|`identity_object.json` |Janus identity and personality definition                                          |
|`persistent_state.json`|Persisted runtime state across sessions                                            |
|`schemas/`             |JSON schemas for observe/plan/propose/reflect cycle                                |

-----

### Web UI (`components/`, `App.tsx`, `index.tsx`)

React/TypeScript frontend. Not the primary interface — Janus operates autonomously without it. Includes components for chat, voice assistant, image/video tools, analysis, and a unified studio view.

-----

### Legacy / Archive

|Location                                        |Contents                                                          |
|------------------------------------------------|------------------------------------------------------------------|
|`archive/`                                      |Older versions of core files superseded by current implementations|
|`my-llm-project/`                               |Early curriculum training experiments                             |
|`*.txt` files                                   |Debug logs from Kiro sessions — safe to delete                    |
|`workspace_errors.txt`, `workspace_errors_2.txt`|Kiro debug output — not needed                                    |
|`runtime_error.txt`, `test_output.txt`          |Same — delete these                                               |

-----

## The Full Pipeline

```
Text prompt
    ↓  avus_inference.py
3D parameters (JSON)
    ↓  game_generation_pipeline.py
Geometry + PBR textures
    ↓  kiro_scene_exporter.py
.ks KiroScene file
    ↓  engine-assets KiroScene loader
Live entity in the OSS Game Engine
```

-----

## Weights

Avus weights are NOT stored in Git. They live on Kaggle.

- **Dataset:** [kaggle.com/datasets/ishmaelsears/janus-avus-weights](https://www.kaggle.com/datasets/ishmaelsears/janus-avus-weights)
- **File:** `avus_1b_weights.pt` (584MB)
- **License:** CC0 Public Domain

Download with:

```bash
python pull_weights.py --setup   # saves Kaggle API key to ~/.kaggle/
python pull_weights.py           # downloads weights
```

Never commit `avus_1b_weights.pt` or `kaggle.json` to git.

-----

## Repository Rules

- **Never commit:** `avus_1b_weights.pt`, `kaggle.json`, `*.pyc`, `__pycache__/`, `oss-game-engine/target/`, `demo_frames/`, `training_metrics.png`, `*.txt` debug logs
- **Never force push.** The repo has been force-pushed twice by external AI tools and required full recovery both times.
- **Last known clean commit:** `cb1efdb3c837968d8075d87773eb7ac05c0a874b`

-----

## Avus Training Status

|Phase  |Status        |Loss                       |
|-------|--------------|---------------------------|
|Phase 1|✅ Complete    |0.2253 (20 epochs, dual T4)|
|Phase 2|🔄 Ready to run|5 dataset generators built |

-----

## Payment

Janus can receive payments and track commissions via `payment_pipeline.py`:

- **Revolut:** `@i_sears`
- **PayPal:** Ishmael Sears

-----

*Built session by session from February 2026 onward.*