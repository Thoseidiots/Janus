# Critical Files Inventory

## Root-level Python Files
- **janus_*.py** (36 files):
  - janus_agent.py
  - janus_anti_halting.py
  - janus_binary_decider.py
  - janus_captcha_solver.py
  - janus_causal_horizon.py
  - janus_ceo_main.py
  - janus_email_composer.py
  - janus_enhanced_integration.py
  - janus_entanglement.py
  - janus_file_intelligence.py
  - janus_ghost_code_detector.py
  - janus_gpt.py
  - janus_humanization_layer.py
  - janus_local_llm.py
  - janus_loop_detector.py
  - janus_main.py
  - janus_memory.py
  - janus_memory_integration.py
  - janus_monitoring_dashboard.py
  - janus_notify_bridge.py
  - janus_optimizer.py
  - janus_relay_client.py
  - janus_reservoir.py
  - janus_self_improvement.py
  - janus_service_gateway.py
  - janus_synthetic_sql.py
  - janus_system_orchestrator.py
  - janus_turing_bypass.py
  - janus_unified.py
  - janus_video_comprehension.py
  - janus_voip.py
  - janus_vram_test.py
  - janus_web_reader.py
  - janus_worker.py
  - Plus archive files: janus_capability_hub.py, janus_core.py

- **avus*.py** (7 files):
  - avus.py
  - avus_brain.py
  - avus_brain_enhanced.py
  - avus_cognitive_dataset_generator_v3.py
  - avus_eval.py
  - avus_inference.py
  - avus_tokenizer.py

- **train.py** (2 files):
  - train.py (root level)
  - my-llm-project/scripts/train.py

## Rust Workspace
- **Cargo.toml** (16 files):
  - Cargo.toml (root workspace)
  - crates/janus-cli/Cargo.toml
  - crates/janus-core/Cargo.toml
  - crates/janus-llm/Cargo.toml
  - crates/janus-wasm/Cargo.toml
  - crates/nexus-core/Cargo.toml
  - oss-game-engine/Cargo.toml
  - oss-game-engine/engine-assets/Cargo.toml
  - oss-game-engine/engine-audio/Cargo.toml
  - oss-game-engine/engine-build/Cargo.toml
  - oss-game-engine/engine-core/Cargo.toml
  - oss-game-engine/engine-editor/Cargo.toml
  - oss-game-engine/engine-physics/Cargo.toml
  - oss-game-engine/engine-renderer/Cargo.toml
  - oss-game-engine/engine-runtime/Cargo.toml
  - oss-game-engine/engine-scripting/Cargo.toml

- **crates/** directory:
  - janus-cli/
  - janus-core/
  - janus-llm/
  - janus-wasm/
  - nexus-core/

- **oss-game-engine/** directory:
  - Contains 11 engine submodules (assets, audio, build, core, editor, physics, renderer, runtime, scripting)

## Sub-applications
- **jmaxing_app/** directory:
  - Frontend application

- **jmaxing_backend/** directory:
  - Backend application with schemas/ and core/ subdirectories

- **mesh_isp_dashboard/** directory:
  - Dashboard application

## Config and Deployment
- **config*.json** (12 files):
  - config.json
  - config_100m.json
  - config_33m.json
  - config_65m.json
  - config_avus_13b.json
  - config_avus_1b.json
  - config_avus_34b.json
  - config_avus_3b.json
  - config_avus_70b.json
  - config_avus_7b.json
  - core/config.json

- **schemas/** directories (3 locations):
  - schemas/ (root)
  - game_ai_database/schemas/
  - jmaxing_backend/schemas/

- **Deployment files**:
  - docker-compose.yml
  - Dockerfile (root and my-llm-project/)
  - setup.sh
  - setup_enhanced.sh

## Data Directories
- **holographic_brain_memory/** directory
- **training_data/** directory
- **core/** directory (root level - contains config.json)
- **game_ai_database/** directory

## Target Commit Verification
- Target commit **3854486** confirmed in git history
- Commit message: "Add Turing Halting Problem bypass system via holographic entanglement"
- Located 9 commits from current HEAD
