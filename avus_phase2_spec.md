JANUS PROJECT
Avus Phase 2 Training Specification

AAA Game Generation & Neural Rendering Quality
 
 
 
Author: Ishmael Sears
Version: 2.0  |  March 2026
 
 
 
 
1. Executive Summary
Avus Phase 1 training completed successfully with a final loss of 0.2253 over 20 epochs on dual T4 GPUs. The model demonstrated strong learning of structured 3D generation parameters and desktop screen-action pairs. Phase 2 addresses the next critical requirement: AAA-quality game rendering.
 
The failures identified in NVIDIA DLSS 5 provide a direct blueprint for what Avus must learn to avoid. This document specifies the training objectives, dataset requirements, architectural modifications, and loss functions needed to train Avus as a high-quality AAA game generation brain.
 
 
2. Phase 1 Results Summary
 
Field
Value
Model
Avus (142M parameters)
Architecture
768 dim, 12 layers, 12 heads, 4 KV heads
Training Hardware
Dual T4 GPU (Kaggle)
Epochs Completed
20 / 20
Final Loss
0.2253
Starting Loss
13.3236 (Epoch 1)
Weights Size
584.1 MB
Curricula Trained
3D Generation + Screen-Action
Status
Complete — weights saved to Kaggle dataset
 
 
3. DLSS 5 Failure Analysis & Janus Implications
Analysis of NVIDIA DLSS 5 failures reveals five critical failure modes that any AI-driven game renderer must solve. Each maps directly to a training requirement for Avus Phase 2.
 
3.1  Squibbling — Temporal Instability
Description: Pixels in eyes, textures, and fine details jitter or morph between frames because the AI has no memory of what it generated previously. Each frame is treated as a new image.
• Root Cause: No object permanence between frames — the model lacks a hidden state pool.
• Janus Impact: Any frame-by-frame game asset generation will produce flickering, unstable visuals.
• Fix: Recurrent Video Restoration (VSR) with Hidden State Attention. The model maintains a pool of previous hidden states and uses Selective Cross-Attention to anchor stable features (eyes, textures) to their history.
 
3.2  Yosification — Over-Smoothing
Description: The AI over-softens details, making everything look like a Snapchat filter. Skin pores, iris textures, and high-frequency surface details are destroyed by aggressive upscaling.
• Root Cause: Uniform application of the neural model across all regions — no distinction between detail-critical zones and smooth zones.
• Janus Impact: Generated characters and environments will look plastic and fake, breaking AAA quality standards.
• Fix: Spatial Masking with Native-Scale Processing. Define Protection Zones around high-frequency detail regions. These zones are processed at native resolution, not resized or upscaled, preserving the micro-details that make renders feel real.
 
3.3  Uncanny Valley — Geometric Drift
Description: Character faces deform during motion. Eye proportions change, blinks look alien, and facial geometry shifts in ways that trigger the uncanny valley response.
• Root Cause: No geometric constraint — the AI can freely distort pixel positions without penalty.
• Janus Impact: Generated characters will look wrong in motion even if they look correct in still frames.
• Fix: Geometric Constraint Loss. Generation is anchored to a 3D bone rig or mesh. The model is penalized if inter-ocular distance, pupil size, or facial proportions deviate beyond a strict percentage of the base model. Stylized aesthetics (manga, Seinen) are safer here — a distinct art style reduces direct comparison to real humans.
 
3.4  Ghosting — Disocclusion Failure
Description: When objects move and reveal the background, the AI hallucinates background content instead of correctly predicting what was occluded. Results in halos and ghosting artifacts.
• Root Cause: No optical flow guidance — the model does not understand how pixels transform between frames based on motion.
• Janus Impact: Moving objects in generated game scenes will leave trails and halos that break immersion.
• Fix: Optical Flow Warping Loss. Forces the model to ensure pixels in Frame B are logically transformed versions of Frame A based on motion vectors. Any hallucinated background content is penalized unless it matches the motion-predicted data from the previous frame.
 
3.5  Imaginary Lighting — Semantic Inconsistency
Description: The AI adds studio lighting to characters in pitch-black environments because its training data associates faces with bright lighting. Scene semantics are ignored.
• Root Cause: No semantic regularization — the model generates aesthetically pleasing outputs without respecting scene context.
• Janus Impact: Generated scenes will have lighting that contradicts the intended atmosphere, breaking Global Illumination logic.
• Fix: Image Semantic Regularization Loss. Scene descriptions are semantically tagged. If a scene is tagged Dark Alleyway, the model is penalized for adding edge highlights or rim lighting to characters within it.
 
 
4. Phase 2 Training Objectives
Phase 2 trains Avus on five new curricula, each targeting one of the identified failure modes. All curricula are synthetic — no real game footage or licensed assets required.
 
4.1  Temporal Consistency Curriculum
• Goal: Teach Avus to maintain object permanence across frames.
• Dataset: TemporalConsistencyDataset — sequences of (frame N description, frame N+1 description, stable feature list, action) tuples.
• Loss Addition: Hidden state similarity penalty — generated features that deviate too far from previous state history are penalized.
• Target: Avus produces frame sequences where named features (eyes, textures, geometry) remain stable across 10+ frames.
 
4.2  Spatial Detail Curriculum
• Goal: Teach Avus to distinguish between detail-critical zones and smooth zones in a scene.
• Dataset: SpatialMaskDataset — scene descriptions with explicitly labeled Protection Zones (faces, eyes, inscriptions, fabric weave) and Smooth Zones (sky, flat walls).
• Loss Addition: Zone-aware reconstruction loss — detail accuracy is weighted higher inside Protection Zones.
• Target: Avus preserves high-frequency detail descriptors in Protection Zones while allowing simplification in Smooth Zones.
 
4.3  Geometric Constraint Curriculum
• Goal: Teach Avus to respect character proportions during motion.
• Dataset: GeometricConstraintDataset — character pose sequences with bone rig parameters, proportion bounds, and acceptable deviation ranges.
• Loss Addition: Proportional rigidity loss — penalizes any generated parameters that move key ratios (inter-ocular distance, limb proportions) outside defined bounds.
• Target: Avus generates character animations that maintain consistent proportions across all frames.
 
4.4  Optical Flow Curriculum
• Goal: Teach Avus to understand pixel motion between frames.
• Dataset: OpticalFlowDataset — frame pairs with motion vectors, occlusion masks, and correct background fill predictions.
• Loss Addition: Warping loss — penalizes hallucinated background content that does not match motion-predicted data.
• Target: Avus correctly predicts disoccluded background content when objects move, eliminating ghosting artifacts.
 
4.5  Semantic Lighting Curriculum
• Goal: Teach Avus to respect scene lighting semantics.
• Dataset: SemanticLightingDataset — scene descriptions with semantic tags (Dark Alleyway, Bright Exterior, Underground Cave) paired with valid and invalid lighting configurations.
• Loss Addition: Semantic regularization loss — penalizes lighting parameters that contradict scene semantic tags.
• Target: Avus never adds studio lighting to dark scenes or removes ambient light from bright scenes.
 
 
5. Architectural Modifications to Avus
 
5.1  Hidden State Pool
Add a rolling hidden state cache to the Avus forward pass. During sequence generation, the model attends to the last N hidden states using cross-attention, providing memory of previously generated features without requiring full context window expansion.
• Implementation: Add HiddenStatePool module to AvusConfig with pool_size parameter (default: 8 frames).
• Training: Pool is filled during curriculum training, initialized empty at inference start.
 
5.2  Zone-Aware Attention Masking
Extend the tokenized input format to include zone tags alongside scene descriptions. Attention weights are scaled higher for tokens within Protection Zone boundaries.
• New tokens: [ZONE_PROTECT_START], [ZONE_PROTECT_END], [ZONE_SMOOTH_START], [ZONE_SMOOTH_END]
• These join the existing special token set alongside [JSON_START], [ACT_START] etc.
 
5.3  Semantic Tag Conditioning
Scene semantic tags are prepended to prompts as structured conditioning signals. The model learns to condition all lighting and atmosphere outputs on these tags.
• Format: [SEM:dark_alleyway] Generate a hooded figure standing in shadow...
• New token: [SEM:tag_name] prefix added to tokenizer special tokens.
 
 
6. Dataset Generators to Build
The following Python files need to be created and added to the Janus repository. Each follows the same pattern as Grade3DGeneration and ScreenActionDataset — fully synthetic, no licensed content.
 
File
Curriculum
temporal_consistency_dataset.py
Temporal Consistency (Squibbling fix)
spatial_detail_dataset.py
Spatial Detail / Protection Zones (Yosification fix)
geometric_constraint_dataset.py
Geometric Constraint (Uncanny Valley fix)
optical_flow_dataset.py
Optical Flow / Warping (Ghosting fix)
semantic_lighting_dataset.py
Semantic Lighting (Imaginary Light fix)
 
 
7. Janus Integration Plan
Parallel to Phase 2 training, the following integration work connects existing Janus components into a working system. This work happens on the EliteDesk while Kaggle handles training.
 
7.1  Files to Create (New)
• janus_system_orchestrator.py — Master coordinator connecting all subsystems
• avus_inference.py — Loads trained Avus weights, exposes generate() for the rest of the system
• screen_interpreter.py — Converts raw screen pixels to text descriptions Avus can read
• game_generation_pipeline.py — Avus output → aaa_stack generators → game-ready assets
• revenue_execution.py — Real implementation replacing the hollow placeholder
 
7.2  Files to Make Real (Currently Hollow)
• jae_agent.py — Remove random.uniform() fake value generation, wire to real CEO state
• skill_executor.py — Already updated with JanusOS integration (done)
• continuous_operation.py — Verify Manus-built version is real, test it
• autonomous_improvement.py — Verify Manus-built version is real, test it
 
7.3  Integration Priority Order
1. avus_inference.py — Everything else depends on Avus being loadable
2. game_generation_pipeline.py — Connects Avus to aaa_stack
3. screen_interpreter.py — Gives Avus vision
4. janus_system_orchestrator.py — Ties everything together
5. revenue_execution.py — Real income loop
 
 
8. Phase 2 Training Schedule
 
Stage
Description
Stage 1
Train all 5 Phase 2 curricula combined (same approach as Phase 1 combined run)
Stage 2
Fine-tune on aaa_stack-specific game asset generation
Stage 3
Fine-tune on language coherence dataset v2 (basic language)
Stage 4
Fine-tune on cognitive architecture dataset v3 (Janus identity)
Hardware
Kaggle dual T4 x2 — 12 hour sessions, resume from saved weights
Target Loss
< 0.20 across all curricula
 
 
9. Success Criteria
Phase 2 is complete when Avus can demonstrate the following capabilities:
 
• Temporal: Generate a 10-frame character animation sequence with stable eye and texture features across all frames
• Spatial: Preserve described skin pore and iris detail in Protection Zone outputs while simplifying background geometry
• Geometric: Generate character poses that maintain proportional bounds across 20+ frame sequences
• Optical Flow: Correctly predict background fill when a moving object is removed from a scene description
• Semantic: Never generate bright rim lighting for scenes tagged as dark or underground
• Game Pipeline: End-to-end test — text prompt → Avus parameters → aaa_stack mesh → renderable asset
 
 
 
This document was generated as part of the Janus Project development session, March 2026.
Avus Phase 1 weights: avus_1b_weights.pt (584MB) — Kaggle dataset: janus-avus-weights (CC0 Public Domain)
Janus Project — Avus Phase 2 Training Specification    |    Page