import * as THREE from 'three';
import { SceneRenderer } from '../../rendering/SceneRenderer.js';
import { ZoneRenderer } from '../../rendering/ZoneRenderer.js';
import { EnemyRenderer } from '../../rendering/EnemyRenderer.js';
import { PlayerRenderer } from '../../rendering/PlayerRenderer.js';
import { EffectsRenderer } from '../../rendering/EffectsRenderer.js';
import { HUDRenderer } from '../../rendering/HUDRenderer.js';
import { InputManager } from './InputManager.js';
import { PhysicsEngine } from './PhysicsEngine.js';
import { AudioManager } from './AudioManager.js';
import { ZoneManager } from '../world/ZoneManager.js';
import { ShrineSystem } from '../world/ShrineSystem.js';
import { CombatSystem } from '../combat/CombatSystem.js';
import { InventorySystem } from '../systems/InventorySystem.js';
import { QuestSystem } from '../systems/QuestSystem.js';
import { LoreSystem } from '../systems/LoreSystem.js';
import { SaveSystem } from '../systems/SaveSystem.js';
import { LevelSystem } from '../systems/LevelSystem.js';
import { DialogueSystem } from '../systems/DialogueSystem.js';
import { PlayerController, createPlayer, createStartingInventory } from '../entities/Player.js';
import { ZONES } from '../../data/zones.js';
import { NPC_DATA } from '../../data/npcs.js';

export class GameEngine {
  constructor(canvas, gameState, onStateChange) {
    this.canvas = canvas;
    this.gameState = gameState;
    this.onStateChange = onStateChange;
    this.running = false;
    this.animFrameId = null;
    this.lastTime = 0;

    // Core systems
    this.sceneRenderer = new SceneRenderer(canvas);
    this.inputManager = new InputManager();
    this.physicsEngine = new PhysicsEngine();
    this.audioManager = new AudioManager();

    // Rendering subsystems
    const scene = this.sceneRenderer.getScene();
    this.zoneRenderer = new ZoneRenderer(scene);
    this.enemyRenderer = new EnemyRenderer(scene);
    this.playerRenderer = new PlayerRenderer(scene);
    this.effectsRenderer = new EffectsRenderer(scene);
    this.hudRenderer = new HUDRenderer();

    // Game systems
    this.combatSystem = new CombatSystem(gameState);
    this.inventorySystem = new InventorySystem(gameState);
    this.questSystem = new QuestSystem(gameState);
    this.loreSystem = new LoreSystem(gameState);
    this.saveSystem = new SaveSystem(gameState);
    this.levelSystem = new LevelSystem(gameState);
    this.dialogueSystem = new DialogueSystem(gameState);
    this.zoneManager = new ZoneManager(gameState);
    this.shrineSystem = new ShrineSystem(gameState);

    // Attach systems to gameState for cross-system access
    gameState.inventorySystem = this.inventorySystem;
    gameState.questSystem = this.questSystem;
    gameState.loreSystem = this.loreSystem;
    gameState.levelSystem = this.levelSystem;
    gameState.dialogueSystem = this.dialogueSystem;
    gameState.combatSystem = this.combatSystem;
    gameState.audioManager = this.audioManager;

    // Player controller
    this.playerController = new PlayerController(gameState, this.inputManager, this.combatSystem);

    // State
    this.currentZoneId = null;
    this.enemies = [];
    this.npcs = [];
    this.bossEnemy = null;
    this.zoneTransitionCooldown = 0;
    this.deathTimer = 0;
    this.isPlayerDead = false;
    this.pendingLevelUp = false;
    this.bossDialogueTimer = 0;
    this.soulsDropPosition = null;
    this.soulsDropMesh = null;
  }

  init(savedData = null) {
    if (savedData) {
      this._loadFromSave(savedData);
    } else {
      this._initNewGame();
    }

    // Load starting zone
    this._loadZone(this.gameState.currentZone || 'ashfeld');

    // Create player mesh
    this.playerRenderer.createPlayerMesh();
    this._updateWeaponMesh();

    // Start audio
    this.audioManager.resume();
    this.audioManager.playZoneAmbient(this.gameState.currentZone || 'ashfeld');

    this.running = true;
    this._loop(0);
  }

  _initNewGame() {
    this.gameState.player = createPlayer();
    this.gameState.inventory = createStartingInventory();
    this.gameState.currentZone = 'ashfeld';
    this.gameState.defeatedBosses = [];
    this.gameState.discoveredZones = ['ashfeld'];
    this.gameState.respawnZone = 'ashfeld';
    this.gameState.respawnPosition = { x: 0, y: 0, z: 0 };
    this.gameState.trueNameKnown = false;
    this.gameState.allLoreCollected = false;
    this.gameState.endingChosen = null;
    this.gameState.playTime = 0;
    this.gameState.newGamePlus = 0;
    this.gameState.droppedSouls = null;

    this.questSystem.initQuests();
    this.loreSystem.initLore();
    this.inventorySystem._applyEquipmentStats();
  }

  _loadFromSave(data) {
    this.gameState.player = data.player;
    this.gameState.inventory = data.inventory;
    this.gameState.currentZone = data.currentZone;
    this.gameState.defeatedBosses = data.defeatedBosses || [];
    this.gameState.discoveredZones = data.discoveredZones || [];
    this.gameState.quests = data.quests;
    this.gameState.lore = data.lore;
    this.gameState.trueNameKnown = data.trueNameKnown || false;
    this.gameState.allLoreCollected = data.allLoreCollected || false;
    this.gameState.endingChosen = data.endingChosen || null;
    this.gameState.playTime = data.playTime || 0;
    this.gameState.newGamePlus = data.newGamePlus || 0;
    this.gameState.respawnZone = data.currentZone;
    this.gameState.respawnPosition = { x: 0, y: 0, z: 0 };

    // Re-apply equipment stats
    this.inventorySystem._applyEquipmentStats();
    this.inventorySystem._applyHollowingPenalty();
  }

  _loadZone(zoneId) {
    this.currentZoneId = zoneId;

    // Clear old enemies
    this.enemyRenderer.clearAllEnemies();
    this.effectsRenderer.clearAll();

    // Load zone data
    const result = this.zoneManager.loadZone(zoneId);
    if (!result) return;

    const { enemies, boss, npcs, zoneData } = result;

    // Register shrine
    this.shrineSystem.registerShrine(zoneId, zoneData.shrinePosition);

    // Render zone
    this.zoneRenderer.loadZone(zoneData);

    // Create enemy meshes
    this.enemies = enemies;
    this.bossEnemy = boss;
    this.npcs = npcs;

    enemies.forEach(enemy => {
      this.enemyRenderer.createEnemyMesh(enemy);
    });

    if (boss) {
      this.enemyRenderer.createEnemyMesh(boss);
    }

    // Create NPC meshes
    this._createNPCMeshes(npcs);

    // Update game state
    this.gameState.currentZoneData = zoneData;
    this.gameState.currentZone = zoneId;

    // Notify UI
    this.onStateChange({ ...this.gameState });
  }

  _createNPCMeshes(npcs) {
    const scene = this.sceneRenderer.getScene();
    npcs.forEach(npc => {
      if (!npc) return;
      this._createNPCMesh(npc, scene);
    });
  }

  _createNPCMesh(npc, scene) {
    const group = new THREE.Group();

    const mat = new THREE.MeshStandardMaterial({
      color: npc.color || 0x88aaff,
      emissive: npc.emissive || 0x4466ff,
      emissiveIntensity: 0.8,
      transparent: true,
      opacity: 0.85,
    });

    // Ghost-like body
    const bodyGeo = new THREE.CapsuleGeometry(0.3, 1.0, 4, 8);
    const body = new THREE.Mesh(bodyGeo, mat);
    body.position.y = 0.8;
    group.add(body);

    // Glow
    const light = new THREE.PointLight(npc.emissive || 0x4466ff, 1.0, 5);
    light.position.y = 1;
    group.add(light);

    group.position.set(npc.position.x, npc.position.y, npc.position.z);
    group.userData.npcId = npc.id;
    scene.add(group);
    npc.mesh = group;
  }

  _loop(timestamp) {
    if (!this.running) return;

    const deltaTime = Math.min((timestamp - this.lastTime) / 1000, 0.05);
    this.lastTime = timestamp;

    this._update(deltaTime);
    this._render(deltaTime);

    this.animFrameId = requestAnimationFrame(this._loop.bind(this));
  }

  _update(deltaTime) {
    const player = this.gameState.player;
    if (!player) return;

    // Update play time
    this.gameState.playTime = (this.gameState.playTime || 0) + deltaTime;

    // Update input
    this.inputManager.update();
    const input = this.inputManager.getState();

    // Handle UI toggles
    if (input.iJustPressed) {
      this.onStateChange({ ...this.gameState, showInventory: !this.gameState.showInventory });
    }
    if (input.tabJustPressed) {
      this.onStateChange({ ...this.gameState, showQuests: !this.gameState.showQuests });
    }
    if (input.escapeJustPressed) {
      this.onStateChange({ ...this.gameState, showInventory: false, showQuests: false, showLore: false });
    }

    // Don't update game if UI is open or dialogue active
    if (this.gameState.showInventory || this.gameState.showQuests || this.gameState.showLore) return;
    if (this.gameState.activeDialogue) {
      this.dialogueSystem.update(deltaTime);
      return;
    }

    // Death handling
    if (this.isPlayerDead) {
      this.deathTimer -= deltaTime;
      if (this.deathTimer <= 0) {
        this._respawnPlayer();
      }
      return;
    }

    // Check player death
    if (player.hp <= 0 && !this.isPlayerDead) {
      this._handlePlayerDeath();
      return;
    }

    // Update combat system
    this.combatSystem.update(deltaTime);

    // Update player controller
    const allEnemies = this.zoneManager.getAllEnemies();
    this.playerController.update(deltaTime, allEnemies, this.npcs);

    // Handle pending interact
    if (this.gameState.pendingInteract) {
      this._handleInteract(this.gameState.pendingInteract);
      this.gameState.pendingInteract = null;
    }

    // Update enemies
    this._updateEnemies(deltaTime, player);

    // Update NPCs
    this.npcs.forEach(npc => npc && npc.update(deltaTime));

    // Check shrine interaction
    this._checkShrineInteraction(player, input);

    // Check zone transition
    this._checkZoneTransition(player);

    // Check souls recovery
    this._checkSoulsRecovery(player);

    // Physics
    this.physicsEngine.update([player, ...allEnemies], deltaTime, this.gameState.currentZoneData);

    // Pending XP
    if (this.gameState.pendingXP) {
      const leveled = this.levelSystem.addXP(this.gameState.pendingXP);
      this.gameState.pendingXP = 0;
      if (leveled) {
        this.pendingLevelUp = true;
        this.audioManager.playSFX('levelUp');
        this.onStateChange({ ...this.gameState, showLevelUp: true });
      }
    }

    // Enemy respawn after shrine rest
    if (this.gameState.pendingEnemyRespawn) {
      this.gameState.pendingEnemyRespawn = false;
      this.enemyRenderer.clearAllEnemies();
      this.zoneManager.respawnEnemies();
      this.enemies = this.zoneManager.enemies;
      this.enemies.forEach(enemy => this.enemyRenderer.createEnemyMesh(enemy));
    }

    // Autosave
    this.saveSystem.autoSave();

    // Notify UI of state changes
    this.onStateChange({ ...this.gameState });
  }

  _updateEnemies(deltaTime, player) {
    const allEnemies = this.zoneManager.getAllEnemies();

    allEnemies.forEach(enemy => {
      if (!enemy) return;

      enemy.update(deltaTime, player.position, this.combatSystem);

      // Handle pending attack
      if (enemy.pendingAttack) {
        const attack = enemy.pendingAttack;
        enemy.pendingAttack = null;

        // Check if player is in range
        const dist = this.physicsEngine.getDistance(enemy.position, player.position);
        if (dist <= enemy.attackRange + 1) {
          // Check if player is dodging (invincible)
          if (!this.playerController.isInvincible()) {
            const damage = this.combatSystem.receivePlayerDamage(attack.damage, enemy);
            if (damage > 0) {
              this.audioManager.playSFX('hit');
              this.effectsRenderer.createHitEffect(player.position, 0xff0000);

              // Apply status effects
              for (const [effectId] of Object.entries(attack.statusEffects || {})) {
                this.combatSystem.applyStatusToPlayer(effectId);
                this.effectsRenderer.createStatusEffect(player.position, effectId);
              }
            }
          }
        }
      }

      // Handle boss dialogue
      if (enemy.pendingDialogue) {
        this.gameState.bossDialogue = { text: enemy.pendingDialogue, timer: 4.0 };
        enemy.pendingDialogue = null;
        if (enemy.isBoss) this.audioManager.playSFX('bossRoar');
      }

      // Handle boss phase change
      if (enemy.phaseChanged) {
        enemy.phaseChanged = false;
        this.effectsRenderer.createBossPhaseEffect(enemy.position, enemy.stats.emissiveColor || 0xff0000);
        this.audioManager.playSFX('bossRoar');
      }

      // Handle boss arena expand
      if (enemy.pendingArenaExpand) {
        enemy.pendingArenaExpand = false;
        this.gameState.arenaExpanded = true;
      }

      // Handle final erasure
      if (enemy.pendingFinalErasure) {
        enemy.pendingFinalErasure = false;
        // Massive damage if not interrupted
        const damage = this.combatSystem.receivePlayerDamage(9999, enemy);
        if (damage > 0) {
          this._handlePlayerDeath();
        }
      }

      // Handle void copies spawn
      if (enemy.pendingAttack?.spawnCopies) {
        // Spawn phantom copies (visual only, low HP)
        // Simplified: just create visual effects
        for (let i = 0; i < enemy.pendingAttack.spawnCopies; i++) {
          this.effectsRenderer.createVoidEffect({
            x: enemy.position.x + (Math.random() - 0.5) * 10,
            y: 0,
            z: enemy.position.z + (Math.random() - 0.5) * 10,
          });
        }
      }

      // Handle enemy death
      if (enemy.isDead && !enemy._deathProcessed) {
        enemy._deathProcessed = true;
        this._handleEnemyDeath(enemy);
      }

      // Update mesh
      this.enemyRenderer.updateEnemyMesh(enemy);
    });

    // Update boss dialogue timer
    if (this.gameState.bossDialogue) {
      this.gameState.bossDialogue.timer -= deltaTime;
      if (this.gameState.bossDialogue.timer <= 0) {
        this.gameState.bossDialogue = null;
      }
    }
  }

  _handleEnemyDeath(enemy) {
    // Grant rewards
    const xpGained = this.levelSystem.addXP(enemy.xpReward);
    this.levelSystem.addSouls(enemy.soulsReward);

    this.audioManager.playSFX('death');
    this.effectsRenderer.createDeathEffect(enemy.position, enemy.stats.emissiveColor || 0xff0000);

    // Quest tracking
    if (enemy.isBoss) {
      this.questSystem.checkBossDefeat(enemy.id);
      this.zoneManager.onBossDefeated(enemy.id);
      this.saveSystem.saveOnEvent('boss_defeat');
      this.audioManager.playSFX('levelUp');

      // Check if final boss
      if (enemy.isFinalBoss) {
        this.gameState.finalBossDefeated = true;
        this.onStateChange({ ...this.gameState, showEnding: true });
      }
    }

    if (xpGained) {
      this.pendingLevelUp = true;
      this.onStateChange({ ...this.gameState, showLevelUp: true });
    }

    this.onStateChange({ ...this.gameState });
  }

  _handlePlayerDeath() {
    this.isPlayerDead = true;
    this.deathTimer = 3.0;
    this.audioManager.playSFX('death');

    // Hollowing
    const player = this.gameState.player;
    player.hollowing = Math.min(5, (player.hollowing || 0) + 1);
    this.inventorySystem._applyHollowingPenalty();

    // Drop souls
    this.levelSystem.loseSouls();

    this.onStateChange({ ...this.gameState, showDeath: true });
  }

  _respawnPlayer() {
    this.isPlayerDead = false;
    const respawn = this.shrineSystem.getRespawnPoint();

    // Load respawn zone if different
    if (respawn.zone !== this.currentZoneId) {
      this._loadZone(respawn.zone);
    }

    const player = this.gameState.player;
    player.position = { ...respawn.position };
    player.hp = player.maxHp;
    player.stamina = player.maxStamina;
    player.canAct = true;
    player.canAttack = true;

    // Clear status effects
    this.combatSystem.playerStatusEffects.clearAll(player);

    this.onStateChange({ ...this.gameState, showDeath: false });
  }

  _checkShrineInteraction(player, input) {
    const zoneId = this.currentZoneId;
    if (this.shrineSystem.isNearShrine(player.position, zoneId)) {
      this.gameState.nearShrine = true;
      if (input.eJustPressed) {
        this.shrineSystem.activateShrine(zoneId);
        this.audioManager.playSFX('shrine');
        this.effectsRenderer.createShrineActivationEffect(
          this.gameState.currentZoneData?.shrinePosition || { x: 0, y: 0, z: 0 }
        );
        this.saveSystem.saveOnEvent('shrine_rest');
        this.questSystem.completeObjective('theAshenPath', 'reach_ashfeld_shrine');
        this.onStateChange({ ...this.gameState, shrineActivated: true });
        setTimeout(() => this.onStateChange({ ...this.gameState, shrineActivated: false }), 2000);
      }
    } else {
      this.gameState.nearShrine = false;
    }
  }

  _checkZoneTransition(player) {
    if (this.zoneTransitionCooldown > 0) {
      this.zoneTransitionCooldown -= 0.016;
      return;
    }

    const zoneData = this.gameState.currentZoneData;
    const targetZone = this.zoneManager.getTransitionZones(player.position, zoneData);

    if (targetZone) {
      this.zoneTransitionCooldown = 2.0;
      this.questSystem.checkZoneTransition(targetZone);
      this.saveSystem.saveOnEvent('zone_transition');
      this.audioManager.playZoneAmbient(targetZone);
      this._loadZone(targetZone);
      player.position = { x: 0, y: 0, z: 0 };
    }
  }

  _checkSoulsRecovery(player) {
    if (!this.gameState.droppedSouls) return;
    if (this.gameState.droppedSouls.zone !== this.currentZoneId) return;

    const dist = this.physicsEngine.getDistance(player.position, this.gameState.droppedSouls.position);
    if (dist < 3) {
      const recovered = this.levelSystem.recoverSouls();
      if (recovered > 0) {
        this.audioManager.playSFX('itemPickup');
        this.gameState.lastMessage = `Recovered ${recovered} souls!`;
      }
    }
  }

  _handleInteract(npcId) {
    const npc = this.npcs.find(n => n && n.id === npcId);
    if (!npc) return;

    const npcData = NPC_DATA[npcId];
    if (!npcData) return;

    if (npcData.isMerchant) {
      this.dialogueSystem.startDialogue(
        npcData,
        () => this.onStateChange({ ...this.gameState }),
        () => this.onStateChange({ ...this.gameState, showShop: npcId })
      );
    } else {
      this.dialogueSystem.startDialogue(
        npcData,
        () => this.onStateChange({ ...this.gameState })
      );
    }

    // Quest triggers
    if (npcId === 'ghostNavigator') {
      this.questSystem.activateQuest('theHollowKnight');
    }
    if (npcId === 'archivistKael') {
      this.gameState.trueNameKnown = true;
      this.questSystem.completeObjective('theSovereignsName', 'learn_true_name');
    }

    this.onStateChange({ ...this.gameState });
  }

  _updateWeaponMesh() {
    const weaponId = this.gameState.inventory?.equipped?.weapon;
    this.playerRenderer.updateWeaponMesh(weaponId);
  }

  _render(deltaTime) {
    const player = this.gameState.player;

    // Update camera
    if (player) {
      this.sceneRenderer.updateCamera(player.position, player.rotation);
    }

    // Update player mesh
    if (player) {
      this.playerRenderer.updatePlayerMesh(player, deltaTime);
    }

    // Update zone particles/animations
    this.zoneRenderer.updateParticles(deltaTime);

    // Update effects
    this.effectsRenderer.update(deltaTime);

    // Update minimap
    const allEnemies = this.zoneManager.getAllEnemies();
    const shrinePos = this.gameState.currentZoneData?.shrinePosition;
    const minimapCanvas = this.hudRenderer.renderMinimap(
      player, allEnemies, this.npcs, this.gameState.currentZoneData, shrinePos
    );
    if (minimapCanvas) {
      this.gameState.minimapDataURL = minimapCanvas.toDataURL();
    }

    // Render scene
    this.sceneRenderer.render();
  }

  // Public API
  handleDialogueResponse(index) {
    this.dialogueSystem.selectResponse(index);
    this.onStateChange({ ...this.gameState });
  }

  handleDialogueSkip() {
    this.dialogueSystem.skipTyping();
    this.onStateChange({ ...this.gameState });
  }

  handleLevelUpStat(stat) {
    this.levelSystem.levelUpStat(stat);
    this.onStateChange({ ...this.gameState });
  }

  handleEquipItem(itemId) {
    this.inventorySystem.equipItem(itemId);
    this._updateWeaponMesh();
    this.onStateChange({ ...this.gameState });
  }

  handleUnequipSlot(slot) {
    this.inventorySystem.unequipSlot(slot);
    if (slot === 'weapon') this._updateWeaponMesh();
    this.onStateChange({ ...this.gameState });
  }

  handleUseItem(itemId) {
    this.inventorySystem.useConsumable(itemId);
    this.onStateChange({ ...this.gameState });
  }

  handleChooseEnding(choice) {
    this.gameState.endingChosen = choice;
    this.saveSystem.save('main');
    this.onStateChange({ ...this.gameState, showEnding: true, endingChosen: choice });
  }

  handleTrueName() {
    if (!this.gameState.trueNameKnown) return;
    const boss = this.bossEnemy;
    if (boss && boss.isFinalBoss && boss.receiveTrueName) {
      boss.receiveTrueName();
      this.effectsRenderer.createVoidEffect(boss.position);
      this.audioManager.playSFX('voidEffect');
    }
  }

  handleInterruptFinalErasure() {
    const boss = this.bossEnemy;
    if (boss && boss.isFinalBoss && boss.interruptFinalErasure) {
      const interrupted = boss.interruptFinalErasure();
      if (interrupted) {
        this.effectsRenderer.createBossPhaseEffect(boss.position, 0xffffff);
        this.audioManager.playSFX('parry');
      }
    }
  }

  stop() {
    this.running = false;
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
    }
    this.inputManager.destroy();
    this.audioManager.destroy();
    this.sceneRenderer.dispose();
  }
}
