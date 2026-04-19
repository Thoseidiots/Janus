import { STARTING_ITEMS } from '../../data/items.js';

export const DEFAULT_PLAYER = {
  hp: 100,
  maxHp: 100,
  stamina: 100,
  maxStamina: 100,
  xp: 0,
  level: 1,
  souls: 0,
  statPoints: 0,
  strength: 10,
  dexterity: 10,
  arcane: 10,
  endurance: 10,
  vitality: 10,
  hollowing: 0,
  maxPoise: 50,
  currentPoise: 50,
  position: { x: 0, y: 0, z: 0 },
  rotation: 0,
  velocity: { x: 0, y: 0, z: 0 },
  isGrounded: true,
  isSprinting: false,
  isAttacking: false,
  isDodging: false,
  canAct: true,
  canAttack: true,
  equipment: { weapon: null, armor: null, offhand: null, talisman: null },
  statusEffects: {},
  equipmentBonus: {
    damage: 0,
    defense: 0,
    blockValue: 0,
    parryWindow: 0,
    stats: {},
    resistances: {},
    passiveEffects: {},
  },
  staminaRegenMult: 1,
  damageMult: 1,
  staminaCostMult: 1,
  facingDirection: { x: 0, y: 0, z: -1 },
};

export function createPlayer() {
  return JSON.parse(JSON.stringify(DEFAULT_PLAYER));
}

export function createStartingInventory() {
  const slots = [];
  const equipped = { weapon: null, armor: null, offhand: null, talisman: null };

  STARTING_ITEMS.forEach(itemId => {
    const existing = slots.find(s => s.itemId === itemId);
    if (existing) {
      existing.quantity = (existing.quantity || 1) + 1;
    } else {
      slots.push({ itemId, quantity: 1 });
    }
  });

  // Auto-equip starting weapon and armor
  const weaponSlot = slots.find(s => s.itemId === 'rustySword');
  const armorSlot = slots.find(s => s.itemId === 'raggedCloak');
  if (weaponSlot) equipped.weapon = 'rustySword';
  if (armorSlot) equipped.armor = 'raggedCloak';

  return { slots, equipped };
}

export class PlayerController {
  constructor(gameState, inputManager, combatSystem) {
    this.gameState = gameState;
    this.inputManager = inputManager;
    this.combatSystem = combatSystem;
    this.moveSpeed = 5.0;
    this.sprintSpeed = 8.0;
    this.dodgeDistance = 4.0;
    this.dodgeTimer = 0;
    this.dodgeDirection = { x: 0, z: 0 };
    this.isDodging = false;
    this.invincibilityTimer = 0;
    this.attackTimer = 0;
    this.isAttacking = false;
    this.attackHitbox = null;
    this.interactCooldown = 0;
  }

  update(deltaTime, enemies, npcs) {
    const player = this.gameState.player;
    if (!player) return;

    // Don't process movement during dialogue
    if (this.gameState.activeDialogue) return;

    const input = this.inputManager.getState();

    // Update timers
    if (this.dodgeTimer > 0) this.dodgeTimer -= deltaTime;
    if (this.invincibilityTimer > 0) this.invincibilityTimer -= deltaTime;
    if (this.attackTimer > 0) this.attackTimer -= deltaTime;
    if (this.interactCooldown > 0) this.interactCooldown -= deltaTime;

    // Handle dodge
    if (this.isDodging) {
      const dodgeSpeed = this.dodgeDistance / 0.4;
      player.position.x += this.dodgeDirection.x * dodgeSpeed * deltaTime;
      player.position.z += this.dodgeDirection.z * dodgeSpeed * deltaTime;
      if (this.dodgeTimer <= 0) {
        this.isDodging = false;
        player.isDodging = false;
      }
      return;
    }

    // Movement
    if (player.canAct) {
      const isSprinting = input.shift && (input.w || input.a || input.s || input.d);
      const speed = isSprinting ? this.sprintSpeed : this.moveSpeed;

      let dx = 0, dz = 0;
      if (input.w) dz -= 1;
      if (input.s) dz += 1;
      if (input.a) dx -= 1;
      if (input.d) dx += 1;

      // Normalize diagonal movement
      if (dx !== 0 && dz !== 0) {
        const len = Math.sqrt(dx * dx + dz * dz);
        dx /= len;
        dz /= len;
      }

      // Sprint stamina cost
      if (isSprinting && (dx !== 0 || dz !== 0)) {
        player.stamina -= 5 * deltaTime;
        if (player.stamina < 0) {
          player.stamina = 0;
          player.isSprinting = false;
        } else {
          player.isSprinting = true;
          this.combatSystem.lastStaminaUse = Date.now() / 1000;
        }
      } else {
        player.isSprinting = false;
      }

      player.position.x += dx * speed * deltaTime;
      player.position.z += dz * speed * deltaTime;

      // Update facing direction
      if (dx !== 0 || dz !== 0) {
        player.facingDirection = { x: dx, y: 0, z: dz };
        player.rotation = Math.atan2(dx, dz);
      }

      // Clamp to zone bounds
      const zone = this.gameState.currentZoneData;
      if (zone) {
        const halfW = zone.size.width / 2;
        const halfD = zone.size.depth / 2;
        player.position.x = Math.max(-halfW, Math.min(halfW, player.position.x));
        player.position.z = Math.max(-halfD, Math.min(halfD, player.position.z));
      }
    }

    // Dodge (Space)
    if (input.spaceJustPressed && !this.isDodging && player.canAct) {
      const result = this.combatSystem.playerDodge(player.facingDirection);
      if (result) {
        this.isDodging = true;
        player.isDodging = true;
        this.dodgeTimer = 0.4;
        this.invincibilityTimer = result.invincibilityFrames;
        this.dodgeDirection = { x: player.facingDirection.x, z: player.facingDirection.z };
        if (this.dodgeDirection.x === 0 && this.dodgeDirection.z === 0) {
          this.dodgeDirection = { x: 0, z: 1 }; // default dodge backward
        }
      }
    }

    // Attack (F key)
    if (input.fJustPressed && !this.isAttacking && player.canAct && player.canAttack) {
      this._performAttack(enemies, false);
    }

    // Heavy attack (Shift+F)
    if (input.fJustPressed && input.shift && !this.isAttacking && player.canAct && player.canAttack) {
      this._performAttack(enemies, true);
    }

    // Parry (Q key)
    if (input.qJustPressed && player.canAct) {
      this.combatSystem.startParry();
    }

    // Interact (E key)
    if (input.eJustPressed && this.interactCooldown <= 0) {
      this._tryInteract(npcs);
    }
  }

  _performAttack(enemies, isHeavy) {
    const player = this.gameState.player;
    const inv = this.gameState.inventory;
    const weaponId = inv?.equipped?.weapon;

    // Find nearest enemy in range
    const range = this._getAttackRange(weaponId);
    let target = null;
    let minDist = range;

    if (enemies) {
      for (const enemy of enemies) {
        if (!enemy || enemy.hp <= 0) continue;
        const dx = enemy.position.x - player.position.x;
        const dz = enemy.position.z - player.position.z;
        const dist = Math.sqrt(dx * dx + dz * dz);
        if (dist < minDist) {
          minDist = dist;
          target = enemy;
        }
      }
    }

    // Check for backstab
    if (target && this._isBackstabPosition(player, target)) {
      const result = this.combatSystem.playerBackstab(target);
      if (result) {
        this.isAttacking = true;
        this.attackTimer = 0.5;
        player.isAttacking = true;
        setTimeout(() => {
          this.isAttacking = false;
          player.isAttacking = false;
        }, 500);
        this.gameState.lastAttackResult = result;
        return;
      }
    }

    const result = this.combatSystem.playerAttack(target, isHeavy);
    if (result !== null) {
      this.isAttacking = true;
      this.attackTimer = isHeavy ? 0.8 : 0.5;
      player.isAttacking = true;
      setTimeout(() => {
        this.isAttacking = false;
        player.isAttacking = false;
      }, isHeavy ? 800 : 500);
      this.gameState.lastAttackResult = result;
    }
  }

  _isBackstabPosition(player, enemy) {
    if (!enemy.facingDirection) return false;
    const dx = player.position.x - enemy.position.x;
    const dz = player.position.z - enemy.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist > 2.0) return false;

    // Check if player is behind enemy
    const dot = dx * enemy.facingDirection.x + dz * enemy.facingDirection.z;
    return dot > 0.5; // player is behind enemy
  }

  _getAttackRange(weaponId) {
    if (!weaponId) return 2.0;
    // Range is determined by weapon type; ranged weapons have longer range
    // We use a simple lookup based on known ranged weapon IDs
    const rangedWeapons = { voidStaff: 8 };
    if (rangedWeapons[weaponId]) return rangedWeapons[weaponId];
    return 3.0;
  }

  _tryInteract(npcs) {
    const player = this.gameState.player;
    const interactRange = 3.0;

    if (!npcs) return;
    for (const npc of npcs) {
      if (!npc) continue;
      const dx = npc.position.x - player.position.x;
      const dz = npc.position.z - player.position.z;
      const dist = Math.sqrt(dx * dx + dz * dz);
      if (dist < interactRange) {
        this.gameState.pendingInteract = npc.id;
        this.interactCooldown = 1.0;
        return;
      }
    }
  }

  isInvincible() {
    return this.invincibilityTimer > 0 || this.isDodging;
  }
}
