import { StatusEffectManager } from './StatusEffects.js';
import { WeaponSystem } from './WeaponSystem.js';
import { ITEMS } from '../../data/items.js';

export const STAMINA_COSTS = {
  attack: 15,
  heavyAttack: 25,
  dodge: 20,
  parry: 25,
  sprintPerSec: 5,
  block: 10,
};

export const COMBAT_CONSTANTS = {
  critMultiplier: 2.5,
  backstabMultiplier: 3.0,
  parryCounterMultiplier: 3.0,
  poiseBreakStagger: 1.5,
  staggerDuration: 1.5,
  parryWindow: 0.3,
  criticalWindow: 2.0,
  disarmDuration: 5.0,
  staminaRegenRate: 15, // per second
  staminaRegenDelay: 1.5, // seconds after last stamina use
};

export class CombatSystem {
  constructor(gameState) {
    this.gameState = gameState;
    this.weaponSystem = new WeaponSystem(gameState);
    this.playerStatusEffects = new StatusEffectManager();
    this.combatLog = [];
    this.lastStaminaUse = 0;
    this.isParrying = false;
    this.parryTimer = 0;
    this.criticalWindowTimer = 0;
    this.isInCriticalWindow = false;
    this.attackCooldown = 0;
    this.dodgeCooldown = 0;
  }

  update(deltaTime) {
    const player = this.gameState.player;
    if (!player) return;

    // Stamina regeneration
    const now = Date.now() / 1000;
    const timeSinceStaminaUse = now - this.lastStaminaUse;
    if (timeSinceStaminaUse >= COMBAT_CONSTANTS.staminaRegenDelay) {
      const regenRate = COMBAT_CONSTANTS.staminaRegenRate * (player.staminaRegenMult || 1);
      player.stamina = Math.min(player.maxStamina, player.stamina + regenRate * deltaTime);
    }

    // Update status effects
    this.playerStatusEffects.update(player, deltaTime);

    // Update parry timer
    if (this.isParrying) {
      this.parryTimer -= deltaTime;
      if (this.parryTimer <= 0) {
        this.isParrying = false;
        this.parryTimer = 0;
      }
    }

    // Update critical window
    if (this.isInCriticalWindow) {
      this.criticalWindowTimer -= deltaTime;
      if (this.criticalWindowTimer <= 0) {
        this.isInCriticalWindow = false;
      }
    }

    // Update cooldowns
    if (this.attackCooldown > 0) this.attackCooldown -= deltaTime;
    if (this.dodgeCooldown > 0) this.dodgeCooldown -= deltaTime;

    // Check zero stamina stagger
    if (player.stamina <= 0 && !this.playerStatusEffects.hasEffect('staggered')) {
      this.playerStatusEffects.applyEffect(player, 'staggered');
    }
  }

  playerAttack(targetEnemy, isHeavy = false) {
    const player = this.gameState.player;
    if (!player || this.attackCooldown > 0) return null;
    if (this.playerStatusEffects.hasEffect('staggered')) return null;

    const inv = this.gameState.inventory;
    const weaponId = inv?.equipped?.weapon;
    const moveSet = this.weaponSystem.getWeaponMoveSet(weaponId);
    const move = isHeavy ? moveSet.heavyAttack : moveSet.lightAttack;

    const staminaCost = move.staminaCost * (player.staminaCostMult || 1);
    if (player.stamina < staminaCost) {
      this.playerStatusEffects.applyEffect(player, 'staggered');
      return null;
    }

    player.stamina -= staminaCost;
    this.lastStaminaUse = Date.now() / 1000;
    this.attackCooldown = move.speed;

    // Calculate damage
    let damage = this._calculatePlayerDamage(weaponId, move);

    // Check for critical hit
    const critChance = 0.05 + ((player.dexterity || 10) - 10) * 0.01;
    const isCrit = Math.random() < critChance;
    if (isCrit) {
      damage *= COMBAT_CONSTANTS.critMultiplier;
    }

    // Check if in critical window (after poise break)
    if (this.isInCriticalWindow) {
      damage *= COMBAT_CONSTANTS.critMultiplier;
      this.isInCriticalWindow = false;
    }

    // Apply damage to enemy
    const result = this._applyDamageToEnemy(targetEnemy, damage, weaponId);
    result.isCrit = isCrit;
    result.attackType = isHeavy ? 'heavy' : 'light';

    // Apply weapon status effect
    const statusEffect = this.weaponSystem.getWeaponStatusEffect(weaponId);
    if (statusEffect && targetEnemy) {
      for (const [effectId, chance] of Object.entries(statusEffect)) {
        if (Math.random() < chance) {
          if (targetEnemy.statusEffects) {
            targetEnemy.statusEffects.applyEffect(targetEnemy, effectId);
          }
        }
      }
    }

    this._addCombatLog(`Player attacks for ${Math.floor(damage)} damage${isCrit ? ' (CRITICAL!)' : ''}`);
    return result;
  }

  playerBackstab(targetEnemy) {
    const player = this.gameState.player;
    if (!player) return null;

    const inv = this.gameState.inventory;
    const weaponId = inv?.equipped?.weapon;
    const moveSet = this.weaponSystem.getWeaponMoveSet(weaponId);
    const move = moveSet.backstab;

    let damage = this._calculatePlayerDamage(weaponId, move) * COMBAT_CONSTANTS.backstabMultiplier;

    const result = this._applyDamageToEnemy(targetEnemy, damage, weaponId);
    result.isBackstab = true;

    this._addCombatLog(`BACKSTAB! ${Math.floor(damage)} damage!`);
    return result;
  }

  startParry() {
    const player = this.gameState.player;
    if (!player || player.stamina < STAMINA_COSTS.parry) return false;
    if (this.playerStatusEffects.hasEffect('staggered')) return false;

    const inv = this.gameState.inventory;
    const offhandId = inv?.equipped?.offhand;
    const offhand = offhandId ? ITEMS[offhandId] : null;
    const parryWindow = offhand?.parryWindow || COMBAT_CONSTANTS.parryWindow;

    player.stamina -= STAMINA_COSTS.parry;
    this.lastStaminaUse = Date.now() / 1000;
    this.isParrying = true;
    this.parryTimer = parryWindow;
    return true;
  }

  checkParry(incomingDamage, attacker) {
    if (!this.isParrying) return false;

    // Successful parry!
    this.isParrying = false;
    this.parryTimer = 0;

    // Apply parry counter damage
    const counterDamage = incomingDamage * COMBAT_CONSTANTS.parryCounterMultiplier;
    if (attacker) {
      this._applyDamageToEnemy(attacker, counterDamage, null);
      // Disarm attacker
      if (attacker.statusEffects) {
        attacker.statusEffects.applyEffect(attacker, 'disarmed');
      }
    }

    // Check for parry effect on offhand
    const inv = this.gameState.inventory;
    const offhandId = inv?.equipped?.offhand;
    const offhand = offhandId ? ITEMS[offhandId] : null;
    if (offhand?.parryEffect && attacker?.statusEffects) {
      for (const [effectId, chance] of Object.entries(offhand.parryEffect)) {
        if (Math.random() < chance) {
          attacker.statusEffects.applyEffect(attacker, effectId);
        }
      }
    }

    this._addCombatLog(`PARRY! Counter for ${Math.floor(counterDamage)} damage!`);
    return true;
  }

  playerDodge(direction) {
    const player = this.gameState.player;
    if (!player || this.dodgeCooldown > 0) return false;
    if (player.stamina < STAMINA_COSTS.dodge) return false;
    if (this.playerStatusEffects.hasEffect('staggered')) return false;

    player.stamina -= STAMINA_COSTS.dodge;
    this.lastStaminaUse = Date.now() / 1000;
    this.dodgeCooldown = 0.8;

    return { direction, invincibilityFrames: 0.3 };
  }

  receivePlayerDamage(damage, attacker) {
    const player = this.gameState.player;
    if (!player) return 0;

    // Check parry
    if (this.checkParry(damage, attacker)) return 0;

    // Apply defense
    const defense = player.equipmentBonus?.defense || 0;
    const resistances = player.equipmentBonus?.resistances || {};
    let finalDamage = Math.max(1, damage - defense);

    // Apply hollowing damage reduction (inverse — hollowing makes you weaker)
    const hollowingStacks = player.hollowing || 0;
    // Hollowing is already applied to maxHp, no additional damage mod

    player.hp -= finalDamage;
    player.hp = Math.max(0, player.hp);

    // Poise damage
    player.currentPoise = (player.currentPoise || player.maxPoise || 50) - (damage * 0.5);
    if (player.currentPoise <= 0) {
      player.currentPoise = player.maxPoise || 50;
      this.isInCriticalWindow = true;
      this.criticalWindowTimer = COMBAT_CONSTANTS.criticalWindow;
      this.playerStatusEffects.applyEffect(player, 'staggered');
    }

    return finalDamage;
  }

  _calculatePlayerDamage(weaponId, move) {
    const player = this.gameState.player;
    const baseDamage = weaponId ? (ITEMS[weaponId]?.damage || 10) : 10;
    const strBonus = ((player.strength || 10) - 10) * 2;
    const arcBonus = ((player.arcane || 10) - 10) * 1;
    const damageMult = player.damageMult || 1;
    const equipBonus = player.equipmentBonus?.passiveEffects?.fireDamageBonus || 0;

    return (baseDamage + strBonus + arcBonus) * move.damage * damageMult;
  }

  _applyDamageToEnemy(enemy, damage, weaponId) {
    if (!enemy) return { damage: 0, killed: false };

    const defense = enemy.defense || 0;
    const finalDamage = Math.max(1, damage - defense);

    enemy.hp -= finalDamage;
    enemy.hp = Math.max(0, enemy.hp);

    // Poise damage
    enemy.currentPoise = (enemy.currentPoise || enemy.poise || 50) - (finalDamage * 0.3);
    if (enemy.currentPoise <= 0) {
      enemy.currentPoise = enemy.poise || 50;
      if (enemy.statusEffects) {
        enemy.statusEffects.applyEffect(enemy, 'staggered');
      }
    }

    return {
      damage: finalDamage,
      killed: enemy.hp <= 0,
      enemy,
    };
  }

  _addCombatLog(message) {
    this.combatLog.push({ message, time: Date.now() });
    if (this.combatLog.length > 20) this.combatLog.shift();
    this.gameState.lastCombatMessage = message;
  }

  getCombatLog() {
    return this.combatLog;
  }

  applyStatusToPlayer(effectId, stacks = 1) {
    const player = this.gameState.player;
    this.playerStatusEffects.applyEffect(player, effectId);
  }

  getPlayerStatusEffects() {
    return this.playerStatusEffects.getActiveEffects();
  }
}
