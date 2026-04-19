import { StatusEffectManager } from '../combat/StatusEffects.js';

export const AI_STATES = {
  PATROL: 'PATROL',
  AGGRO: 'AGGRO',
  ATTACK: 'ATTACK',
  RETREAT: 'RETREAT',
  DEAD: 'DEAD',
  IDLE: 'IDLE',
  STAGGERED: 'STAGGERED',
};

export class Enemy {
  constructor(stats, position, id) {
    this.id = id || `enemy_${Math.random().toString(36).substr(2, 9)}`;
    this.stats = stats;
    this.name = stats.name;
    this.hp = stats.hp;
    this.maxHp = stats.maxHp;
    this.damage = stats.damage;
    this.defense = stats.defense;
    this.poise = stats.poise;
    this.currentPoise = stats.poise;
    this.speed = stats.speed;
    this.aggroRange = stats.aggroRange;
    this.attackRange = stats.attackRange;
    this.xpReward = stats.xpReward;
    this.soulsReward = stats.soulsReward;
    this.isBoss = stats.isBoss || false;
    this.isFinalBoss = stats.isFinalBoss || false;
    this.phaseThresholds = stats.phaseThresholds || [];
    this.currentPhase = 0;
    this.attacks = stats.attacks || ['slash'];
    this.dialogue = stats.dialogue || [];
    this.statusChance = stats.statusChance || {};

    this.position = { ...position };
    this.startPosition = { ...position };
    this.facingDirection = { x: 0, y: 0, z: 1 };
    this.velocity = { x: 0, y: 0, z: 0 };

    this.aiState = AI_STATES.PATROL;
    this.patrolTimer = 0;
    this.patrolTarget = { ...position };
    this.attackCooldown = 0;
    this.attackTimer = 0;
    this.aggroTimer = 0;
    this.retreatTimer = 0;
    this.staggerTimer = 0;
    this.dialogueIndex = 0;
    this.lastDialogueTime = 0;

    this.statusEffects = new StatusEffectManager();
    this.canAct = true;
    this.canAttack = true;
    this.isAttacking = false;

    this.mesh = null; // Set by renderer
    this.isDead = false;
    this.deathTimer = 0;

    // Boss phase tracking
    this.phaseChanged = false;
    this.phaseChangeTimer = 0;
  }

  update(deltaTime, playerPosition, combatSystem) {
    if (this.isDead) {
      this.deathTimer += deltaTime;
      return;
    }

    // Update status effects
    this.statusEffects.update(this, deltaTime);

    // Update cooldowns
    if (this.attackCooldown > 0) this.attackCooldown -= deltaTime;
    if (this.staggerTimer > 0) {
      this.staggerTimer -= deltaTime;
      if (this.staggerTimer <= 0) {
        this.canAct = true;
        this.aiState = AI_STATES.AGGRO;
      }
      return;
    }

    // Check stagger from status
    if (this.statusEffects.hasEffect('staggered')) {
      this.canAct = false;
      return;
    }

    // Check death
    if (this.hp <= 0) {
      this.die();
      return;
    }

    // Check boss phase transitions
    this._checkPhaseTransition();

    // AI state machine
    const distToPlayer = this._distanceTo(playerPosition);

    switch (this.aiState) {
      case AI_STATES.PATROL:
        this._updatePatrol(deltaTime, playerPosition, distToPlayer);
        break;
      case AI_STATES.AGGRO:
        this._updateAggro(deltaTime, playerPosition, distToPlayer);
        break;
      case AI_STATES.ATTACK:
        this._updateAttack(deltaTime, playerPosition, distToPlayer, combatSystem);
        break;
      case AI_STATES.RETREAT:
        this._updateRetreat(deltaTime, playerPosition, distToPlayer);
        break;
    }
  }

  _updatePatrol(deltaTime, playerPosition, distToPlayer) {
    // Check if player is in aggro range
    if (distToPlayer < this.aggroRange) {
      this.aiState = AI_STATES.AGGRO;
      this.aggroTimer = 0;
      this._speakDialogue();
      return;
    }

    // Patrol around start position
    this.patrolTimer -= deltaTime;
    if (this.patrolTimer <= 0) {
      this.patrolTimer = 2 + Math.random() * 3;
      const angle = Math.random() * Math.PI * 2;
      const radius = 3 + Math.random() * 4;
      this.patrolTarget = {
        x: this.startPosition.x + Math.cos(angle) * radius,
        z: this.startPosition.z + Math.sin(angle) * radius,
      };
    }

    this._moveToward(this.patrolTarget, this.speed * 0.4, deltaTime);
  }

  _updateAggro(deltaTime, playerPosition, distToPlayer) {
    this.aggroTimer += deltaTime;

    // Retreat if low HP or player too far
    if (this.hp < this.maxHp * 0.2 && !this.isBoss) {
      this.aiState = AI_STATES.RETREAT;
      this.retreatTimer = 3;
      return;
    }

    if (distToPlayer > this.aggroRange * 1.5 && !this.isBoss) {
      this.aiState = AI_STATES.PATROL;
      return;
    }

    // Move toward player
    if (distToPlayer > this.attackRange) {
      this._moveToward(playerPosition, this.speed, deltaTime);
    } else {
      this.aiState = AI_STATES.ATTACK;
    }
  }

  _updateAttack(deltaTime, playerPosition, distToPlayer, combatSystem) {
    // If player moved away, go back to aggro
    if (distToPlayer > this.attackRange * 1.3) {
      this.aiState = AI_STATES.AGGRO;
      return;
    }

    // Face player
    this._faceTarget(playerPosition);

    // Attack if cooldown is ready
    if (this.attackCooldown <= 0 && this.canAttack) {
      this._performAttack(combatSystem);
    }
  }

  _updateRetreat(deltaTime, playerPosition, distToPlayer) {
    this.retreatTimer -= deltaTime;

    // Move away from player
    const dx = this.position.x - playerPosition.x;
    const dz = this.position.z - playerPosition.z;
    const len = Math.sqrt(dx * dx + dz * dz);
    if (len > 0) {
      this.position.x += (dx / len) * this.speed * 0.6 * deltaTime;
      this.position.z += (dz / len) * this.speed * 0.6 * deltaTime;
    }

    if (this.retreatTimer <= 0 || distToPlayer > this.aggroRange) {
      this.aiState = AI_STATES.PATROL;
    }
  }

  _performAttack(combatSystem) {
    if (!combatSystem) return;

    const attackName = this.attacks[Math.floor(Math.random() * this.attacks.length)];
    const damage = this._getAttackDamage(attackName);

    this.isAttacking = true;
    this.attackCooldown = 1.5 + Math.random() * 1.0;

    // Apply status effects based on chance
    const statusResult = {};
    for (const [effectId, chance] of Object.entries(this.statusChance)) {
      if (Math.random() < chance) {
        statusResult[effectId] = true;
      }
    }

    // Signal to game engine that this enemy is attacking
    this.pendingAttack = {
      damage,
      attackName,
      statusEffects: statusResult,
    };

    setTimeout(() => {
      this.isAttacking = false;
    }, 600);
  }

  _getAttackDamage(attackName) {
    const multipliers = {
      slash: 1.0, lunge: 1.2, bite: 0.9, pounce: 1.3,
      heavySlash: 1.8, shieldBash: 0.7, emberBurst: 1.5,
      groundSlam: 2.0, lavaSpew: 1.4, stomp: 1.6,
      infernoSlam: 2.2, ashWave: 1.3, emberRain: 1.1, korgathsRoar: 0.5,
      frostSlash: 1.0, iceSpike: 1.2, glacialSlam: 2.0, iceWall: 0.8,
      freezeBreath: 1.3, sonicBlast: 1.1, freezeChant: 0.6,
      crystalSlash: 1.0, shardBurst: 1.4, iceArmor: 0,
      tailSwipe: 1.5, iceBreath: 1.6, coilCrush: 2.5, blizzardSummon: 0.8,
      boneSlash: 1.0, boneSpear: 1.3, holySmite: 1.5, shieldCharge: 1.2,
      divineWrath: 1.8, memoryDrain: 1.0, phaseSlash: 1.1,
      loreBlast: 1.3, bookBarrage: 0.9, memoryErase: 1.2,
      godSmite: 2.0, memoryFlood: 1.4, aethonRoar: 0.6, realityTear: 1.7,
      mindDrain: 1.0, crystalSpike: 1.2, shardExplosion: 1.6, crystalArmor: 0,
      forgottenSlash: 1.0, erasure: 1.4, constructSlam: 1.8, crystalBeam: 1.5,
      dataOverload: 1.3, memoryConsume: 1.6, crystalStorm: 1.2, mindErase: 1.5,
      thoughtAbsorb: 1.1, veilSlam: 1.8, realityPunch: 1.5, voidRoar: 0.7,
      fractureSlash: 1.3, realityTear2: 1.6, mirrorSlash: 1.2, copyAttack: 1.0,
      voidStep: 0.5, unwriteSlash: 1.4, existenceErase: 1.8, voidPulse: 1.2,
      architectsWill: 2.0, realityCollapse: 2.5, veilTear: 1.7, sovereignCall: 0.8,
      voidSlam: 2.0, eyeBeam: 1.8, voidPull: 1.0, summonWraiths: 0.5,
      realityErasure: 2.2, voidCopies: 0.8, starConsumption: 1.9, memoryDrain2: 1.3,
      voidRain: 1.5, trueNameDamage: 3.0, finalErasure: 5.0,
    };
    return this.damage * (multipliers[attackName] || 1.0);
  }

  _checkPhaseTransition() {
    if (!this.isBoss || this.phaseThresholds.length === 0) return;

    const hpPercent = this.hp / this.maxHp;
    const nextPhase = this.currentPhase + 1;

    if (nextPhase <= this.phaseThresholds.length) {
      const threshold = this.phaseThresholds[nextPhase - 1];
      if (hpPercent <= threshold) {
        this.currentPhase = nextPhase;
        this.phaseChanged = true;
        this.phaseChangeTimer = 2.0;
        this._onPhaseChange(nextPhase);
      }
    }
  }

  _onPhaseChange(phase) {
    // Boost stats on phase change
    this.speed *= 1.2;
    this.damage *= 1.15;
    this.attackCooldown = 0; // Reset cooldown for dramatic effect

    // Speak dialogue
    if (this.dialogue && this.dialogue.length > 0) {
      const idx = Math.min(phase, this.dialogue.length - 1);
      this.pendingDialogue = this.dialogue[idx];
    }
  }

  _speakDialogue() {
    if (!this.dialogue || this.dialogue.length === 0) return;
    const now = Date.now();
    if (now - this.lastDialogueTime < 10000) return; // Don't spam
    this.pendingDialogue = this.dialogue[this.dialogueIndex % this.dialogue.length];
    this.dialogueIndex++;
    this.lastDialogueTime = now;
  }

  _moveToward(target, speed, deltaTime) {
    const dx = target.x - this.position.x;
    const dz = (target.z || 0) - this.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist < 0.1) return;

    const nx = dx / dist;
    const nz = dz / dist;
    this.position.x += nx * speed * deltaTime;
    this.position.z += nz * speed * deltaTime;
    this.facingDirection = { x: nx, y: 0, z: nz };
  }

  _faceTarget(target) {
    const dx = target.x - this.position.x;
    const dz = target.z - this.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist > 0) {
      this.facingDirection = { x: dx / dist, y: 0, z: dz / dist };
    }
  }

  _distanceTo(target) {
    const dx = target.x - this.position.x;
    const dz = target.z - this.position.z;
    return Math.sqrt(dx * dx + dz * dz);
  }

  die() {
    this.isDead = true;
    this.aiState = AI_STATES.DEAD;
    this.hp = 0;
    this.canAct = false;
    this.canAttack = false;
  }

  takeDamage(damage) {
    if (this.isDead) return 0;
    const finalDamage = Math.max(1, damage - this.defense);
    this.hp = Math.max(0, this.hp - finalDamage);

    // Poise damage
    this.currentPoise -= finalDamage * 0.3;
    if (this.currentPoise <= 0) {
      this.currentPoise = this.poise;
      this.staggerTimer = 0.8;
      this.canAct = false;
    }

    // Transition to aggro if patrolling
    if (this.aiState === AI_STATES.PATROL || this.aiState === AI_STATES.IDLE) {
      this.aiState = AI_STATES.AGGRO;
    }

    if (this.hp <= 0) this.die();
    return finalDamage;
  }
}
