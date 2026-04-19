import { Enemy } from '../Enemy.js';
import { ENEMY_STATS } from '../../../data/enemies.js';

export class VoidSovereign extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.voidSovereign, position);
    this.trueNameSpoken = false;
    this.finalErasureCharging = false;
    this.finalErasureTimer = 0;
    this.arenaExpanded = false;
    this.wallsClosing = false;
    this.wallCloseTimer = 0;
    this.summonedCopies = [];
    this.hasSpokenIntro = false;
  }

  update(deltaTime, playerPosition, combatSystem) {
    super.update(deltaTime, playerPosition, combatSystem);

    // Phase-specific behaviors
    if (this.currentPhase >= 2 && !this.arenaExpanded) {
      this.arenaExpanded = true;
      this.pendingArenaExpand = true;
    }

    if (this.currentPhase >= 3 && !this.wallsClosing) {
      this.wallsClosing = true;
      this.wallCloseTimer = 0;
    }

    if (this.wallsClosing) {
      this.wallCloseTimer += deltaTime;
    }

    // Final Erasure charge
    if (this.finalErasureCharging) {
      this.finalErasureTimer -= deltaTime;
      if (this.finalErasureTimer <= 0) {
        this.finalErasureCharging = false;
        this.pendingFinalErasure = true;
      }
    }
  }

  _onPhaseChange(phase) {
    super._onPhaseChange(phase);

    if (phase === 1) {
      // Phase 2: Arena expands, new attacks
      this.attacks = [
        'realityErasure', 'voidCopies', 'starConsumption', 'memoryDrain',
        'voidSlam', 'eyeBeam', 'realityErasure'
      ];
      this.pendingDialogue = this.stats.dialogue[1];
    } else if (phase === 2) {
      // Phase 3: Walls close, desperate attacks
      this.attacks = [
        'voidRain', 'finalErasure', 'voidSlam', 'eyeBeam',
        'realityErasure', 'voidRain', 'voidRain'
      ];
      this.pendingDialogue = this.stats.dialogue[3];
      this.speed *= 1.3;
    }
  }

  _performAttack(combatSystem) {
    const attackName = this.attacks[Math.floor(Math.random() * this.attacks.length)];

    if (attackName === 'finalErasure' && !this.finalErasureCharging) {
      this.finalErasureCharging = true;
      this.finalErasureTimer = 10.0;
      this.pendingDialogue = "FINAL ERASURE... charging...";
      this.pendingAttack = { damage: 0, attackName: 'finalErasureCharge', statusEffects: {} };
      this.attackCooldown = 12.0;
      return;
    }

    if (attackName === 'voidCopies') {
      this.pendingAttack = { damage: 0, attackName: 'voidCopies', statusEffects: {}, spawnCopies: 3 };
      this.attackCooldown = 8.0;
      return;
    }

    if (attackName === 'trueNameDamage' && this.trueNameSpoken) {
      const damage = this._getAttackDamage('trueNameDamage');
      this.pendingAttack = { damage, attackName, statusEffects: { voidCorruption: true } };
      this.attackCooldown = 2.0;
      return;
    }

    super._performAttack(combatSystem);
  }

  receiveTrueName() {
    this.trueNameSpoken = true;
    // Massive damage vulnerability
    this.defense = Math.floor(this.defense * 0.3);
    this.pendingDialogue = "...you know my name... how... HOW?!";
    this.pendingTrueNameEffect = true;
  }

  interruptFinalErasure() {
    if (this.finalErasureCharging) {
      this.finalErasureCharging = false;
      this.finalErasureTimer = 0;
      this.pendingDialogue = "...impossible...";
      // Stagger the boss
      this.staggerTimer = 3.0;
      this.canAct = false;
      return true;
    }
    return false;
  }
}
