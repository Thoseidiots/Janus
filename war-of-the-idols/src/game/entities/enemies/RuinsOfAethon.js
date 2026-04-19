import { Enemy } from '../Enemy.js';
import { ENEMY_STATS } from '../../../data/enemies.js';

export class BoneConstruct extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.boneConstruct, position);
  }
}

export class FallenPaladin extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.fallenPaladin, position);
  }
}

export class MemorySpecter extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.memorySpecter, position);
  }
}

export class ArchivistsRegret extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.archivistsRegret, position);
    this.isRanged = true;
  }
}

export class EchoOfAethon extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.echoOfAethon, position);
  }

  _onPhaseChange(phase) {
    super._onPhaseChange(phase);
    if (phase === 1) {
      this.attacks = ['godSmite', 'memoryFlood', 'aethonRoar', 'realityTear', 'godSmite'];
    } else if (phase === 2) {
      this.attacks = ['godSmite', 'realityTear', 'realityTear', 'memoryFlood', 'aethonRoar'];
      this.attackRange *= 1.5;
    }
  }
}
