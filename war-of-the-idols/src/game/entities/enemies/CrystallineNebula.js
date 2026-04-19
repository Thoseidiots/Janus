import { Enemy } from '../Enemy.js';
import { ENEMY_STATS } from '../../../data/enemies.js';

export class ThoughtLeech extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.thoughtLeech, position);
  }
}

export class CrystalHorror extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.crystalHorror, position);
  }
}

export class TheForgotten extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.theForgotten, position);
  }
}

export class ArchivistsConstruct extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.archivistsConstruct, position);
  }
}

export class MemoryEater extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.memoryEater, position);
  }

  _onPhaseChange(phase) {
    super._onPhaseChange(phase);
    if (phase === 1) {
      this.attacks = ['memoryConsume', 'crystalStorm', 'mindErase', 'thoughtAbsorb', 'memoryConsume'];
      this.aggroRange = 30;
    } else if (phase === 2) {
      this.attacks = ['mindErase', 'mindErase', 'crystalStorm', 'memoryConsume', 'thoughtAbsorb'];
      this.speed *= 1.4;
      this.attackRange *= 1.3;
    }
  }
}
