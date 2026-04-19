import { Enemy } from '../Enemy.js';
import { ENEMY_STATS } from '../../../data/enemies.js';

export class FrostSpecter extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.frostSpecter, position);
  }
}

export class IceColossus extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.iceColossus, position);
  }
}

export class FrozenChoir extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.frozenChoir, position);
    this.isRanged = true;
  }
}

export class YsoldesShard extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.ysoldesShard, position);
  }
}

export class MawSerpent extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.mawSerpent, position);
  }

  _onPhaseChange(phase) {
    super._onPhaseChange(phase);
    if (phase === 1) {
      this.attacks = ['tailSwipe', 'iceBreath', 'coilCrush', 'blizzardSummon', 'iceBreath'];
    } else if (phase === 2) {
      this.attacks = ['coilCrush', 'coilCrush', 'iceBreath', 'blizzardSummon', 'tailSwipe'];
      this.aggroRange = 30;
    }
  }
}
