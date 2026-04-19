import { Enemy } from '../Enemy.js';
import { ENEMY_STATS } from '../../../data/enemies.js';

export class AshenWraith extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.ashenWraith, position);
  }
}

export class CinderHound extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.cinderHound, position);
    this.speed = ENEMY_STATS.cinderHound.speed;
  }
}

export class EmberKnight extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.emberKnight, position);
  }
}

export class SlagTitan extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.slagTitan, position);
  }
}

export class KorgathsEcho extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.korgathsEcho, position);
  }

  _onPhaseChange(phase) {
    super._onPhaseChange(phase);
    if (phase === 1) {
      // Phase 2: Faster, more fire attacks
      this.attacks = ['infernoSlam', 'ashWave', 'emberRain', 'korgathsRoar', 'infernoSlam'];
    } else if (phase === 2) {
      // Phase 3: Desperate, all attacks
      this.attacks = ['infernoSlam', 'infernoSlam', 'ashWave', 'emberRain', 'korgathsRoar'];
      this.speed *= 1.3;
    }
  }
}
