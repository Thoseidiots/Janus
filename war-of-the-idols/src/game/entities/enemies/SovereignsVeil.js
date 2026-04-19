import { Enemy } from '../Enemy.js';
import { ENEMY_STATS } from '../../../data/enemies.js';

export class VeilTitan extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.veilTitan, position);
  }
}

export class RealityFracture extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.realityFracture, position);
  }
}

export class VoidDoppelganger extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.voidDoppelganger, position);
  }
}

export class TheUnwritten extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.theUnwritten, position);
  }
}

export class VeilArchitect extends Enemy {
  constructor(position) {
    super(ENEMY_STATS.veilArchitect, position);
  }

  _onPhaseChange(phase) {
    super._onPhaseChange(phase);
    if (phase === 1) {
      this.attacks = ['architectsWill', 'realityCollapse', 'veilTear', 'sovereignCall', 'architectsWill'];
      this.aggroRange = 35;
    } else if (phase === 2) {
      this.attacks = ['realityCollapse', 'realityCollapse', 'architectsWill', 'veilTear', 'sovereignCall'];
      this.speed *= 1.5;
      this.damage *= 1.2;
    }
  }
}
