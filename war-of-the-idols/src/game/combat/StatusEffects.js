export const STATUS_DEFINITIONS = {
  burning: {
    id: 'burning',
    name: 'Burning',
    color: '#ff4400',
    icon: '🔥',
    damagePerSecond: 5,
    duration: 8,
    description: 'Taking 5 fire damage per second for 8 seconds.',
    stackable: false,
    onApply: (entity) => {},
    onTick: (entity, delta) => {
      entity.hp -= STATUS_DEFINITIONS.burning.damagePerSecond * delta;
    },
    onRemove: (entity) => {},
  },
  frostbite: {
    id: 'frostbite',
    name: 'Frostbite',
    color: '#4488ff',
    icon: '❄️',
    duration: 10,
    description: 'Stamina regeneration halved for 10 seconds.',
    stackable: false,
    onApply: (entity) => {
      entity.staminaRegenMult = (entity.staminaRegenMult || 1) * 0.5;
    },
    onTick: (entity, delta) => {},
    onRemove: (entity) => {
      entity.staminaRegenMult = Math.min(1, (entity.staminaRegenMult || 0.5) * 2);
    },
  },
  voidCorruption: {
    id: 'voidCorruption',
    name: 'Void Corruption',
    color: '#aa00ff',
    icon: '💜',
    duration: 15,
    description: 'Max HP reduced by 20% while active.',
    stackable: false,
    onApply: (entity) => {
      entity._preCorruptionMaxHp = entity.maxHp;
      entity.maxHp = Math.floor(entity.maxHp * 0.8);
      entity.hp = Math.min(entity.hp, entity.maxHp);
    },
    onTick: (entity, delta) => {},
    onRemove: (entity) => {
      if (entity._preCorruptionMaxHp) {
        entity.maxHp = entity._preCorruptionMaxHp;
        delete entity._preCorruptionMaxHp;
      }
    },
  },
  hollowing: {
    id: 'hollowing',
    name: 'Hollowing',
    color: '#333333',
    icon: '💀',
    duration: -1, // permanent until cured
    description: 'Each stack reduces max HP by 10%. Max 5 stacks.',
    stackable: true,
    maxStacks: 5,
    onApply: (entity) => {
      // Applied via death mechanic, not status tick
    },
    onTick: (entity, delta) => {},
    onRemove: (entity) => {},
  },
  enlightened: {
    id: 'enlightened',
    name: 'Enlightened',
    color: '#ffdd00',
    icon: '✨',
    duration: 20,
    description: 'Damage increased by 25% and stamina costs reduced.',
    stackable: false,
    onApply: (entity) => {
      entity.damageMult = (entity.damageMult || 1) * 1.25;
      entity.staminaCostMult = (entity.staminaCostMult || 1) * 0.75;
    },
    onTick: (entity, delta) => {},
    onRemove: (entity) => {
      entity.damageMult = (entity.damageMult || 1.25) / 1.25;
      entity.staminaCostMult = (entity.staminaCostMult || 0.75) / 0.75;
    },
  },
  staggered: {
    id: 'staggered',
    name: 'Staggered',
    color: '#ffaa00',
    icon: '⚡',
    duration: 1.5,
    description: 'Cannot act for 1.5 seconds.',
    stackable: false,
    onApply: (entity) => {
      entity.canAct = false;
    },
    onTick: (entity, delta) => {},
    onRemove: (entity) => {
      entity.canAct = true;
    },
  },
  disarmed: {
    id: 'disarmed',
    name: 'Disarmed',
    color: '#ff8800',
    icon: '🗡️',
    duration: 5,
    description: 'Cannot attack for 5 seconds.',
    stackable: false,
    onApply: (entity) => {
      entity.canAttack = false;
    },
    onTick: (entity, delta) => {},
    onRemove: (entity) => {
      entity.canAttack = true;
    },
  },
};

export class StatusEffectManager {
  constructor() {
    this.activeEffects = {}; // { effectId: { timeRemaining, stacks } }
  }

  applyEffect(entity, effectId, stacks = 1) {
    const def = STATUS_DEFINITIONS[effectId];
    if (!def) return false;

    if (this.activeEffects[effectId]) {
      if (def.stackable) {
        const current = this.activeEffects[effectId];
        current.stacks = Math.min((current.stacks || 1) + stacks, def.maxStacks || 5);
        current.timeRemaining = def.duration; // refresh duration
      } else {
        this.activeEffects[effectId].timeRemaining = def.duration; // refresh
      }
    } else {
      this.activeEffects[effectId] = {
        timeRemaining: def.duration,
        stacks: stacks,
      };
      def.onApply(entity);
    }
    return true;
  }

  removeEffect(entity, effectId) {
    if (this.activeEffects[effectId]) {
      const def = STATUS_DEFINITIONS[effectId];
      if (def) def.onRemove(entity);
      delete this.activeEffects[effectId];
    }
  }

  clearAll(entity) {
    for (const effectId of Object.keys(this.activeEffects)) {
      this.removeEffect(entity, effectId);
    }
  }

  update(entity, deltaTime) {
    for (const [effectId, data] of Object.entries(this.activeEffects)) {
      const def = STATUS_DEFINITIONS[effectId];
      if (!def) continue;

      def.onTick(entity, deltaTime);

      if (data.timeRemaining > 0) {
        data.timeRemaining -= deltaTime;
        if (data.timeRemaining <= 0) {
          this.removeEffect(entity, effectId);
        }
      }
    }
  }

  hasEffect(effectId) {
    return !!this.activeEffects[effectId];
  }

  getActiveEffects() {
    return Object.entries(this.activeEffects).map(([id, data]) => ({
      id,
      ...data,
      def: STATUS_DEFINITIONS[id],
    }));
  }
}
