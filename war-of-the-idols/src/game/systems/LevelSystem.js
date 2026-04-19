export const LEVEL_COSTS = {
  getXPForLevel: (level) => {
    if (level <= 10) return level * 100;
    if (level <= 25) return 1000 + (level - 10) * 200;
    if (level <= 50) return 4000 + (level - 25) * 500;
    return 16500 + (level - 50) * 1000;
  },
  getSoulCostForLevel: (level) => {
    if (level <= 10) return 100;
    if (level <= 25) return 300;
    if (level <= 50) return 800;
    return 2000;
  },
};

export const STAT_EFFECTS = {
  strength: {
    description: 'Increases physical damage and carry weight',
    perPoint: { physicalDamage: 2, carryWeight: 5 },
  },
  dexterity: {
    description: 'Increases attack speed and critical chance',
    perPoint: { attackSpeed: 0.02, critChance: 0.01 },
  },
  arcane: {
    description: 'Increases void/magic damage and status effect potency',
    perPoint: { voidDamage: 2.5, statusPotency: 0.02 },
  },
  endurance: {
    description: 'Increases max stamina and stamina regeneration',
    perPoint: { maxStamina: 8, staminaRegen: 0.5 },
  },
  vitality: {
    description: 'Increases max HP and poise',
    perPoint: { maxHp: 15, poise: 5 },
  },
};

export class LevelSystem {
  constructor(gameState) {
    this.gameState = gameState;
  }

  addXP(amount) {
    const player = this.gameState.player;
    player.xp += amount;
    let leveled = false;
    while (player.xp >= LEVEL_COSTS.getXPForLevel(player.level)) {
      player.xp -= LEVEL_COSTS.getXPForLevel(player.level);
      player.level += 1;
      player.statPoints = (player.statPoints || 0) + 3;
      leveled = true;
    }
    return leveled;
  }

  canLevelUp(stat) {
    const player = this.gameState.player;
    const cost = LEVEL_COSTS.getSoulCostForLevel(player.level);
    return player.souls >= cost && (player.statPoints || 0) > 0;
  }

  levelUpStat(stat) {
    const player = this.gameState.player;
    if (!STAT_EFFECTS[stat]) return false;
    if ((player.statPoints || 0) <= 0) return false;

    player[stat] = (player[stat] || 10) + 1;
    player.statPoints -= 1;

    // Apply stat effects
    const effects = STAT_EFFECTS[stat].perPoint;
    if (effects.maxHp) {
      player.maxHp += effects.maxHp;
      player.hp = Math.min(player.hp + effects.maxHp, player.maxHp);
    }
    if (effects.maxStamina) {
      player.maxStamina += effects.maxStamina;
      player.stamina = Math.min(player.stamina + effects.maxStamina, player.maxStamina);
    }
    if (effects.poise) {
      player.maxPoise = (player.maxPoise || 50) + effects.poise;
    }

    return true;
  }

  getXPProgress() {
    const player = this.gameState.player;
    const needed = LEVEL_COSTS.getXPForLevel(player.level);
    return { current: player.xp, needed, percent: (player.xp / needed) * 100 };
  }

  addSouls(amount) {
    this.gameState.player.souls += amount;
  }

  loseSouls() {
    // On death, souls are dropped at death location
    const lost = this.gameState.player.souls;
    this.gameState.player.souls = 0;
    this.gameState.droppedSouls = {
      amount: lost,
      position: { ...this.gameState.player.position },
      zone: this.gameState.currentZone,
    };
    return lost;
  }

  recoverSouls() {
    if (this.gameState.droppedSouls && this.gameState.droppedSouls.zone === this.gameState.currentZone) {
      const recovered = this.gameState.droppedSouls.amount;
      this.gameState.player.souls += recovered;
      this.gameState.droppedSouls = null;
      return recovered;
    }
    return 0;
  }

  getPlayerDerivedStats(player) {
    const str = player.strength || 10;
    const dex = player.dexterity || 10;
    const arc = player.arcane || 10;
    const end = player.endurance || 10;
    const vit = player.vitality || 10;

    return {
      physicalDamage: 10 + str * 2,
      voidDamage: 5 + arc * 2.5,
      attackSpeed: 1.0 + (dex - 10) * 0.02,
      critChance: 0.05 + (dex - 10) * 0.01,
      maxHp: 100 + (vit - 10) * 15,
      maxStamina: 100 + (end - 10) * 8,
      staminaRegen: 15 + (end - 10) * 0.5,
      maxPoise: 50 + (vit - 10) * 5,
      carryWeight: 50 + str * 5,
    };
  }
}
