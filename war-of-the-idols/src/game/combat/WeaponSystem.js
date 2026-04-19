import { ITEMS } from '../../data/items.js';

export const WEAPON_MOVE_SETS = {
  sword: {
    lightAttack: { damage: 1.0, staminaCost: 15, range: 2.5, speed: 0.4, animation: 'slash' },
    heavyAttack: { damage: 1.8, staminaCost: 25, range: 3.0, speed: 0.7, animation: 'heavySlash' },
    backstab: { damage: 3.0, staminaCost: 0, range: 1.5, speed: 0.3, animation: 'backstab' },
  },
  dao: {
    lightAttack: { damage: 1.0, staminaCost: 14, range: 2.8, speed: 0.35, animation: 'curvedSlash' },
    heavyAttack: { damage: 1.7, staminaCost: 22, range: 3.2, speed: 0.65, animation: 'spinSlash' },
    backstab: { damage: 3.0, staminaCost: 0, range: 1.5, speed: 0.3, animation: 'backstab' },
  },
  axe: {
    lightAttack: { damage: 1.2, staminaCost: 18, range: 2.5, speed: 0.5, animation: 'axeSwing' },
    heavyAttack: { damage: 2.2, staminaCost: 30, range: 3.0, speed: 0.8, animation: 'heavyAxe' },
    backstab: { damage: 3.0, staminaCost: 0, range: 1.5, speed: 0.3, animation: 'backstab' },
  },
  staff: {
    lightAttack: { damage: 0.9, staminaCost: 12, range: 8.0, speed: 0.5, animation: 'voidBlast', isRanged: true },
    heavyAttack: { damage: 1.6, staminaCost: 20, range: 10.0, speed: 0.8, animation: 'voidBeam', isRanged: true },
    backstab: { damage: 2.5, staminaCost: 0, range: 1.5, speed: 0.3, animation: 'backstab' },
  },
  greatsword: {
    lightAttack: { damage: 1.3, staminaCost: 20, range: 3.5, speed: 0.6, animation: 'greatSlash' },
    heavyAttack: { damage: 2.5, staminaCost: 35, range: 4.0, speed: 1.0, animation: 'greatSmash' },
    backstab: { damage: 3.0, staminaCost: 0, range: 1.5, speed: 0.3, animation: 'backstab' },
  },
  shard: {
    lightAttack: { damage: 1.1, staminaCost: 16, range: 3.0, speed: 0.3, animation: 'shardSlash' },
    heavyAttack: { damage: 2.0, staminaCost: 24, range: 5.0, speed: 0.6, animation: 'voidExplosion' },
    backstab: { damage: 3.0, staminaCost: 0, range: 1.5, speed: 0.3, animation: 'backstab' },
  },
  unarmed: {
    lightAttack: { damage: 0.5, staminaCost: 15, range: 1.5, speed: 0.3, animation: 'punch' },
    heavyAttack: { damage: 0.8, staminaCost: 20, range: 1.8, speed: 0.5, animation: 'kick' },
    backstab: { damage: 2.0, staminaCost: 0, range: 1.5, speed: 0.3, animation: 'backstab' },
  },
};

export class WeaponSystem {
  constructor(gameState) {
    this.gameState = gameState;
  }

  getWeaponMoveSet(weaponId) {
    if (!weaponId) return WEAPON_MOVE_SETS.unarmed;
    const item = ITEMS[weaponId];
    if (!item) return WEAPON_MOVE_SETS.unarmed;
    return WEAPON_MOVE_SETS[item.meshType] || WEAPON_MOVE_SETS.sword;
  }

  getLightAttack(weaponId) {
    return this.getWeaponMoveSet(weaponId).lightAttack;
  }

  getHeavyAttack(weaponId) {
    return this.getWeaponMoveSet(weaponId).heavyAttack;
  }

  getBackstabAttack(weaponId) {
    return this.getWeaponMoveSet(weaponId).backstab;
  }

  getWeaponStatusEffect(weaponId) {
    if (!weaponId) return null;
    const item = ITEMS[weaponId];
    return item?.statusEffect || null;
  }

  getWeaponRange(weaponId) {
    const moveSet = this.getWeaponMoveSet(weaponId);
    return moveSet.lightAttack.range;
  }

  isRangedWeapon(weaponId) {
    if (!weaponId) return false;
    const item = ITEMS[weaponId];
    return item?.isRanged || false;
  }
}
