import { ITEMS } from '../../data/items.js';

export const MAX_SLOTS = 20;

export class InventorySystem {
  constructor(gameState) {
    this.gameState = gameState;
  }

  getInventory() {
    return this.gameState.inventory || { slots: [], equipped: { weapon: null, armor: null, offhand: null, talisman: null } };
  }

  addItem(itemId, quantity = 1) {
    const inv = this.getInventory();
    const itemDef = ITEMS[itemId];
    if (!itemDef) return false;

    if (itemDef.stackable) {
      const existing = inv.slots.find(s => s && s.itemId === itemId);
      if (existing) {
        existing.quantity = Math.min((existing.quantity || 1) + quantity, itemDef.maxStack || 99);
        return true;
      }
    }

    if (inv.slots.length >= MAX_SLOTS) return false;

    inv.slots.push({ itemId, quantity: itemDef.stackable ? quantity : 1 });
    this.gameState.inventory = inv;
    return true;
  }

  removeItem(itemId, quantity = 1) {
    const inv = this.getInventory();
    const idx = inv.slots.findIndex(s => s && s.itemId === itemId);
    if (idx === -1) return false;

    const slot = inv.slots[idx];
    if (slot.quantity && slot.quantity > quantity) {
      slot.quantity -= quantity;
    } else {
      inv.slots.splice(idx, 1);
      // Unequip if equipped
      for (const [slot2, equippedId] of Object.entries(inv.equipped)) {
        if (equippedId === itemId) {
          inv.equipped[slot2] = null;
        }
      }
    }
    this.gameState.inventory = inv;
    return true;
  }

  equipItem(itemId) {
    const inv = this.getInventory();
    const itemDef = ITEMS[itemId];
    if (!itemDef) return false;

    const slotMap = {
      weapon: 'weapon',
      armor: 'armor',
      offhand: 'offhand',
      talisman: 'talisman',
    };

    const slot = slotMap[itemDef.type];
    if (!slot) return false;

    // Check if item is in inventory
    const hasItem = inv.slots.some(s => s && s.itemId === itemId);
    if (!hasItem) return false;

    // Unequip current
    if (inv.equipped[slot]) {
      this.unequipSlot(slot);
    }

    inv.equipped[slot] = itemId;
    this.gameState.inventory = inv;
    this._applyEquipmentStats();
    return true;
  }

  unequipSlot(slot) {
    const inv = this.getInventory();
    if (inv.equipped[slot]) {
      inv.equipped[slot] = null;
      this.gameState.inventory = inv;
      this._applyEquipmentStats();
    }
  }

  useConsumable(itemId) {
    const inv = this.getInventory();
    const itemDef = ITEMS[itemId];
    if (!itemDef || itemDef.type !== 'consumable') return false;

    const hasItem = inv.slots.some(s => s && s.itemId === itemId);
    if (!hasItem) return false;

    const player = this.gameState.player;
    const effect = itemDef.effect;

    if (effect.heal) {
      player.hp = Math.min(player.hp + effect.heal, player.maxHp);
    }
    if (effect.stamina) {
      player.stamina = Math.min(player.stamina + effect.stamina, player.maxStamina);
    }
    if (effect.clearStatus) {
      player.statusEffects = {};
    }
    if (effect.removeHollowing) {
      player.hollowing = Math.max(0, (player.hollowing || 0) - effect.removeHollowing);
      // Recalculate max HP
      this._applyHollowingPenalty();
    }

    this.removeItem(itemId, 1);
    return true;
  }

  _applyEquipmentStats() {
    const inv = this.getInventory();
    const player = this.gameState.player;

    // Reset equipment bonuses
    player.equipmentBonus = {
      damage: 0,
      defense: 0,
      blockValue: 0,
      parryWindow: 0,
      stats: {},
      resistances: {},
      passiveEffects: {},
    };

    for (const [slot, itemId] of Object.entries(inv.equipped)) {
      if (!itemId) continue;
      const itemDef = ITEMS[itemId];
      if (!itemDef) continue;

      if (itemDef.damage) player.equipmentBonus.damage += itemDef.damage;
      if (itemDef.defense) player.equipmentBonus.defense += itemDef.defense;
      if (itemDef.blockValue) player.equipmentBonus.blockValue += itemDef.blockValue;
      if (itemDef.parryWindow) player.equipmentBonus.parryWindow = Math.max(player.equipmentBonus.parryWindow, itemDef.parryWindow);

      if (itemDef.stats) {
        for (const [stat, val] of Object.entries(itemDef.stats)) {
          player.equipmentBonus.stats[stat] = (player.equipmentBonus.stats[stat] || 0) + val;
        }
      }
      if (itemDef.resistances) {
        for (const [res, val] of Object.entries(itemDef.resistances)) {
          player.equipmentBonus.resistances[res] = (player.equipmentBonus.resistances[res] || 0) + val;
        }
      }
      if (itemDef.passiveEffect) {
        Object.assign(player.equipmentBonus.passiveEffects, itemDef.passiveEffect);
      }
    }
  }

  _applyHollowingPenalty() {
    const player = this.gameState.player;
    const stacks = player.hollowing || 0;
    const baseMaxHp = 100 + ((player.vitality || 10) - 10) * 15;
    const penalty = stacks * 0.1;
    player.maxHp = Math.floor(baseMaxHp * (1 - penalty));
    player.hp = Math.min(player.hp, player.maxHp);
  }

  getEquippedWeapon() {
    const inv = this.getInventory();
    const weaponId = inv.equipped.weapon;
    return weaponId ? ITEMS[weaponId] : null;
  }

  getEquippedArmor() {
    const inv = this.getInventory();
    const armorId = inv.equipped.armor;
    return armorId ? ITEMS[armorId] : null;
  }

  getTotalDefense() {
    const player = this.gameState.player;
    return (player.equipmentBonus?.defense || 0);
  }

  getTotalDamage() {
    const player = this.gameState.player;
    const weapon = this.getEquippedWeapon();
    const baseDamage = weapon ? weapon.damage : 10;
    const strBonus = ((player.strength || 10) - 10) * 2;
    const equipBonus = player.equipmentBonus?.damage || 0;
    return baseDamage + strBonus + equipBonus;
  }

  hasItem(itemId) {
    const inv = this.getInventory();
    return inv.slots.some(s => s && s.itemId === itemId);
  }

  getItemCount(itemId) {
    const inv = this.getInventory();
    const slot = inv.slots.find(s => s && s.itemId === itemId);
    return slot ? (slot.quantity || 1) : 0;
  }
}
