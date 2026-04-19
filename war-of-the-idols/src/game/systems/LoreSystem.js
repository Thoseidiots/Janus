import { LORE_ENTRIES } from '../../data/lore.js';

export class LoreSystem {
  constructor(gameState) {
    this.gameState = gameState;
  }

  initLore() {
    const lore = {};
    LORE_ENTRIES.forEach(entry => {
      lore[entry.id] = { ...entry, discovered: false };
    });
    this.gameState.lore = lore;
  }

  discoverEntry(entryId) {
    const lore = this.gameState.lore || {};
    if (lore[entryId]) {
      lore[entryId].discovered = true;
      this.gameState.lore = lore;
      this._checkAllLoreCollected();
      return true;
    }
    return false;
  }

  discoverZoneLore(zoneId) {
    const lore = this.gameState.lore || {};
    let discovered = [];
    Object.values(lore).forEach(entry => {
      if (entry.zone === zoneId && !entry.discovered) {
        entry.discovered = true;
        discovered.push(entry.id);
      }
    });
    this.gameState.lore = lore;
    if (discovered.length > 0) this._checkAllLoreCollected();
    return discovered;
  }

  getDiscoveredEntries() {
    const lore = this.gameState.lore || {};
    return Object.values(lore).filter(e => e.discovered);
  }

  getAllEntries() {
    return Object.values(this.gameState.lore || {});
  }

  getEntriesByCategory(category) {
    const lore = this.gameState.lore || {};
    return Object.values(lore).filter(e => e.category === category && e.discovered);
  }

  _checkAllLoreCollected() {
    const lore = this.gameState.lore || {};
    const all = Object.values(lore);
    const allDiscovered = all.length > 0 && all.every(e => e.discovered);
    this.gameState.allLoreCollected = allDiscovered;
    return allDiscovered;
  }

  isAllLoreCollected() {
    return this.gameState.allLoreCollected || false;
  }

  getDiscoveryCount() {
    const lore = this.gameState.lore || {};
    const all = Object.values(lore);
    const discovered = all.filter(e => e.discovered).length;
    return { discovered, total: all.length };
  }
}
