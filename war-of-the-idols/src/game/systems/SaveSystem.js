const SAVE_KEY = 'woti_save_v1';
const SETTINGS_KEY = 'woti_settings_v1';

export class SaveSystem {
  constructor(gameState) {
    this.gameState = gameState;
    this.lastAutoSave = Date.now();
    this.autoSaveInterval = 30000; // 30 seconds
  }

  save(slot = 'main') {
    try {
      const saveData = {
        version: 1,
        slot,
        timestamp: Date.now(),
        player: { ...this.gameState.player },
        currentZone: this.gameState.currentZone,
        defeatedBosses: [...(this.gameState.defeatedBosses || [])],
        discoveredZones: [...(this.gameState.discoveredZones || [])],
        quests: JSON.parse(JSON.stringify(this.gameState.quests)),
        lore: JSON.parse(JSON.stringify(this.gameState.lore)),
        inventory: JSON.parse(JSON.stringify(this.gameState.inventory)),
        newGamePlus: this.gameState.newGamePlus || 0,
        playTime: this.gameState.playTime || 0,
        trueNameKnown: this.gameState.trueNameKnown || false,
        allLoreCollected: this.gameState.allLoreCollected || false,
        endingChosen: this.gameState.endingChosen || null,
      };
      const key = slot === 'main' ? SAVE_KEY : `${SAVE_KEY}_${slot}`;
      localStorage.setItem(key, JSON.stringify(saveData));
      return true;
    } catch (e) {
      console.error('Save failed:', e);
      return false;
    }
  }

  load(slot = 'main') {
    try {
      const key = slot === 'main' ? SAVE_KEY : `${SAVE_KEY}_${slot}`;
      const raw = localStorage.getItem(key);
      if (!raw) return null;
      const data = JSON.parse(raw);
      if (!data || data.version !== 1) return null;
      return data;
    } catch (e) {
      console.error('Load failed:', e);
      return null;
    }
  }

  hasSave(slot = 'main') {
    const key = slot === 'main' ? SAVE_KEY : `${SAVE_KEY}_${slot}`;
    return localStorage.getItem(key) !== null;
  }

  deleteSave(slot = 'main') {
    const key = slot === 'main' ? SAVE_KEY : `${SAVE_KEY}_${slot}`;
    localStorage.removeItem(key);
  }

  autoSave() {
    const now = Date.now();
    if (now - this.lastAutoSave >= this.autoSaveInterval) {
      this.save('main');
      this.lastAutoSave = now;
    }
  }

  saveOnEvent(event) {
    // Autosave triggers
    const triggers = ['shrine_rest', 'zone_transition', 'quest_complete', 'item_pickup', 'boss_defeat'];
    if (triggers.includes(event)) {
      this.save('main');
      this.lastAutoSave = Date.now();
    }
  }

  saveSettings(settings) {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
    } catch (e) {
      console.error('Settings save failed:', e);
    }
  }

  loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch (e) {
      return null;
    }
  }

  getSaveInfo(slot = 'main') {
    const data = this.load(slot);
    if (!data) return null;
    return {
      timestamp: data.timestamp,
      zone: data.currentZone,
      level: data.player?.level || 1,
      playTime: data.playTime || 0,
      newGamePlus: data.newGamePlus || 0,
    };
  }
}
