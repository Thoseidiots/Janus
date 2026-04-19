import { ZONES, ZONE_ORDER } from '../../data/zones.js';
import { ENEMY_STATS } from '../../data/enemies.js';
import { NPC_DATA } from '../../data/npcs.js';
import { Enemy } from '../entities/Enemy.js';
import { NPC } from '../entities/NPC.js';

// Enemy type mappings per zone
const ZONE_ENEMY_TYPES = {
  ashfeld: ['ashenWraith', 'cinderHound', 'emberKnight', 'slagTitan'],
  frozenMaw: ['frostSpecter', 'iceColossus', 'frozenChoir', 'ysoldesShard'],
  ruinsOfAethon: ['boneConstruct', 'fallenPaladin', 'memorySpecter', 'archivistsRegret'],
  crystallineNebula: ['thoughtLeech', 'crystalHorror', 'theForgotten', 'archivistsConstruct'],
  sovereignsVeil: ['veilTitan', 'realityFracture', 'voidDoppelganger', 'theUnwritten'],
  voidCore: ['voidSovereign'],
};

const ZONE_NPC_TYPES = {
  ashfeld: ['ghostNavigator'],
  ruinsOfAethon: ['echoOfAethonNPC'],
  crystallineNebula: ['archivistKael'],
  sovereignsVeil: ['voidMerchant'],
};

export class ZoneManager {
  constructor(gameState) {
    this.gameState = gameState;
    this.zones = ZONES;
    this.currentZoneId = 'ashfeld';
    this.enemies = [];
    this.npcs = [];
    this.bossEnemy = null;
    this.transitionCooldown = 0;
    this.defeatedBosses = new Set();
  }

  getCurrentZone() {
    return this.zones[this._zoneKeyFromId(this.currentZoneId)];
  }

  _zoneKeyFromId(id) {
    const map = {
      ashfeld: 'ASHFELD',
      frozenMaw: 'FROZEN_MAW',
      ruinsOfAethon: 'RUINS_OF_AETHON',
      crystallineNebula: 'CRYSTALLINE_NEBULA',
      sovereignsVeil: 'SOVEREIGNS_VEIL',
      voidCore: 'VOID_CORE',
    };
    return map[id] || 'ASHFELD';
  }

  loadZone(zoneId) {
    this.currentZoneId = zoneId;
    const zoneKey = this._zoneKeyFromId(zoneId);
    const zoneData = this.zones[zoneKey];

    if (!zoneData) {
      console.error('Zone not found:', zoneId);
      return;
    }

    this.gameState.currentZone = zoneId;
    this.gameState.currentZoneData = zoneData;

    // Spawn enemies
    this.enemies = this._spawnEnemies(zoneId, zoneData);
    this.bossEnemy = this._spawnBoss(zoneId, zoneData);

    // Spawn NPCs
    this.npcs = this._spawnNPCs(zoneId);

    // Add discovered zones
    if (!this.gameState.discoveredZones) this.gameState.discoveredZones = [];
    if (!this.gameState.discoveredZones.includes(zoneId)) {
      this.gameState.discoveredZones.push(zoneId);
    }

    // Discover zone lore
    if (this.gameState.loreSystem) {
      this.gameState.loreSystem.discoverZoneLore(zoneId);
    }

    return { enemies: this.enemies, boss: this.bossEnemy, npcs: this.npcs, zoneData };
  }

  _spawnEnemies(zoneId, zoneData) {
    const enemyTypes = ZONE_ENEMY_TYPES[zoneId] || [];
    if (enemyTypes.length === 0) return [];

    const enemies = [];
    const count = zoneData.enemyDensity || 5;
    const halfW = zoneData.size.width / 2 - 5;
    const halfD = zoneData.size.depth / 2 - 5;

    // Don't spawn boss enemies as regular enemies
    const bossId = zoneData.bossId;

    for (let i = 0; i < count; i++) {
      const typeId = enemyTypes[Math.floor(Math.random() * enemyTypes.length)];
      if (typeId === bossId) continue;

      const stats = ENEMY_STATS[typeId];
      if (!stats) continue;

      const position = {
        x: (Math.random() * 2 - 1) * halfW,
        y: 0,
        z: (Math.random() * 2 - 1) * halfD,
      };

      // Don't spawn too close to origin (player start)
      if (Math.abs(position.x) < 8 && Math.abs(position.z) < 8) {
        position.x += position.x > 0 ? 8 : -8;
      }

      const enemy = new Enemy(stats, position, `${typeId}_${i}`);
      enemies.push(enemy);
    }

    return enemies;
  }

  _spawnBoss(zoneId, zoneData) {
    const bossId = zoneData.bossId;
    if (!bossId) return null;

    // Check if boss already defeated
    if (this.defeatedBosses.has(bossId) ||
        (this.gameState.defeatedBosses && this.gameState.defeatedBosses.includes(bossId))) {
      return null;
    }

    const stats = ENEMY_STATS[bossId];
    if (!stats) return null;

    const halfW = zoneData.size.width / 2 - 10;
    const halfD = zoneData.size.depth / 2 - 10;

    const position = {
      x: (Math.random() > 0.5 ? 1 : -1) * (halfW * 0.6 + Math.random() * halfW * 0.3),
      y: 0,
      z: (Math.random() > 0.5 ? 1 : -1) * (halfD * 0.6 + Math.random() * halfD * 0.3),
    };

    const boss = new Enemy(stats, position, bossId);
    return boss;
  }

  _spawnNPCs(zoneId) {
    const npcTypes = ZONE_NPC_TYPES[zoneId] || [];
    return npcTypes.map(npcId => {
      const data = NPC_DATA[npcId];
      if (!data) return null;
      return new NPC(data);
    }).filter(Boolean);
  }

  respawnEnemies() {
    const zoneData = this.getCurrentZone();
    if (!zoneData) return;
    this.enemies = this._spawnEnemies(this.currentZoneId, zoneData);
    // Don't respawn boss
  }

  canTransitionTo(targetZoneId) {
    const currentOrder = ZONE_ORDER.indexOf(this.currentZoneId);
    const targetOrder = ZONE_ORDER.indexOf(targetZoneId);

    // Can always go back
    if (targetOrder < currentOrder) return true;

    // Can go forward if boss is defeated
    if (targetOrder === currentOrder + 1) {
      const currentZone = this.getCurrentZone();
      const bossId = currentZone?.bossId;
      if (!bossId) return true;
      return this.defeatedBosses.has(bossId) ||
             (this.gameState.defeatedBosses && this.gameState.defeatedBosses.includes(bossId));
    }

    return false;
  }

  onBossDefeated(bossId) {
    this.defeatedBosses.add(bossId);
    if (!this.gameState.defeatedBosses) this.gameState.defeatedBosses = [];
    if (!this.gameState.defeatedBosses.includes(bossId)) {
      this.gameState.defeatedBosses.push(bossId);
    }
    this.bossEnemy = null;
  }

  getTransitionZones(playerPosition, zoneData) {
    if (!zoneData) return null;
    const halfW = zoneData.size.width / 2;
    const halfD = zoneData.size.depth / 2;
    const threshold = 3;

    // Check if player is near zone edge
    if (playerPosition.z < -halfD + threshold && zoneData.prevZone) {
      return zoneData.prevZone;
    }
    if (playerPosition.z > halfD - threshold && zoneData.nextZone) {
      if (this.canTransitionTo(zoneData.nextZone)) {
        return zoneData.nextZone;
      }
    }
    return null;
  }

  getAllEnemies() {
    const all = [...this.enemies];
    if (this.bossEnemy && !this.bossEnemy.isDead) all.push(this.bossEnemy);
    return all;
  }

  getNPCs() {
    return this.npcs;
  }
}
