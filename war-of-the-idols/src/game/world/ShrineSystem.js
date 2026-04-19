export class ShrineSystem {
  constructor(gameState) {
    this.gameState = gameState;
    this.shrines = {};
    this.activatedShrines = new Set();
    this.currentShrine = null;
  }

  registerShrine(zoneId, position) {
    this.shrines[zoneId] = {
      id: `shrine_${zoneId}`,
      zoneId,
      position: { ...position },
      activated: false,
    };
  }

  activateShrine(zoneId) {
    const shrine = this.shrines[zoneId];
    if (!shrine) return false;

    shrine.activated = true;
    this.activatedShrines.add(zoneId);
    this.currentShrine = zoneId;

    // Set respawn point
    this.gameState.respawnZone = zoneId;
    this.gameState.respawnPosition = { ...shrine.position };

    // Restore player
    const player = this.gameState.player;
    if (player) {
      player.hp = player.maxHp;
      player.stamina = player.maxStamina;
    }

    // Respawn enemies (shrine rest respawns enemies)
    this.gameState.pendingEnemyRespawn = true;

    return true;
  }

  isActivated(zoneId) {
    return this.activatedShrines.has(zoneId);
  }

  getRespawnPoint() {
    return {
      zone: this.gameState.respawnZone || 'ashfeld',
      position: this.gameState.respawnPosition || { x: 0, y: 0, z: 0 },
    };
  }

  getNearestShrine(position, zoneId) {
    const shrine = this.shrines[zoneId];
    if (!shrine) return null;
    const dx = shrine.position.x - position.x;
    const dz = shrine.position.z - position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    return { shrine, distance: dist };
  }

  isNearShrine(position, zoneId, range = 3.0) {
    const result = this.getNearestShrine(position, zoneId);
    return result && result.distance <= range;
  }
}
