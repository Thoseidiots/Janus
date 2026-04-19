// HUDRenderer manages the minimap canvas
export class HUDRenderer {
  constructor() {
    this.minimapCanvas = null;
    this.minimapCtx = null;
    this._initMinimap();
  }

  _initMinimap() {
    this.minimapCanvas = document.createElement('canvas');
    this.minimapCanvas.width = 150;
    this.minimapCanvas.height = 150;
    this.minimapCtx = this.minimapCanvas.getContext('2d');
  }

  renderMinimap(player, enemies, npcs, zoneData, shrinePos) {
    if (!this.minimapCtx || !zoneData) return;

    const ctx = this.minimapCtx;
    const w = this.minimapCanvas.width;
    const h = this.minimapCanvas.height;

    // Clear
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(0, 0, w, h);

    // Border
    ctx.strokeStyle = '#4400aa';
    ctx.lineWidth = 2;
    ctx.strokeRect(1, 1, w - 2, h - 2);

    const halfW = zoneData.size.width / 2;
    const halfD = zoneData.size.depth / 2;

    const toMapX = (worldX) => ((worldX + halfW) / (halfW * 2)) * (w - 10) + 5;
    const toMapY = (worldZ) => ((worldZ + halfD) / (halfD * 2)) * (h - 10) + 5;

    // Draw shrine
    if (shrinePos) {
      ctx.fillStyle = '#6600ff';
      ctx.beginPath();
      ctx.arc(toMapX(shrinePos.x), toMapY(shrinePos.z), 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw enemies
    if (enemies) {
      enemies.forEach(enemy => {
        if (!enemy || enemy.isDead) return;
        ctx.fillStyle = enemy.isBoss ? '#ff0000' : '#ff4400';
        ctx.beginPath();
        ctx.arc(toMapX(enemy.position.x), toMapY(enemy.position.z), enemy.isBoss ? 4 : 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw NPCs
    if (npcs) {
      npcs.forEach(npc => {
        if (!npc) return;
        ctx.fillStyle = '#00ffcc';
        ctx.beginPath();
        ctx.arc(toMapX(npc.position.x), toMapY(npc.position.z), 3, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Draw player
    if (player) {
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(toMapX(player.position.x), toMapY(player.position.z), 4, 0, Math.PI * 2);
      ctx.fill();

      // Player direction indicator
      const angle = player.rotation || 0;
      const px = toMapX(player.position.x);
      const py = toMapY(player.position.z);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + Math.sin(angle) * 8, py + Math.cos(angle) * 8);
      ctx.stroke();
    }

    return this.minimapCanvas;
  }

  getMinimapDataURL() {
    return this.minimapCanvas?.toDataURL();
  }
}
