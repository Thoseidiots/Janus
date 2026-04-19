export class PhysicsEngine {
  constructor() {
    this.gravity = -20;
    this.groundY = 0;
  }

  update(entities, deltaTime, zoneData) {
    for (const entity of entities) {
      if (!entity || !entity.position) continue;
      this._updateEntity(entity, deltaTime, zoneData);
    }
  }

  _updateEntity(entity, deltaTime, zoneData) {
    // Simple ground clamping
    if (entity.position.y < this.groundY) {
      entity.position.y = this.groundY;
      if (entity.velocity) entity.velocity.y = 0;
      entity.isGrounded = true;
    }

    // Zone boundary clamping
    if (zoneData) {
      const halfW = zoneData.size.width / 2;
      const halfD = zoneData.size.depth / 2;
      entity.position.x = Math.max(-halfW + 1, Math.min(halfW - 1, entity.position.x));
      entity.position.z = Math.max(-halfD + 1, Math.min(halfD - 1, entity.position.z));
    }
  }

  checkCollision(entityA, entityB) {
    if (!entityA || !entityB) return false;
    const dx = entityA.position.x - entityB.position.x;
    const dz = entityA.position.z - entityB.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    const minDist = (entityA.collisionRadius || 0.5) + (entityB.collisionRadius || 0.5);
    return dist < minDist;
  }

  resolveCollision(entityA, entityB) {
    const dx = entityA.position.x - entityB.position.x;
    const dz = entityA.position.z - entityB.position.z;
    const dist = Math.sqrt(dx * dx + dz * dz);
    if (dist === 0) return;

    const minDist = (entityA.collisionRadius || 0.5) + (entityB.collisionRadius || 0.5);
    if (dist < minDist) {
      const overlap = minDist - dist;
      const nx = dx / dist;
      const nz = dz / dist;
      entityA.position.x += nx * overlap * 0.5;
      entityA.position.z += nz * overlap * 0.5;
      entityB.position.x -= nx * overlap * 0.5;
      entityB.position.z -= nz * overlap * 0.5;
    }
  }

  isInRange(entityA, entityB, range) {
    const dx = entityA.position.x - entityB.position.x;
    const dz = entityA.position.z - entityB.position.z;
    return Math.sqrt(dx * dx + dz * dz) <= range;
  }

  getDistance(posA, posB) {
    const dx = posA.x - posB.x;
    const dz = posA.z - posB.z;
    return Math.sqrt(dx * dx + dz * dz);
  }
}
