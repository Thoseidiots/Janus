import * as THREE from 'three';
import { ENEMY_STATS } from '../data/enemies.js';

export class EnemyRenderer {
  constructor(scene) {
    this.scene = scene;
    this.enemyMeshes = new Map(); // enemyId -> mesh group
    this.healthBars = new Map();
    this.bossHealthBar = null;
  }

  createEnemyMesh(enemy) {
    const stats = enemy.stats || ENEMY_STATS[enemy.id] || {};
    const group = new THREE.Group();

    const size = stats.size || { width: 0.8, height: 1.8, depth: 0.6 };
    const color = stats.color || 0x333333;
    const emissiveColor = stats.emissiveColor || 0x660000;
    const emissiveIntensity = stats.emissiveIntensity || 0.4;

    const mat = new THREE.MeshStandardMaterial({
      color,
      emissive: emissiveColor,
      emissiveIntensity,
      roughness: 0.7,
      metalness: 0.3,
    });

    if (stats.isBoss) {
      this._createBossMesh(group, size, mat, stats);
    } else {
      this._createRegularMesh(group, size, mat, stats);
    }

    // Emissive glow light
    const glowLight = new THREE.PointLight(emissiveColor, 0.8, 4);
    glowLight.position.y = size.height / 2;
    group.add(glowLight);
    group.userData.glowLight = glowLight;

    group.position.set(enemy.position.x, enemy.position.y, enemy.position.z);
    group.userData.enemyId = enemy.id;
    group.userData.isBoss = stats.isBoss || false;

    this.scene.add(group);
    this.enemyMeshes.set(enemy.id, group);
    enemy.mesh = group;

    return group;
  }

  _createRegularMesh(group, size, mat, stats) {
    // Body
    const bodyGeo = new THREE.BoxGeometry(size.width, size.height * 0.6, size.depth);
    const body = new THREE.Mesh(bodyGeo, mat);
    body.position.y = size.height * 0.5;
    body.castShadow = true;
    group.add(body);

    // Head
    const headSize = size.width * 0.7;
    const headGeo = new THREE.BoxGeometry(headSize, headSize, headSize);
    const head = new THREE.Mesh(headGeo, mat);
    head.position.y = size.height * 0.85 + headSize / 2;
    head.castShadow = true;
    group.add(head);

    // Eyes (emissive)
    const eyeMat = new THREE.MeshStandardMaterial({
      color: stats.emissiveColor || 0xff0000,
      emissive: stats.emissiveColor || 0xff0000,
      emissiveIntensity: 2.0,
    });
    const eyeGeo = new THREE.SphereGeometry(0.06, 6, 6);
    [-0.12, 0.12].forEach(xOff => {
      const eye = new THREE.Mesh(eyeGeo, eyeMat);
      eye.position.set(xOff, size.height * 0.85 + headSize / 2, headSize / 2 - 0.05);
      group.add(eye);
    });

    // Legs
    const legGeo = new THREE.BoxGeometry(size.width * 0.3, size.height * 0.35, size.depth * 0.3);
    [-size.width * 0.2, size.width * 0.2].forEach(xOff => {
      const leg = new THREE.Mesh(legGeo, mat);
      leg.position.set(xOff, size.height * 0.175, 0);
      leg.castShadow = true;
      group.add(leg);
    });
  }

  _createBossMesh(group, size, mat, stats) {
    // Larger, more imposing boss mesh
    const bodyGeo = new THREE.BoxGeometry(size.width, size.height * 0.5, size.depth);
    const body = new THREE.Mesh(bodyGeo, mat);
    body.position.y = size.height * 0.5;
    body.castShadow = true;
    group.add(body);

    // Shoulders
    const shoulderGeo = new THREE.BoxGeometry(size.width * 1.4, size.height * 0.2, size.depth * 0.8);
    const shoulders = new THREE.Mesh(shoulderGeo, mat);
    shoulders.position.y = size.height * 0.75;
    group.add(shoulders);

    // Head
    const headGeo = new THREE.BoxGeometry(size.width * 0.8, size.height * 0.25, size.depth * 0.7);
    const head = new THREE.Mesh(headGeo, mat);
    head.position.y = size.height * 0.9;
    group.add(head);

    // Crown/horns
    const hornMat = new THREE.MeshStandardMaterial({
      color: stats.emissiveColor || 0xff0000,
      emissive: stats.emissiveColor || 0xff0000,
      emissiveIntensity: 1.5,
    });
    [-size.width * 0.3, 0, size.width * 0.3].forEach((xOff, i) => {
      const hornGeo = new THREE.ConeGeometry(0.15, 0.6 + i * 0.2, 6);
      const horn = new THREE.Mesh(hornGeo, hornMat);
      horn.position.set(xOff, size.height + 0.3 + i * 0.1, 0);
      group.add(horn);
    });

    // Arms
    const armGeo = new THREE.BoxGeometry(size.width * 0.25, size.height * 0.45, size.depth * 0.25);
    [-size.width * 0.65, size.width * 0.65].forEach(xOff => {
      const arm = new THREE.Mesh(armGeo, mat);
      arm.position.set(xOff, size.height * 0.55, 0);
      arm.castShadow = true;
      group.add(arm);
    });

    // Legs
    const legGeo = new THREE.BoxGeometry(size.width * 0.35, size.height * 0.4, size.depth * 0.35);
    [-size.width * 0.25, size.width * 0.25].forEach(xOff => {
      const leg = new THREE.Mesh(legGeo, mat);
      leg.position.set(xOff, size.height * 0.2, 0);
      leg.castShadow = true;
      group.add(leg);
    });

    // Extra glow for boss
    const extraGlow = new THREE.PointLight(stats.emissiveColor || 0xff0000, 2.0, 8);
    extraGlow.position.y = size.height;
    group.add(extraGlow);
  }

  updateEnemyMesh(enemy) {
    const mesh = this.enemyMeshes.get(enemy.id);
    if (!mesh) return;

    if (enemy.isDead) {
      // Death animation - sink into ground
      mesh.position.y -= 0.02;
      mesh.rotation.z += 0.02;
      if (mesh.position.y < -3) {
        this.removeEnemyMesh(enemy.id);
      }
      return;
    }

    // Update position
    mesh.position.set(enemy.position.x, enemy.position.y, enemy.position.z);

    // Update rotation to face direction
    if (enemy.facingDirection) {
      const angle = Math.atan2(enemy.facingDirection.x, enemy.facingDirection.z);
      mesh.rotation.y = angle;
    }

    // Attack animation
    if (enemy.isAttacking) {
      mesh.position.y = enemy.position.y + Math.sin(Date.now() * 0.02) * 0.1;
    }

    // Stagger animation
    if (enemy.staggerTimer > 0) {
      mesh.rotation.z = Math.sin(Date.now() * 0.05) * 0.2;
    } else {
      mesh.rotation.z = 0;
    }

    // Phase change flash
    if (enemy.phaseChanged) {
      const intensity = Math.sin(Date.now() * 0.01) * 2 + 2;
      mesh.children.forEach(child => {
        if (child.material && child.material.emissiveIntensity !== undefined) {
          child.material.emissiveIntensity = intensity;
        }
      });
    }

    // Pulse glow light
    if (mesh.userData.glowLight) {
      mesh.userData.glowLight.intensity = 0.8 + Math.sin(Date.now() * 0.003) * 0.3;
    }
  }

  removeEnemyMesh(enemyId) {
    const mesh = this.enemyMeshes.get(enemyId);
    if (mesh) {
      this.scene.remove(mesh);
      mesh.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
      this.enemyMeshes.delete(enemyId);
    }
  }

  clearAllEnemies() {
    this.enemyMeshes.forEach((mesh, id) => {
      this.removeEnemyMesh(id);
    });
    this.enemyMeshes.clear();
  }

  getEnemyMesh(enemyId) {
    return this.enemyMeshes.get(enemyId);
  }
}
