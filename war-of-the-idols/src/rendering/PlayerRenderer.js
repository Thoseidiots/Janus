import * as THREE from 'three';
import { ITEMS } from '../data/items.js';

export class PlayerRenderer {
  constructor(scene) {
    this.scene = scene;
    this.playerGroup = null;
    this.weaponMesh = null;
    this.dodgeTrail = [];
    this.attackFlash = null;
    this.attackFlashTimer = 0;
  }

  createPlayerMesh() {
    const group = new THREE.Group();

    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x2a2a3e,
      roughness: 0.6,
      metalness: 0.4,
    });

    const accentMat = new THREE.MeshStandardMaterial({
      color: 0x4400aa,
      emissive: 0x2200aa,
      emissiveIntensity: 0.5,
      roughness: 0.4,
      metalness: 0.6,
    });

    // Legs
    const legGeo = new THREE.BoxGeometry(0.25, 0.5, 0.25);
    [-0.18, 0.18].forEach(xOff => {
      const leg = new THREE.Mesh(legGeo, bodyMat);
      leg.position.set(xOff, 0.25, 0);
      leg.castShadow = true;
      group.add(leg);
    });

    // Torso
    const torsoGeo = new THREE.BoxGeometry(0.55, 0.6, 0.3);
    const torso = new THREE.Mesh(torsoGeo, bodyMat);
    torso.position.y = 0.8;
    torso.castShadow = true;
    group.add(torso);

    // Chest accent
    const chestGeo = new THREE.BoxGeometry(0.4, 0.4, 0.05);
    const chest = new THREE.Mesh(chestGeo, accentMat);
    chest.position.set(0, 0.8, 0.18);
    group.add(chest);

    // Arms
    const armGeo = new THREE.BoxGeometry(0.2, 0.5, 0.2);
    [-0.38, 0.38].forEach((xOff, i) => {
      const arm = new THREE.Mesh(armGeo, bodyMat);
      arm.position.set(xOff, 0.75, 0);
      arm.castShadow = true;
      arm.userData.isArm = true;
      arm.userData.side = i === 0 ? 'left' : 'right';
      group.add(arm);
    });

    // Head
    const headGeo = new THREE.BoxGeometry(0.4, 0.4, 0.35);
    const head = new THREE.Mesh(headGeo, bodyMat);
    head.position.y = 1.3;
    head.castShadow = true;
    group.add(head);

    // Helmet visor
    const visorGeo = new THREE.BoxGeometry(0.35, 0.12, 0.05);
    const visor = new THREE.Mesh(visorGeo, accentMat);
    visor.position.set(0, 1.32, 0.2);
    group.add(visor);

    // Player glow
    const playerLight = new THREE.PointLight(0x4400aa, 0.5, 3);
    playerLight.position.y = 1;
    group.add(playerLight);
    group.userData.playerLight = playerLight;

    // Weapon attachment point
    const weaponPoint = new THREE.Object3D();
    weaponPoint.position.set(0.45, 0.75, 0);
    group.add(weaponPoint);
    group.userData.weaponPoint = weaponPoint;

    this.playerGroup = group;
    this.scene.add(group);
    return group;
  }

  updateWeaponMesh(weaponId) {
    // Remove old weapon
    if (this.weaponMesh) {
      const weaponPoint = this.playerGroup?.userData.weaponPoint;
      if (weaponPoint) weaponPoint.remove(this.weaponMesh);
      if (this.weaponMesh.geometry) this.weaponMesh.geometry.dispose();
      if (this.weaponMesh.material) this.weaponMesh.material.dispose();
      this.weaponMesh = null;
    }

    if (!weaponId || !this.playerGroup) return;

    const item = ITEMS[weaponId];
    if (!item) return;

    const weaponMat = new THREE.MeshStandardMaterial({
      color: item.color || 0x888888,
      emissive: item.emissive || 0x000000,
      emissiveIntensity: item.emissive ? 0.6 : 0,
      roughness: 0.3,
      metalness: 0.8,
    });

    let weaponGeo;
    switch (item.meshType) {
      case 'dao':
        weaponGeo = this._createDaoGeometry();
        break;
      case 'axe':
        weaponGeo = this._createAxeGeometry();
        break;
      case 'staff':
        weaponGeo = this._createStaffGeometry();
        break;
      case 'greatsword':
        weaponGeo = this._createGreatswordGeometry();
        break;
      case 'shard':
        weaponGeo = this._createShardGeometry();
        break;
      default:
        weaponGeo = this._createSwordGeometry();
    }

    this.weaponMesh = new THREE.Mesh(weaponGeo, weaponMat);
    const weaponPoint = this.playerGroup.userData.weaponPoint;
    if (weaponPoint) {
      weaponPoint.add(this.weaponMesh);
    }
  }

  _createSwordGeometry() {
    const group = new THREE.Group();
    const blade = new THREE.BoxGeometry(0.06, 0.8, 0.04);
    const handle = new THREE.BoxGeometry(0.08, 0.25, 0.08);
    const guard = new THREE.BoxGeometry(0.3, 0.06, 0.06);
    const bladeMesh = new THREE.Mesh(blade);
    bladeMesh.position.y = 0.5;
    const handleMesh = new THREE.Mesh(handle);
    handleMesh.position.y = 0;
    const guardMesh = new THREE.Mesh(guard);
    guardMesh.position.y = 0.15;
    group.add(bladeMesh, handleMesh, guardMesh);
    // Merge into single geometry
    return new THREE.BoxGeometry(0.08, 1.1, 0.06);
  }

  _createDaoGeometry() {
    return new THREE.BoxGeometry(0.06, 0.9, 0.04);
  }

  _createAxeGeometry() {
    return new THREE.BoxGeometry(0.4, 0.6, 0.06);
  }

  _createStaffGeometry() {
    return new THREE.CylinderGeometry(0.04, 0.04, 1.4, 8);
  }

  _createGreatswordGeometry() {
    return new THREE.BoxGeometry(0.1, 1.4, 0.06);
  }

  _createShardGeometry() {
    return new THREE.OctahedronGeometry(0.3);
  }

  updatePlayerMesh(player, deltaTime) {
    if (!this.playerGroup || !player) return;

    this.playerGroup.position.set(player.position.x, player.position.y, player.position.z);
    this.playerGroup.rotation.y = player.rotation || 0;

    // Walking animation
    if (player.isSprinting || (Math.abs(player.velocity?.x || 0) > 0.1 || Math.abs(player.velocity?.z || 0) > 0.1)) {
      const walkSpeed = player.isSprinting ? 8 : 5;
      const walkAmt = Math.sin(Date.now() * 0.01 * walkSpeed) * 0.15;
      this.playerGroup.children.forEach(child => {
        if (child.userData.isArm) {
          child.rotation.x = child.userData.side === 'left' ? walkAmt : -walkAmt;
        }
      });
    }

    // Attack flash
    if (player.isAttacking) {
      this.attackFlashTimer = 0.2;
    }
    if (this.attackFlashTimer > 0) {
      this.attackFlashTimer -= deltaTime;
      if (this.playerGroup.userData.playerLight) {
        this.playerGroup.userData.playerLight.intensity = 2.0;
        this.playerGroup.userData.playerLight.color.setHex(0xffffff);
      }
    } else {
      if (this.playerGroup.userData.playerLight) {
        this.playerGroup.userData.playerLight.intensity = 0.5;
        this.playerGroup.userData.playerLight.color.setHex(0x4400aa);
      }
    }

    // Dodge trail
    if (player.isDodging) {
      this._addDodgeTrail(player.position);
    }
    this._updateDodgeTrail(deltaTime);
  }

  _addDodgeTrail(position) {
    const geo = new THREE.SphereGeometry(0.2, 6, 6);
    const mat = new THREE.MeshBasicMaterial({
      color: 0x4400aa,
      transparent: true,
      opacity: 0.5,
    });
    const trail = new THREE.Mesh(geo, mat);
    trail.position.set(position.x, position.y + 0.8, position.z);
    trail.userData.life = 0.3;
    this.scene.add(trail);
    this.dodgeTrail.push(trail);
  }

  _updateDodgeTrail(deltaTime) {
    this.dodgeTrail = this.dodgeTrail.filter(trail => {
      trail.userData.life -= deltaTime;
      trail.material.opacity = trail.userData.life / 0.3 * 0.5;
      trail.scale.setScalar(1 - (1 - trail.userData.life / 0.3) * 0.5);
      if (trail.userData.life <= 0) {
        this.scene.remove(trail);
        trail.geometry.dispose();
        trail.material.dispose();
        return false;
      }
      return true;
    });
  }

  dispose() {
    if (this.playerGroup) {
      this.scene.remove(this.playerGroup);
      this.playerGroup.traverse(child => {
        if (child.geometry) child.geometry.dispose();
        if (child.material) child.material.dispose();
      });
    }
  }
}
