import * as THREE from 'three';

export class EffectsRenderer {
  constructor(scene) {
    this.scene = scene;
    this.particles = [];
    this.hitEffects = [];
    this.voidEffects = [];
    this.bossEffects = [];
    this.damageNumbers = [];
  }

  createHitEffect(position, color = 0xff4400, isCrit = false) {
    const count = isCrit ? 20 : 10;
    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const velocities = [];

    for (let i = 0; i < count; i++) {
      positions[i * 3] = position.x;
      positions[i * 3 + 1] = position.y + 1;
      positions[i * 3 + 2] = position.z;
      velocities.push({
        x: (Math.random() - 0.5) * 4,
        y: Math.random() * 3 + 1,
        z: (Math.random() - 0.5) * 4,
      });
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      color,
      size: isCrit ? 0.2 : 0.12,
      transparent: true,
      opacity: 1.0,
    });

    const points = new THREE.Points(geo, mat);
    points.userData = { velocities, life: 0.5, maxLife: 0.5 };
    this.scene.add(points);
    this.hitEffects.push(points);
  }

  createVoidEffect(position) {
    const geo = new THREE.SphereGeometry(0.5, 8, 8);
    const mat = new THREE.MeshStandardMaterial({
      color: 0x6600ff,
      emissive: 0x6600ff,
      emissiveIntensity: 2.0,
      transparent: true,
      opacity: 0.8,
    });
    const sphere = new THREE.Mesh(geo, mat);
    sphere.position.set(position.x, position.y + 1, position.z);
    sphere.userData = { life: 0.8, maxLife: 0.8, expanding: true };
    this.scene.add(sphere);
    this.voidEffects.push(sphere);
  }

  createBossPhaseEffect(position, color) {
    // Ring explosion effect
    const geo = new THREE.TorusGeometry(0.5, 0.1, 8, 16);
    const mat = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 2.0,
      transparent: true,
      opacity: 1.0,
    });
    const ring = new THREE.Mesh(geo, mat);
    ring.position.set(position.x, position.y + 1, position.z);
    ring.userData = { life: 1.0, maxLife: 1.0, expandSpeed: 8 };
    this.scene.add(ring);
    this.bossEffects.push(ring);
  }

  createShrineActivationEffect(position) {
    for (let i = 0; i < 5; i++) {
      setTimeout(() => {
        const geo = new THREE.SphereGeometry(0.15, 6, 6);
        const mat = new THREE.MeshStandardMaterial({
          color: 0x6600ff,
          emissive: 0x6600ff,
          emissiveIntensity: 2.0,
          transparent: true,
          opacity: 1.0,
        });
        const orb = new THREE.Mesh(geo, mat);
        orb.position.set(
          position.x + (Math.random() - 0.5) * 2,
          position.y,
          position.z + (Math.random() - 0.5) * 2
        );
        orb.userData = {
          life: 1.5,
          maxLife: 1.5,
          velocity: { x: (Math.random() - 0.5) * 2, y: 3 + Math.random() * 2, z: (Math.random() - 0.5) * 2 },
        };
        this.scene.add(orb);
        this.particles.push(orb);
      }, i * 100);
    }
  }

  createDeathEffect(position, color) {
    for (let i = 0; i < 15; i++) {
      const geo = new THREE.BoxGeometry(0.1, 0.1, 0.1);
      const mat = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 1.0,
        transparent: true,
        opacity: 1.0,
      });
      const piece = new THREE.Mesh(geo, mat);
      piece.position.set(position.x, position.y + 1, position.z);
      piece.userData = {
        life: 1.0,
        maxLife: 1.0,
        velocity: {
          x: (Math.random() - 0.5) * 5,
          y: Math.random() * 4 + 1,
          z: (Math.random() - 0.5) * 5,
        },
        rotVel: {
          x: (Math.random() - 0.5) * 5,
          y: (Math.random() - 0.5) * 5,
          z: (Math.random() - 0.5) * 5,
        },
      };
      this.scene.add(piece);
      this.particles.push(piece);
    }
  }

  createParryEffect(position) {
    const geo = new THREE.RingGeometry(0.3, 0.6, 12);
    const mat = new THREE.MeshBasicMaterial({
      color: 0xffdd00,
      transparent: true,
      opacity: 1.0,
      side: THREE.DoubleSide,
    });
    const ring = new THREE.Mesh(geo, mat);
    ring.position.set(position.x, position.y + 1, position.z);
    ring.rotation.x = -Math.PI / 2;
    ring.userData = { life: 0.4, maxLife: 0.4, expandSpeed: 5 };
    this.scene.add(ring);
    this.hitEffects.push(ring);
  }

  createStatusEffect(position, effectId) {
    const colors = {
      burning: 0xff4400,
      frostbite: 0x4488ff,
      voidCorruption: 0xaa00ff,
      hollowing: 0x333333,
      enlightened: 0xffdd00,
    };
    const color = colors[effectId] || 0xffffff;
    this.createHitEffect(position, color, false);
  }

  update(deltaTime) {
    // Update hit effects
    this.hitEffects = this.hitEffects.filter(effect => {
      effect.userData.life -= deltaTime;
      const t = effect.userData.life / effect.userData.maxLife;

      if (effect.geometry.type === 'BufferGeometry') {
        // Particle system
        const positions = effect.geometry.attributes.position;
        const velocities = effect.userData.velocities;
        for (let i = 0; i < positions.count; i++) {
          positions.setX(i, positions.getX(i) + velocities[i].x * deltaTime);
          positions.setY(i, positions.getY(i) + velocities[i].y * deltaTime);
          positions.setZ(i, positions.getZ(i) + velocities[i].z * deltaTime);
          velocities[i].y -= 9.8 * deltaTime;
        }
        positions.needsUpdate = true;
        effect.material.opacity = t;
      } else {
        // Ring/mesh
        if (effect.userData.expandSpeed) {
          const scale = 1 + (1 - t) * effect.userData.expandSpeed * deltaTime;
          effect.scale.setScalar(effect.scale.x * scale);
        }
        effect.material.opacity = t;
      }

      if (effect.userData.life <= 0) {
        this.scene.remove(effect);
        if (effect.geometry) effect.geometry.dispose();
        if (effect.material) effect.material.dispose();
        return false;
      }
      return true;
    });

    // Update void effects
    this.voidEffects = this.voidEffects.filter(effect => {
      effect.userData.life -= deltaTime;
      const t = effect.userData.life / effect.userData.maxLife;
      if (effect.userData.expanding) {
        effect.scale.setScalar(1 + (1 - t) * 3);
      }
      effect.material.opacity = t * 0.8;
      if (effect.userData.life <= 0) {
        this.scene.remove(effect);
        effect.geometry.dispose();
        effect.material.dispose();
        return false;
      }
      return true;
    });

    // Update boss effects
    this.bossEffects = this.bossEffects.filter(effect => {
      effect.userData.life -= deltaTime;
      const t = effect.userData.life / effect.userData.maxLife;
      effect.scale.setScalar(1 + (1 - t) * effect.userData.expandSpeed);
      effect.material.opacity = t;
      if (effect.userData.life <= 0) {
        this.scene.remove(effect);
        effect.geometry.dispose();
        effect.material.dispose();
        return false;
      }
      return true;
    });

    // Update general particles
    this.particles = this.particles.filter(p => {
      p.userData.life -= deltaTime;
      const t = p.userData.life / p.userData.maxLife;
      if (p.userData.velocity) {
        p.position.x += p.userData.velocity.x * deltaTime;
        p.position.y += p.userData.velocity.y * deltaTime;
        p.position.z += p.userData.velocity.z * deltaTime;
        p.userData.velocity.y -= 9.8 * deltaTime;
      }
      if (p.userData.rotVel) {
        p.rotation.x += p.userData.rotVel.x * deltaTime;
        p.rotation.y += p.userData.rotVel.y * deltaTime;
        p.rotation.z += p.userData.rotVel.z * deltaTime;
      }
      p.material.opacity = t;
      if (p.userData.life <= 0) {
        this.scene.remove(p);
        p.geometry.dispose();
        p.material.dispose();
        return false;
      }
      return true;
    });
  }

  clearAll() {
    [...this.hitEffects, ...this.voidEffects, ...this.bossEffects, ...this.particles].forEach(obj => {
      this.scene.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    });
    this.hitEffects = [];
    this.voidEffects = [];
    this.bossEffects = [];
    this.particles = [];
  }
}
