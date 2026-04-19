import * as THREE from 'three';

export class ZoneRenderer {
  constructor(scene) {
    this.scene = scene;
    this.zoneObjects = [];
    this.lights = [];
    this.fog = null;
    this.ground = null;
    this.shrineObject = null;
    this.decorObjects = [];
    this.portalObjects = [];
  }

  loadZone(zoneData) {
    this.clearZone();
    if (!zoneData) return;

    this._setupFog(zoneData);
    this._setupLighting(zoneData);
    this._createGround(zoneData);
    this._createSkybox(zoneData);
    this._createDecoration(zoneData);
    this._createShrine(zoneData);
    this._createPortals(zoneData);
    this._createBoundaryWalls(zoneData);
  }

  _setupFog(zoneData) {
    this.scene.fog = new THREE.FogExp2(zoneData.fogColor, 0.025);
    this.scene.background = new THREE.Color(zoneData.skyColor);
  }

  _setupLighting(zoneData) {
    // Ambient light
    const ambient = new THREE.AmbientLight(zoneData.ambientColor, zoneData.ambientIntensity);
    this.scene.add(ambient);
    this.lights.push(ambient);

    // Directional light (sun/moon)
    const dirLight = new THREE.DirectionalLight(zoneData.directionalColor, zoneData.directionalIntensity);
    dirLight.position.set(10, 20, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 100;
    dirLight.shadow.camera.left = -50;
    dirLight.shadow.camera.right = 50;
    dirLight.shadow.camera.top = 50;
    dirLight.shadow.camera.bottom = -50;
    this.scene.add(dirLight);
    this.lights.push(dirLight);

    // Zone-specific point lights for atmosphere
    const pointColors = {
      ashfeld: [0xff2200, 0xff4400],
      frozenMaw: [0x0044ff, 0x0088ff],
      ruinsOfAethon: [0x6600ff, 0x8844ff],
      crystallineNebula: [0x00ffcc, 0x00ddaa],
      sovereignsVeil: [0xcc00ff, 0xff00ff],
      voidCore: [0xff00ff, 0x8800ff],
    };

    const colors = pointColors[zoneData.id] || [0xffffff, 0xaaaaaa];
    colors.forEach((color, i) => {
      const pointLight = new THREE.PointLight(color, 1.5, 30);
      pointLight.position.set(
        (i === 0 ? 1 : -1) * 15,
        5,
        (i === 0 ? -1 : 1) * 15
      );
      this.scene.add(pointLight);
      this.lights.push(pointLight);
    });
  }

  _createGround(zoneData) {
    const geo = new THREE.PlaneGeometry(zoneData.size.width, zoneData.size.depth, 20, 20);

    // Displace vertices for terrain variation
    const positions = geo.attributes.position;
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const z = positions.getZ(i);
      // Keep center flat for gameplay
      const distFromCenter = Math.sqrt(x * x + z * z);
      if (distFromCenter > 10) {
        positions.setY(i, (Math.random() - 0.5) * 0.5);
      }
    }
    geo.computeVertexNormals();

    const mat = new THREE.MeshLambertMaterial({
      color: zoneData.groundColor,
      side: THREE.FrontSide,
    });

    this.ground = new THREE.Mesh(geo, mat);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);
    this.zoneObjects.push(this.ground);

    // Add ground details based on zone
    this._addGroundDetails(zoneData);
  }

  _addGroundDetails(zoneData) {
    const detailConfigs = {
      ashfeld: { color: 0x3d1a00, emissive: 0xff2200, count: 20, type: 'ember' },
      frozenMaw: { color: 0x001a33, emissive: 0x0044ff, count: 15, type: 'ice' },
      ruinsOfAethon: { color: 0x0d0d1a, emissive: 0x6600ff, count: 18, type: 'ruin' },
      crystallineNebula: { color: 0x001515, emissive: 0x00ffcc, count: 25, type: 'crystal' },
      sovereignsVeil: { color: 0x0d0020, emissive: 0xcc00ff, count: 12, type: 'void' },
      voidCore: { color: 0x050005, emissive: 0xff00ff, count: 8, type: 'core' },
    };

    const config = detailConfigs[zoneData.id];
    if (!config) return;

    const halfW = zoneData.size.width / 2 - 3;
    const halfD = zoneData.size.depth / 2 - 3;

    for (let i = 0; i < config.count; i++) {
      let geo;
      if (config.type === 'crystal' || config.type === 'ice') {
        geo = new THREE.ConeGeometry(0.3 + Math.random() * 0.5, 1 + Math.random() * 2, 5);
      } else if (config.type === 'ruin') {
        geo = new THREE.BoxGeometry(
          0.5 + Math.random() * 1.5,
          0.5 + Math.random() * 2,
          0.5 + Math.random() * 1.5
        );
      } else {
        geo = new THREE.SphereGeometry(0.2 + Math.random() * 0.4, 6, 6);
      }

      const mat = new THREE.MeshStandardMaterial({
        color: config.color,
        emissive: config.emissive,
        emissiveIntensity: 0.3 + Math.random() * 0.4,
        roughness: 0.8,
      });

      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(
        (Math.random() * 2 - 1) * halfW,
        0.3,
        (Math.random() * 2 - 1) * halfD
      );
      mesh.rotation.y = Math.random() * Math.PI * 2;
      mesh.castShadow = true;
      this.scene.add(mesh);
      this.decorObjects.push(mesh);
      this.zoneObjects.push(mesh);
    }
  }

  _createSkybox(zoneData) {
    // Create a large sphere for sky atmosphere
    const geo = new THREE.SphereGeometry(150, 16, 16);
    const mat = new THREE.MeshBasicMaterial({
      color: zoneData.skyColor,
      side: THREE.BackSide,
    });
    const sky = new THREE.Mesh(geo, mat);
    this.scene.add(sky);
    this.zoneObjects.push(sky);

    // Add floating particles for atmosphere
    this._createAtmosphericParticles(zoneData);
  }

  _createAtmosphericParticles(zoneData) {
    const particleConfigs = {
      ashfeld: { color: 0xff4400, count: 200, size: 0.15 },
      frozenMaw: { color: 0x88ccff, count: 150, size: 0.1 },
      ruinsOfAethon: { color: 0x8866ff, count: 100, size: 0.12 },
      crystallineNebula: { color: 0x00ffcc, count: 300, size: 0.08 },
      sovereignsVeil: { color: 0xff00ff, count: 120, size: 0.14 },
      voidCore: { color: 0xaa00ff, count: 80, size: 0.2 },
    };

    const config = particleConfigs[zoneData.id];
    if (!config) return;

    const geo = new THREE.BufferGeometry();
    const positions = new Float32Array(config.count * 3);
    const halfW = zoneData.size.width / 2;
    const halfD = zoneData.size.depth / 2;

    for (let i = 0; i < config.count; i++) {
      positions[i * 3] = (Math.random() * 2 - 1) * halfW;
      positions[i * 3 + 1] = Math.random() * 15;
      positions[i * 3 + 2] = (Math.random() * 2 - 1) * halfD;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      color: config.color,
      size: config.size,
      transparent: true,
      opacity: 0.6,
      sizeAttenuation: true,
    });

    const particles = new THREE.Points(geo, mat);
    particles.userData.isParticles = true;
    particles.userData.originalPositions = positions.slice();
    this.scene.add(particles);
    this.zoneObjects.push(particles);
  }

  _createShrine(zoneData) {
    const shrineGroup = new THREE.Group();

    // Base
    const baseGeo = new THREE.CylinderGeometry(1.5, 2, 0.5, 8);
    const baseMat = new THREE.MeshStandardMaterial({
      color: 0x1a1a2e,
      roughness: 0.6,
      metalness: 0.4,
    });
    const base = new THREE.Mesh(baseGeo, baseMat);
    base.castShadow = true;
    shrineGroup.add(base);

    // Pillar
    const pillarGeo = new THREE.CylinderGeometry(0.3, 0.4, 3, 8);
    const pillarMat = new THREE.MeshStandardMaterial({
      color: 0x2a2a4e,
      roughness: 0.5,
      metalness: 0.6,
    });
    const pillar = new THREE.Mesh(pillarGeo, pillarMat);
    pillar.position.y = 1.75;
    shrineGroup.add(pillar);

    // Flame/orb at top
    const orbGeo = new THREE.SphereGeometry(0.5, 12, 12);
    const orbMat = new THREE.MeshStandardMaterial({
      color: 0x4400ff,
      emissive: 0x6600ff,
      emissiveIntensity: 1.5,
      transparent: true,
      opacity: 0.9,
    });
    const orb = new THREE.Mesh(orbGeo, orbMat);
    orb.position.y = 3.5;
    orb.userData.isOrb = true;
    shrineGroup.add(orb);

    // Shrine light
    const shrineLight = new THREE.PointLight(0x6600ff, 2, 10);
    shrineLight.position.y = 3.5;
    shrineGroup.add(shrineLight);

    // Position shrine
    const pos = zoneData.shrinePosition || { x: 0, y: 0, z: 0 };
    shrineGroup.position.set(pos.x, pos.y, pos.z);
    shrineGroup.userData.isShrine = true;
    shrineGroup.userData.zoneId = zoneData.id;

    this.scene.add(shrineGroup);
    this.shrineObject = shrineGroup;
    this.zoneObjects.push(shrineGroup);
  }

  _createPortals(zoneData) {
    const createPortal = (position, color, label) => {
      const group = new THREE.Group();

      // Portal frame
      const frameGeo = new THREE.TorusGeometry(2, 0.3, 8, 16);
      const frameMat = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.8,
        metalness: 0.8,
      });
      const frame = new THREE.Mesh(frameGeo, frameMat);
      group.add(frame);

      // Portal surface
      const surfaceGeo = new THREE.CircleGeometry(1.8, 16);
      const surfaceMat = new THREE.MeshStandardMaterial({
        color: color,
        emissive: color,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide,
      });
      const surface = new THREE.Mesh(surfaceGeo, surfaceMat);
      group.add(surface);

      // Portal light
      const light = new THREE.PointLight(color, 1.5, 8);
      group.add(light);

      group.position.copy(position);
      group.rotation.y = Math.PI / 2;
      group.userData.isPortal = true;
      group.userData.label = label;

      this.scene.add(group);
      this.portalObjects.push(group);
      this.zoneObjects.push(group);
    };

    const halfD = zoneData.size.depth / 2 - 2;

    // Next zone portal (forward)
    if (zoneData.nextZone) {
      createPortal(
        new THREE.Vector3(0, 1.5, -halfD),
        0x6600ff,
        `Enter: ${zoneData.nextZone}`
      );
    }

    // Previous zone portal (back)
    if (zoneData.prevZone) {
      createPortal(
        new THREE.Vector3(0, 1.5, halfD),
        0x004466,
        `Return: ${zoneData.prevZone}`
      );
    }
  }

  _createBoundaryWalls(zoneData) {
    const halfW = zoneData.size.width / 2;
    const halfD = zoneData.size.depth / 2;
    const wallHeight = 8;
    const wallThickness = 2;

    const wallMat = new THREE.MeshLambertMaterial({
      color: zoneData.groundColor,
      transparent: true,
      opacity: 0.7,
    });

    const walls = [
      { pos: [0, wallHeight / 2, -halfD], size: [zoneData.size.width, wallHeight, wallThickness] },
      { pos: [0, wallHeight / 2, halfD], size: [zoneData.size.width, wallHeight, wallThickness] },
      { pos: [-halfW, wallHeight / 2, 0], size: [wallThickness, wallHeight, zoneData.size.depth] },
      { pos: [halfW, wallHeight / 2, 0], size: [wallThickness, wallHeight, zoneData.size.depth] },
    ];

    walls.forEach(w => {
      const geo = new THREE.BoxGeometry(...w.size);
      const mesh = new THREE.Mesh(geo, wallMat);
      mesh.position.set(...w.pos);
      this.scene.add(mesh);
      this.zoneObjects.push(mesh);
    });
  }

  updateParticles(deltaTime) {
    this.zoneObjects.forEach(obj => {
      if (obj.userData.isParticles) {
        const positions = obj.geometry.attributes.position;
        const orig = obj.userData.originalPositions;
        for (let i = 0; i < positions.count; i++) {
          positions.setY(i, positions.getY(i) + deltaTime * (0.5 + Math.random() * 0.5));
          if (positions.getY(i) > 15) {
            positions.setY(i, 0);
            positions.setX(i, orig[i * 3]);
            positions.setZ(i, orig[i * 3 + 2]);
          }
        }
        positions.needsUpdate = true;
      }

      // Animate shrine orb
      if (obj.userData.isShrine) {
        obj.children.forEach(child => {
          if (child.userData.isOrb) {
            child.position.y = 3.5 + Math.sin(Date.now() * 0.002) * 0.2;
            child.rotation.y += deltaTime * 0.5;
          }
        });
      }

      // Animate portals
      if (obj.userData.isPortal) {
        obj.rotation.z += deltaTime * 0.3;
      }
    });
  }

  clearZone() {
    this.zoneObjects.forEach(obj => {
      this.scene.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) {
        if (Array.isArray(obj.material)) {
          obj.material.forEach(m => m.dispose());
        } else {
          obj.material.dispose();
        }
      }
    });
    this.zoneObjects = [];
    this.lights = [];
    this.decorObjects = [];
    this.portalObjects = [];
    this.shrineObject = null;
    this.ground = null;
    this.scene.fog = null;
  }
}
