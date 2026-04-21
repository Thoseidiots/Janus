import * as THREE from 'three';

export class SceneRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = null;
    this.scene = null;
    this.camera = null;
    this.clock = new THREE.Clock();
    this._init();
  }

  _init() {
    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
    });

    // Force correct size — canvas may not have layout dimensions yet
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h, false); // false = don't set canvas style
    this.canvas.width = w * Math.min(window.devicePixelRatio, 2);
    this.canvas.height = h * Math.min(window.devicePixelRatio, 2);
    this.canvas.style.width = w + 'px';
    this.canvas.style.height = h + 'px';

    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.8;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    // Scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);

    // Camera - third person
    this.camera = new THREE.PerspectiveCamera(65, w / h, 0.1, 200);
    this.camera.position.set(0, 8, 12);
    this.camera.lookAt(0, 0, 0);

    // Handle resize
    window.addEventListener('resize', this._onResize.bind(this));
  }

  _onResize() {
    this.camera.aspect = window.innerWidth / window.innerHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(window.innerWidth, window.innerHeight);
  }

  updateCamera(playerPosition, playerRotation) {
    if (!playerPosition) return;

    const cameraDistance = 12;
    const cameraHeight = 8;
    const cameraAngle = playerRotation || 0;

    // Smooth camera follow
    const targetX = playerPosition.x - Math.sin(cameraAngle) * cameraDistance * 0.3;
    const targetZ = playerPosition.z + cameraDistance;
    const targetY = playerPosition.y + cameraHeight;

    this.camera.position.lerp(
      new THREE.Vector3(playerPosition.x, targetY, playerPosition.z + cameraDistance),
      0.08
    );
    this.camera.lookAt(playerPosition.x, playerPosition.y + 1, playerPosition.z);
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  getDeltaTime() {
    return this.clock.getDelta();
  }

  getScene() {
    return this.scene;
  }

  getCamera() {
    return this.camera;
  }

  getRenderer() {
    return this.renderer;
  }

  dispose() {
    window.removeEventListener('resize', this._onResize.bind(this));
    this.renderer.dispose();
  }
}
