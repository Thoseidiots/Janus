export class NPC {
  constructor(data) {
    this.id = data.id;
    this.name = data.name;
    this.zone = data.zone;
    this.position = { ...data.position };
    this.color = data.color;
    this.emissive = data.emissive;
    this.dialogueTrees = data.dialogueTrees;
    this.isMerchant = data.isMerchant || false;
    this.inventory = data.inventory || [];
    this.mesh = null;
    this.interactable = true;
    this.facingDirection = { x: 0, y: 0, z: 1 };
    this.bobTimer = 0;
  }

  update(deltaTime) {
    // Gentle floating animation
    this.bobTimer += deltaTime;
    if (this.mesh) {
      this.mesh.position.y = this.position.y + Math.sin(this.bobTimer * 1.5) * 0.1;
    }
  }

  getInteractPrompt() {
    return `[E] Talk to ${this.name}`;
  }
}
