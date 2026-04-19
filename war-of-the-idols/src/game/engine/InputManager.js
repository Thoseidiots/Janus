export class InputManager {
  constructor() {
    this.keys = {};
    this.justPressed = {};
    this.justReleased = {};
    this._pendingJustPressed = {};

    this._onKeyDown = this._onKeyDown.bind(this);
    this._onKeyUp = this._onKeyUp.bind(this);

    window.addEventListener('keydown', this._onKeyDown);
    window.addEventListener('keyup', this._onKeyUp);
  }

  _onKeyDown(e) {
    const key = e.key.toLowerCase();
    if (!this.keys[key]) {
      this._pendingJustPressed[key] = true;
    }
    this.keys[key] = true;

    // Prevent default for game keys
    const gameKeys = ['w', 'a', 's', 'd', ' ', 'f', 'q', 'e', 'shift', 'tab', 'i'];
    if (gameKeys.includes(key)) {
      e.preventDefault();
    }
  }

  _onKeyUp(e) {
    const key = e.key.toLowerCase();
    this.keys[key] = false;
    this.justReleased[key] = true;
  }

  update() {
    // Move pending just-pressed to justPressed
    this.justPressed = { ...this._pendingJustPressed };
    this._pendingJustPressed = {};
    this.justReleased = {};
  }

  getState() {
    return {
      w: !!this.keys['w'],
      a: !!this.keys['a'],
      s: !!this.keys['s'],
      d: !!this.keys['d'],
      shift: !!this.keys['shift'],
      space: !!this.keys[' '],
      f: !!this.keys['f'],
      q: !!this.keys['q'],
      e: !!this.keys['e'],
      i: !!this.keys['i'],
      tab: !!this.keys['tab'],
      escape: !!this.keys['escape'],

      // Just pressed (single frame)
      spaceJustPressed: !!this.justPressed[' '],
      fJustPressed: !!this.justPressed['f'],
      qJustPressed: !!this.justPressed['q'],
      eJustPressed: !!this.justPressed['e'],
      iJustPressed: !!this.justPressed['i'],
      tabJustPressed: !!this.justPressed['tab'],
      escapeJustPressed: !!this.justPressed['escape'],
    };
  }

  isKeyDown(key) {
    return !!this.keys[key.toLowerCase()];
  }

  wasKeyJustPressed(key) {
    return !!this.justPressed[key.toLowerCase()];
  }

  destroy() {
    window.removeEventListener('keydown', this._onKeyDown);
    window.removeEventListener('keyup', this._onKeyUp);
  }
}
