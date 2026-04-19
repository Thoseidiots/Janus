// AudioManager uses Web Audio API for procedural sound generation
export class AudioManager {
  constructor() {
    this.context = null;
    this.masterGain = null;
    this.musicGain = null;
    this.sfxGain = null;
    this.enabled = true;
    this.musicEnabled = true;
    this.currentAmbient = null;
    this.ambientNodes = [];
    this._init();
  }

  _init() {
    try {
      this.context = new (window.AudioContext || window.webkitAudioContext)();
      this.masterGain = this.context.createGain();
      this.masterGain.gain.value = 0.5;
      this.masterGain.connect(this.context.destination);

      this.musicGain = this.context.createGain();
      this.musicGain.gain.value = 0.3;
      this.musicGain.connect(this.masterGain);

      this.sfxGain = this.context.createGain();
      this.sfxGain.gain.value = 0.7;
      this.sfxGain.connect(this.masterGain);
    } catch (e) {
      console.warn('Audio not available:', e);
      this.enabled = false;
    }
  }

  resume() {
    if (this.context && this.context.state === 'suspended') {
      this.context.resume();
    }
  }

  playZoneAmbient(zoneId) {
    if (!this.enabled || !this.context) return;
    this.stopAmbient();

    const configs = {
      ashfeld: { baseFreq: 55, noiseColor: 'red', pulseRate: 0.3 },
      frozenMaw: { baseFreq: 40, noiseColor: 'blue', pulseRate: 0.1 },
      ruinsOfAethon: { baseFreq: 65, noiseColor: 'purple', pulseRate: 0.2 },
      crystallineNebula: { baseFreq: 80, noiseColor: 'cyan', pulseRate: 0.4 },
      sovereignsVeil: { baseFreq: 35, noiseColor: 'violet', pulseRate: 0.15 },
      voidCore: { baseFreq: 25, noiseColor: 'black', pulseRate: 0.05 },
    };

    const config = configs[zoneId] || configs.ashfeld;
    this._createAmbientDrone(config);
  }

  _createAmbientDrone(config) {
    if (!this.context) return;

    // Create noise buffer
    const bufferSize = this.context.sampleRate * 2;
    const buffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
    const data = buffer.getChannelData(0);
    for (let i = 0; i < bufferSize; i++) {
      data[i] = (Math.random() * 2 - 1) * 0.1;
    }

    const noiseSource = this.context.createBufferSource();
    noiseSource.buffer = buffer;
    noiseSource.loop = true;

    const filter = this.context.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.value = 200;

    const droneOsc = this.context.createOscillator();
    droneOsc.type = 'sine';
    droneOsc.frequency.value = config.baseFreq;

    const droneGain = this.context.createGain();
    droneGain.gain.value = 0.15;

    noiseSource.connect(filter);
    filter.connect(this.musicGain);
    droneOsc.connect(droneGain);
    droneGain.connect(this.musicGain);

    noiseSource.start();
    droneOsc.start();

    this.ambientNodes = [noiseSource, droneOsc];
    this.currentAmbient = { noiseSource, droneOsc, filter, droneGain };
  }

  stopAmbient() {
    if (this.currentAmbient) {
      try {
        this.currentAmbient.noiseSource.stop();
        this.currentAmbient.droneOsc.stop();
      } catch (e) {}
      this.currentAmbient = null;
    }
  }

  playSFX(type) {
    if (!this.enabled || !this.context) return;
    this.resume();

    const sfxFunctions = {
      attack: () => this._playAttackSound(),
      hit: () => this._playHitSound(),
      death: () => this._playDeathSound(),
      dodge: () => this._playDodgeSound(),
      parry: () => this._playParrySound(),
      shrine: () => this._playShrineSound(),
      levelUp: () => this._playLevelUpSound(),
      itemPickup: () => this._playItemPickupSound(),
      bossRoar: () => this._playBossRoarSound(),
      voidEffect: () => this._playVoidEffectSound(),
    };

    const fn = sfxFunctions[type];
    if (fn) fn();
  }

  _playAttackSound() {
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.connect(gain);
    gain.connect(this.sfxGain);
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(200, this.context.currentTime);
    osc.frequency.exponentialRampToValueAtTime(80, this.context.currentTime + 0.1);
    gain.gain.setValueAtTime(0.3, this.context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + 0.15);
    osc.start();
    osc.stop(this.context.currentTime + 0.15);
  }

  _playHitSound() {
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.connect(gain);
    gain.connect(this.sfxGain);
    osc.type = 'square';
    osc.frequency.setValueAtTime(150, this.context.currentTime);
    osc.frequency.exponentialRampToValueAtTime(50, this.context.currentTime + 0.2);
    gain.gain.setValueAtTime(0.4, this.context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + 0.2);
    osc.start();
    osc.stop(this.context.currentTime + 0.2);
  }

  _playDeathSound() {
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.connect(gain);
    gain.connect(this.sfxGain);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(300, this.context.currentTime);
    osc.frequency.exponentialRampToValueAtTime(30, this.context.currentTime + 1.5);
    gain.gain.setValueAtTime(0.5, this.context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + 1.5);
    osc.start();
    osc.stop(this.context.currentTime + 1.5);
  }

  _playDodgeSound() {
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.connect(gain);
    gain.connect(this.sfxGain);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(400, this.context.currentTime);
    osc.frequency.exponentialRampToValueAtTime(200, this.context.currentTime + 0.1);
    gain.gain.setValueAtTime(0.2, this.context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + 0.1);
    osc.start();
    osc.stop(this.context.currentTime + 0.1);
  }

  _playParrySound() {
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.connect(gain);
    gain.connect(this.sfxGain);
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(800, this.context.currentTime);
    osc.frequency.exponentialRampToValueAtTime(400, this.context.currentTime + 0.3);
    gain.gain.setValueAtTime(0.5, this.context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + 0.3);
    osc.start();
    osc.stop(this.context.currentTime + 0.3);
  }

  _playShrineSound() {
    for (let i = 0; i < 3; i++) {
      const osc = this.context.createOscillator();
      const gain = this.context.createGain();
      osc.connect(gain);
      gain.connect(this.sfxGain);
      osc.type = 'sine';
      const freq = [440, 550, 660][i];
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0, this.context.currentTime + i * 0.2);
      gain.gain.linearRampToValueAtTime(0.3, this.context.currentTime + i * 0.2 + 0.1);
      gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + i * 0.2 + 0.8);
      osc.start(this.context.currentTime + i * 0.2);
      osc.stop(this.context.currentTime + i * 0.2 + 0.8);
    }
  }

  _playLevelUpSound() {
    const freqs = [261, 329, 392, 523];
    freqs.forEach((freq, i) => {
      const osc = this.context.createOscillator();
      const gain = this.context.createGain();
      osc.connect(gain);
      gain.connect(this.sfxGain);
      osc.type = 'sine';
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0, this.context.currentTime + i * 0.15);
      gain.gain.linearRampToValueAtTime(0.4, this.context.currentTime + i * 0.15 + 0.05);
      gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + i * 0.15 + 0.5);
      osc.start(this.context.currentTime + i * 0.15);
      osc.stop(this.context.currentTime + i * 0.15 + 0.5);
    });
  }

  _playItemPickupSound() {
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.connect(gain);
    gain.connect(this.sfxGain);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(600, this.context.currentTime);
    osc.frequency.exponentialRampToValueAtTime(900, this.context.currentTime + 0.15);
    gain.gain.setValueAtTime(0.3, this.context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + 0.2);
    osc.start();
    osc.stop(this.context.currentTime + 0.2);
  }

  _playBossRoarSound() {
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.connect(gain);
    gain.connect(this.sfxGain);
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(80, this.context.currentTime);
    osc.frequency.exponentialRampToValueAtTime(30, this.context.currentTime + 1.0);
    gain.gain.setValueAtTime(0.6, this.context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + 1.0);
    osc.start();
    osc.stop(this.context.currentTime + 1.0);
  }

  _playVoidEffectSound() {
    const osc = this.context.createOscillator();
    const gain = this.context.createGain();
    osc.connect(gain);
    gain.connect(this.sfxGain);
    osc.type = 'sine';
    osc.frequency.setValueAtTime(200, this.context.currentTime);
    osc.frequency.setValueAtTime(50, this.context.currentTime + 0.3);
    osc.frequency.setValueAtTime(200, this.context.currentTime + 0.6);
    gain.gain.setValueAtTime(0.4, this.context.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, this.context.currentTime + 0.8);
    osc.start();
    osc.stop(this.context.currentTime + 0.8);
  }

  setMasterVolume(vol) {
    if (this.masterGain) this.masterGain.gain.value = Math.max(0, Math.min(1, vol));
  }

  setMusicVolume(vol) {
    if (this.musicGain) this.musicGain.gain.value = Math.max(0, Math.min(1, vol));
  }

  setSFXVolume(vol) {
    if (this.sfxGain) this.sfxGain.gain.value = Math.max(0, Math.min(1, vol));
  }

  destroy() {
    this.stopAmbient();
    if (this.context) {
      this.context.close();
    }
  }
}
