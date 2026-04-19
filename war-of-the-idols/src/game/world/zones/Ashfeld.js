// Ashfeld zone configuration — data lives in src/data/zones.js
// This file provides zone-specific helper functions

export function getAshfeldSpawnPoints() {
  return [
    { x: 15, z: -15 },
    { x: -15, z: 15 },
    { x: 20, z: 10 },
    { x: -20, z: -10 },
    { x: 10, z: 25 },
  ];
}

export function getAshfeldBossArena() {
  return { x: 25, z: -25, radius: 15 };
}
