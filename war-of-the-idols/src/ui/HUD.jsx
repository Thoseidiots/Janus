import React, { useEffect, useRef } from 'react';
import { RARITIES } from '../data/items.js';

function Bar({ value, max, color, bgColor, label, showNumbers = false }) {
  const pct = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="relative">
      {label && <div className="text-xs text-gray-400 mb-0.5">{label}</div>}
      <div className={`h-3 rounded-sm ${bgColor} overflow-hidden`} style={{ width: '100%' }}>
        <div
          className={`h-full ${color} transition-all duration-100`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {showNumbers && (
        <div className="text-xs text-gray-300 mt-0.5">{Math.floor(value)}/{Math.floor(max)}</div>
      )}
    </div>
  );
}

function StatusEffectIcon({ effect }) {
  const colors = {
    burning: 'bg-orange-600',
    frostbite: 'bg-blue-600',
    voidCorruption: 'bg-purple-700',
    hollowing: 'bg-gray-700',
    enlightened: 'bg-yellow-500',
    staggered: 'bg-yellow-600',
    disarmed: 'bg-red-600',
  };
  const icons = {
    burning: '🔥', frostbite: '❄️', voidCorruption: '💜',
    hollowing: '💀', enlightened: '✨', staggered: '⚡', disarmed: '🗡️',
  };
  return (
    <div
      className={`w-7 h-7 rounded flex items-center justify-center text-sm ${colors[effect.id] || 'bg-gray-700'}`}
      title={`${effect.def?.name || effect.id}: ${effect.def?.description || ''}`}
    >
      {icons[effect.id] || '?'}
    </div>
  );
}

export default function HUD({ gameState, engine }) {
  const player = gameState?.player;
  const minimapRef = useRef(null);

  useEffect(() => {
    if (minimapRef.current && gameState?.minimapDataURL) {
      const img = new Image();
      img.onload = () => {
        const ctx = minimapRef.current?.getContext('2d');
        if (ctx) ctx.drawImage(img, 0, 0);
      };
      img.src = gameState.minimapDataURL;
    }
  }, [gameState?.minimapDataURL]);

  if (!player) return null;

  const inv = gameState.inventory || { slots: [], equipped: {} };
  const equippedWeapon = inv.equipped?.weapon;
  const statusEffects = gameState.combatSystem?.getPlayerStatusEffects() || [];
  const hollowingStacks = player.hollowing || 0;
  const xpProgress = gameState.levelSystem?.getXPProgress() || { current: 0, needed: 100, percent: 0 };
  const activeQuests = gameState.questSystem?.getActiveQuests() || [];
  const bossDialogue = gameState.bossDialogue;
  const lastMessage = gameState.lastCombatMessage;
  const nearShrine = gameState.nearShrine;
  const finalErasureCharging = gameState.bossEnemy?.finalErasureCharging;
  const finalErasureTimer = gameState.bossEnemy?.finalErasureTimer;

  return (
    <div className="fixed inset-0 pointer-events-none select-none" style={{ fontFamily: 'Georgia, serif' }}>

      {/* Top-left: Player stats */}
      <div className="absolute top-4 left-4 w-56 space-y-2">
        {/* HP */}
        <div>
          <div className="flex justify-between text-xs mb-0.5">
            <span className="text-red-400">HP</span>
            <span className="text-gray-300">{Math.floor(player.hp)}/{player.maxHp}</span>
          </div>
          <div className="h-4 bg-gray-900 rounded overflow-hidden border border-red-900">
            <div
              className="h-full bg-gradient-to-r from-red-900 to-red-600 transition-all duration-150"
              style={{ width: `${(player.hp / player.maxHp) * 100}%` }}
            />
          </div>
        </div>

        {/* Stamina */}
        <div>
          <div className="flex justify-between text-xs mb-0.5">
            <span className="text-green-400">Stamina</span>
            <span className="text-gray-300">{Math.floor(player.stamina)}/{player.maxStamina}</span>
          </div>
          <div className="h-3 bg-gray-900 rounded overflow-hidden border border-green-900">
            <div
              className="h-full bg-gradient-to-r from-green-900 to-green-500 transition-all duration-100"
              style={{ width: `${(player.stamina / player.maxStamina) * 100}%` }}
            />
          </div>
        </div>

        {/* XP */}
        <div>
          <div className="flex justify-between text-xs mb-0.5">
            <span className="text-yellow-400">XP (Lv.{player.level})</span>
            <span className="text-gray-300">{Math.floor(xpProgress.current)}/{xpProgress.needed}</span>
          </div>
          <div className="h-2 bg-gray-900 rounded overflow-hidden border border-yellow-900">
            <div
              className="h-full bg-gradient-to-r from-yellow-900 to-yellow-500 transition-all duration-300"
              style={{ width: `${xpProgress.percent}%` }}
            />
          </div>
        </div>

        {/* Souls */}
        <div className="flex items-center gap-2">
          <span className="text-purple-400 text-sm">⚡</span>
          <span className="text-purple-300 text-sm font-bold">{player.souls.toLocaleString()} Souls</span>
        </div>

        {/* Hollowing */}
        {hollowingStacks > 0 && (
          <div className="flex items-center gap-1">
            <span className="text-gray-400 text-xs">Hollowing:</span>
            {Array.from({ length: 5 }).map((_, i) => (
              <div
                key={i}
                className={`w-3 h-3 rounded-sm border ${i < hollowingStacks ? 'bg-gray-600 border-gray-400' : 'bg-transparent border-gray-700'}`}
              />
            ))}
          </div>
        )}
      </div>

      {/* Top-center: Zone name */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 text-center">
        <div className="text-gray-400 text-sm tracking-widest uppercase">
          {gameState.currentZoneData?.name || ''}
        </div>
      </div>

      {/* Top-right: Minimap */}
      <div className="absolute top-4 right-4">
        <div className="border border-purple-900 rounded" style={{ width: 150, height: 150 }}>
          <canvas ref={minimapRef} width={150} height={150} className="rounded" />
        </div>
        <div className="text-center text-xs text-gray-500 mt-1">
          <span className="text-white">●</span> You &nbsp;
          <span className="text-red-500">●</span> Enemy &nbsp;
          <span className="text-cyan-400">●</span> NPC
        </div>
      </div>

      {/* Bottom-left: Status effects */}
      {statusEffects.length > 0 && (
        <div className="absolute bottom-24 left-4 flex gap-1">
          {statusEffects.map(effect => (
            <StatusEffectIcon key={effect.id} effect={effect} />
          ))}
        </div>
      )}

      {/* Bottom-center: Controls hint */}
      <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 text-center">
        <div className="text-gray-600 text-xs space-x-4">
          <span>WASD Move</span>
          <span>F Attack</span>
          <span>Space Dodge</span>
          <span>Q Parry</span>
          <span>E Interact</span>
          <span>I Inventory</span>
          <span>Tab Quests</span>
        </div>
      </div>

      {/* Bottom-right: Equipped weapon */}
      {equippedWeapon && (
        <div className="absolute bottom-16 right-4 text-right">
          <div className="text-gray-400 text-xs">Equipped</div>
          <div className="text-white text-sm">{equippedWeapon}</div>
        </div>
      )}

      {/* Active quest tracker */}
      {activeQuests.length > 0 && (
        <div className="absolute top-32 right-4 w-52">
          <div className="text-gray-500 text-xs uppercase tracking-wider mb-1">Active Quest</div>
          <div className="bg-black bg-opacity-60 border border-gray-800 rounded p-2">
            <div className="text-yellow-400 text-xs font-bold">{activeQuests[0].name}</div>
            {activeQuests[0].objectives.filter(o => !o.completed).slice(0, 2).map(obj => (
              <div key={obj.id} className="text-gray-400 text-xs mt-1">▸ {obj.text}</div>
            ))}
          </div>
        </div>
      )}

      {/* Boss dialogue */}
      {bossDialogue && (
        <div className="absolute top-1/4 left-1/2 transform -translate-x-1/2 text-center max-w-lg">
          <div className="text-red-400 text-lg italic" style={{ textShadow: '0 0 20px #ff0000' }}>
            "{bossDialogue.text}"
          </div>
        </div>
      )}

      {/* Final Erasure warning */}
      {finalErasureCharging && (
        <div className="absolute top-1/3 left-1/2 transform -translate-x-1/2 text-center">
          <div className="text-red-500 text-2xl font-bold animate-pulse" style={{ textShadow: '0 0 30px #ff0000' }}>
            ⚠ FINAL ERASURE CHARGING ⚠
          </div>
          <div className="text-red-400 text-lg mt-2">
            {Math.ceil(finalErasureTimer)}s — INTERRUPT NOW!
          </div>
          <div className="text-gray-400 text-sm mt-1">Press F to attack and interrupt</div>
        </div>
      )}

      {/* Shrine prompt */}
      {nearShrine && (
        <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 text-center">
          <div className="bg-black bg-opacity-70 border border-purple-700 rounded px-4 py-2">
            <div className="text-purple-300 text-sm">[E] Rest at Shrine</div>
            <div className="text-gray-500 text-xs">Restores HP & Stamina • Respawn point set</div>
          </div>
        </div>
      )}

      {/* Combat message */}
      {lastMessage && (
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 pointer-events-none">
          <div className="text-yellow-300 text-sm opacity-70">{lastMessage}</div>
        </div>
      )}

      {/* Shrine activated notification */}
      {gameState.shrineActivated && (
        <div className="absolute top-1/3 left-1/2 transform -translate-x-1/2 text-center">
          <div className="text-purple-300 text-xl" style={{ textShadow: '0 0 20px #6600ff' }}>
            ✦ Shrine Activated ✦
          </div>
          <div className="text-gray-400 text-sm mt-1">HP and Stamina restored</div>
        </div>
      )}

      {/* Dropped souls indicator */}
      {gameState.droppedSouls && gameState.droppedSouls.zone === gameState.currentZone && (
        <div className="absolute bottom-32 left-1/2 transform -translate-x-1/2 text-center">
          <div className="text-purple-400 text-sm animate-pulse">
            ⚡ {gameState.droppedSouls.amount} souls dropped nearby
          </div>
        </div>
      )}
    </div>
  );
}
