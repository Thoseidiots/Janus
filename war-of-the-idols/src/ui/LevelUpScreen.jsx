import React from 'react';
import { STAT_EFFECTS } from '../game/systems/LevelSystem.js';

const STAT_NAMES = {
  strength: 'Strength',
  dexterity: 'Dexterity',
  arcane: 'Arcane',
  endurance: 'Endurance',
  vitality: 'Vitality',
};

const STAT_COLORS = {
  strength: 'text-red-400 border-red-800 hover:bg-red-900',
  dexterity: 'text-green-400 border-green-800 hover:bg-green-900',
  arcane: 'text-purple-400 border-purple-800 hover:bg-purple-900',
  endurance: 'text-yellow-400 border-yellow-800 hover:bg-yellow-900',
  vitality: 'text-blue-400 border-blue-800 hover:bg-blue-900',
};

export default function LevelUpScreen({ player, onAllocate, onClose }) {
  const statPoints = player?.statPoints || 0;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-40">
      <div
        className="border border-purple-800 bg-black bg-opacity-90 p-8 max-w-lg w-full mx-4"
        style={{ fontFamily: 'Georgia, serif' }}
      >
        <div className="text-center mb-6">
          <div
            className="text-3xl text-yellow-400 mb-2"
            style={{ textShadow: '0 0 20px rgba(255, 200, 0, 0.6)' }}
          >
            ✦ Level {player?.level} ✦
          </div>
          <div className="text-gray-400 text-sm">
            {statPoints > 0
              ? `${statPoints} stat point${statPoints > 1 ? 's' : ''} to allocate`
              : 'All points allocated'}
          </div>
        </div>

        <div className="grid grid-cols-1 gap-3 mb-6">
          {Object.entries(STAT_NAMES).map(([stat, name]) => {
            const currentVal = player?.[stat] || 10;
            const effects = STAT_EFFECTS[stat];
            return (
              <div
                key={stat}
                className={`border rounded p-3 flex items-center justify-between ${STAT_COLORS[stat]} bg-opacity-10`}
              >
                <div className="flex-1">
                  <div className="flex items-center gap-3">
                    <span className={`font-bold ${STAT_COLORS[stat].split(' ')[0]}`}>{name}</span>
                    <span className="text-white text-lg font-bold">{currentVal}</span>
                  </div>
                  <div className="text-gray-600 text-xs mt-0.5">{effects?.description}</div>
                </div>
                {statPoints > 0 && (
                  <button
                    onClick={() => onAllocate(stat)}
                    className={`ml-4 px-3 py-1 border text-sm transition-all duration-150 ${STAT_COLORS[stat]}`}
                  >
                    +1
                  </button>
                )}
              </div>
            );
          })}
        </div>

        {/* Current derived stats */}
        <div className="border-t border-gray-800 pt-4 mb-4">
          <div className="text-gray-500 text-xs uppercase tracking-wider mb-2">Derived Stats</div>
          <div className="grid grid-cols-2 gap-1 text-xs">
            <div className="text-gray-400">Max HP: <span className="text-white">{player?.maxHp}</span></div>
            <div className="text-gray-400">Max Stamina: <span className="text-white">{player?.maxStamina}</span></div>
            <div className="text-gray-400">Souls: <span className="text-purple-300">{player?.souls?.toLocaleString()}</span></div>
            <div className="text-gray-400">Hollowing: <span className="text-gray-300">{player?.hollowing || 0}/5</span></div>
          </div>
        </div>

        <button
          onClick={onClose}
          className="w-full py-2 border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500 transition-colors text-sm tracking-widest uppercase"
        >
          {statPoints > 0 ? 'Close (points remaining)' : 'Continue'}
        </button>
      </div>
    </div>
  );
}
