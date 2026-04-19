import React, { useState } from 'react';
import { ITEMS, RARITIES } from '../data/items.js';

const SLOT_LABELS = {
  weapon: '⚔ Weapon',
  armor: '🛡 Armor',
  offhand: '🔰 Offhand',
  talisman: '💎 Talisman',
};

function ItemSlot({ slot, onClick, selected, empty = false }) {
  if (empty) {
    return (
      <div
        className="w-14 h-14 border border-gray-800 bg-gray-950 flex items-center justify-center cursor-pointer hover:border-gray-600 transition-colors"
        onClick={onClick}
      >
        <span className="text-gray-700 text-xs">Empty</span>
      </div>
    );
  }

  const item = ITEMS[slot?.itemId];
  if (!item) return <div className="w-14 h-14 border border-gray-800 bg-gray-950" />;

  const rarity = RARITIES[item.rarity] || RARITIES.COMMON;

  return (
    <div
      className={`w-14 h-14 border flex flex-col items-center justify-center cursor-pointer transition-all duration-150 relative ${
        selected ? 'border-white bg-gray-800' : 'border-gray-700 bg-gray-950 hover:border-gray-500'
      }`}
      style={{ borderColor: selected ? '#ffffff' : rarity.color }}
      onClick={onClick}
      title={`${item.name} (${rarity.name})`}
    >
      <div className="text-xs text-center px-1 leading-tight" style={{ color: rarity.color }}>
        {item.name.split(' ').slice(0, 2).join(' ')}
      </div>
      {slot?.quantity > 1 && (
        <div className="absolute bottom-0 right-0 text-xs text-gray-400 bg-gray-900 px-1">
          {slot.quantity}
        </div>
      )}
    </div>
  );
}

export default function InventoryUI({ gameState, engine, onClose }) {
  const [selectedSlot, setSelectedSlot] = useState(null);
  const [selectedEquipSlot, setSelectedEquipSlot] = useState(null);

  const inv = gameState.inventory || { slots: [], equipped: {} };
  const player = gameState.player;

  const selectedItem = selectedSlot !== null ? ITEMS[inv.slots[selectedSlot]?.itemId] : null;
  const selectedSlotData = selectedSlot !== null ? inv.slots[selectedSlot] : null;

  const handleEquip = () => {
    if (selectedSlot === null || !selectedItem) return;
    engine.handleEquipItem(inv.slots[selectedSlot].itemId);
    setSelectedSlot(null);
  };

  const handleUse = () => {
    if (selectedSlot === null || !selectedItem) return;
    engine.handleUseItem(inv.slots[selectedSlot].itemId);
    setSelectedSlot(null);
  };

  const handleUnequip = (slot) => {
    engine.handleUnequipSlot(slot);
    setSelectedEquipSlot(null);
  };

  const totalDefense = gameState.inventorySystem?.getTotalDefense() || 0;
  const totalDamage = gameState.inventorySystem?.getTotalDamage() || 0;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-85 flex items-center justify-center z-40" style={{ fontFamily: 'Georgia, serif' }}>
      <div className="border border-gray-800 bg-black bg-opacity-95 p-6 max-w-4xl w-full mx-4 max-h-screen overflow-y-auto">

        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-gray-300 text-xl tracking-widest uppercase">Inventory</h2>
          <div className="flex gap-6 text-sm">
            <span className="text-red-400">⚔ {totalDamage} ATK</span>
            <span className="text-blue-400">🛡 {totalDefense} DEF</span>
            <span className="text-purple-400">⚡ {player?.souls?.toLocaleString()} Souls</span>
          </div>
          <button onClick={onClose} className="text-gray-600 hover:text-white text-xl">✕</button>
        </div>

        <div className="grid grid-cols-3 gap-6">

          {/* Equipment slots */}
          <div>
            <div className="text-gray-500 text-xs uppercase tracking-wider mb-3">Equipped</div>
            <div className="space-y-2">
              {Object.entries(SLOT_LABELS).map(([slot, label]) => {
                const equippedId = inv.equipped?.[slot];
                const equippedItem = equippedId ? ITEMS[equippedId] : null;
                const rarity = equippedItem ? RARITIES[equippedItem.rarity] : null;
                return (
                  <div
                    key={slot}
                    className={`border p-2 cursor-pointer transition-colors ${
                      selectedEquipSlot === slot ? 'border-white' : 'border-gray-800 hover:border-gray-600'
                    }`}
                    onClick={() => setSelectedEquipSlot(selectedEquipSlot === slot ? null : slot)}
                  >
                    <div className="text-gray-600 text-xs">{label}</div>
                    {equippedItem ? (
                      <div className="text-sm mt-0.5" style={{ color: rarity?.color || '#fff' }}>
                        {equippedItem.name}
                      </div>
                    ) : (
                      <div className="text-gray-700 text-sm mt-0.5">— Empty —</div>
                    )}
                  </div>
                );
              })}
            </div>

            {selectedEquipSlot && inv.equipped?.[selectedEquipSlot] && (
              <button
                onClick={() => handleUnequip(selectedEquipSlot)}
                className="mt-3 w-full py-1 border border-gray-700 text-gray-400 hover:text-white text-xs uppercase tracking-wider"
              >
                Unequip
              </button>
            )}

            {/* Player stats */}
            <div className="mt-4 border-t border-gray-800 pt-4">
              <div className="text-gray-500 text-xs uppercase tracking-wider mb-2">Stats</div>
              <div className="space-y-1 text-xs">
                {['strength', 'dexterity', 'arcane', 'endurance', 'vitality'].map(stat => (
                  <div key={stat} className="flex justify-between">
                    <span className="text-gray-500 capitalize">{stat}</span>
                    <span className="text-white">{player?.[stat] || 10}</span>
                  </div>
                ))}
                <div className="flex justify-between border-t border-gray-800 pt-1 mt-1">
                  <span className="text-gray-500">Level</span>
                  <span className="text-yellow-400">{player?.level}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Hollowing</span>
                  <span className="text-gray-300">{player?.hollowing || 0}/5</span>
                </div>
              </div>
            </div>
          </div>

          {/* Inventory grid */}
          <div>
            <div className="text-gray-500 text-xs uppercase tracking-wider mb-3">
              Items ({inv.slots.length}/{20})
            </div>
            <div className="grid grid-cols-4 gap-1">
              {Array.from({ length: 20 }).map((_, i) => {
                const slot = inv.slots[i];
                return (
                  <ItemSlot
                    key={i}
                    slot={slot}
                    empty={!slot}
                    selected={selectedSlot === i}
                    onClick={() => setSelectedSlot(selectedSlot === i ? null : i)}
                  />
                );
              })}
            </div>
          </div>

          {/* Item details */}
          <div>
            <div className="text-gray-500 text-xs uppercase tracking-wider mb-3">Details</div>
            {selectedItem ? (
              <div className="border border-gray-800 p-4">
                <div
                  className="text-lg font-bold mb-1"
                  style={{ color: RARITIES[selectedItem.rarity]?.color || '#fff' }}
                >
                  {selectedItem.name}
                </div>
                <div className="text-gray-500 text-xs mb-3">
                  {RARITIES[selectedItem.rarity]?.name} {selectedItem.type}
                </div>
                <div className="text-gray-300 text-sm mb-4 leading-relaxed">
                  {selectedItem.description}
                </div>

                {/* Stats */}
                <div className="space-y-1 text-xs mb-4">
                  {selectedItem.damage && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Damage</span>
                      <span className="text-red-400">{selectedItem.damage}</span>
                    </div>
                  )}
                  {selectedItem.defense && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Defense</span>
                      <span className="text-blue-400">{selectedItem.defense}</span>
                    </div>
                  )}
                  {selectedItem.blockValue && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Block</span>
                      <span className="text-green-400">{selectedItem.blockValue}</span>
                    </div>
                  )}
                  {selectedItem.stats && Object.entries(selectedItem.stats).map(([stat, val]) => (
                    <div key={stat} className="flex justify-between">
                      <span className="text-gray-500 capitalize">{stat}</span>
                      <span className="text-yellow-400">+{val}</span>
                    </div>
                  ))}
                  {selectedItem.statusEffect && Object.entries(selectedItem.statusEffect).map(([effect, chance]) => (
                    <div key={effect} className="flex justify-between">
                      <span className="text-gray-500 capitalize">{effect}</span>
                      <span className="text-purple-400">{Math.floor(chance * 100)}% chance</span>
                    </div>
                  ))}
                  <div className="flex justify-between">
                    <span className="text-gray-500">Value</span>
                    <span className="text-yellow-600">{selectedItem.value} souls</span>
                  </div>
                </div>

                {/* Actions */}
                <div className="space-y-2">
                  {['weapon', 'armor', 'offhand', 'talisman'].includes(selectedItem.type) && (
                    <button
                      onClick={handleEquip}
                      className="w-full py-2 border border-gray-600 text-gray-300 hover:bg-gray-800 text-xs uppercase tracking-wider transition-colors"
                    >
                      {inv.equipped?.[selectedItem.type] === selectedSlotData?.itemId ? 'Equipped' : 'Equip'}
                    </button>
                  )}
                  {selectedItem.type === 'consumable' && (
                    <button
                      onClick={handleUse}
                      className="w-full py-2 border border-green-800 text-green-400 hover:bg-green-900 hover:bg-opacity-30 text-xs uppercase tracking-wider transition-colors"
                    >
                      Use ({selectedSlotData?.quantity || 1} remaining)
                    </button>
                  )}
                </div>
              </div>
            ) : (
              <div className="border border-gray-800 p-4 text-gray-700 text-sm text-center">
                Select an item to view details
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
