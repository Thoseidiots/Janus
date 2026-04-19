import React, { useState } from 'react';
import { LORE_CATEGORIES } from '../data/lore.js';

const CATEGORY_COLORS = {
  Cosmology: 'text-purple-400',
  History: 'text-yellow-400',
  Characters: 'text-blue-400',
  Prophecy: 'text-red-400',
  Bestiary: 'text-green-400',
};

export default function LoreUI({ gameState, onClose }) {
  const [selectedCategory, setSelectedCategory] = useState('All');
  const [selectedEntry, setSelectedEntry] = useState(null);

  const lore = gameState.lore || {};
  const allEntries = Object.values(lore);
  const discovered = allEntries.filter(e => e.discovered);

  const filtered = selectedCategory === 'All'
    ? discovered
    : discovered.filter(e => e.category === selectedCategory);

  const selected = selectedEntry ? lore[selectedEntry] : null;
  const { discovered: discoveredCount, total } = gameState.loreSystem?.getDiscoveryCount() || { discovered: discovered.length, total: allEntries.length };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-85 flex items-center justify-center z-40" style={{ fontFamily: 'Georgia, serif' }}>
      <div className="border border-gray-800 bg-black bg-opacity-95 p-6 max-w-4xl w-full mx-4 max-h-screen overflow-y-auto">

        {/* Header */}
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-gray-300 text-xl tracking-widest uppercase">Lore Archive</h2>
          <div className="flex items-center gap-4">
            <span className="text-gray-600 text-sm">{discoveredCount}/{total} entries</span>
            {gameState.allLoreCollected && (
              <span className="text-yellow-400 text-xs">✦ All Lore Collected — True Name Empowered</span>
            )}
            <button onClick={onClose} className="text-gray-600 hover:text-white text-xl">✕</button>
          </div>
        </div>

        {/* Category filter */}
        <div className="flex gap-2 mb-4 flex-wrap">
          {['All', ...LORE_CATEGORIES].map(cat => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1 text-xs uppercase tracking-wider border transition-colors ${
                selectedCategory === cat
                  ? 'border-gray-400 text-white'
                  : 'border-gray-800 text-gray-600 hover:border-gray-600 hover:text-gray-400'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-3 gap-4">
          {/* Entry list */}
          <div className="space-y-1 col-span-1">
            {filtered.length === 0 && (
              <div className="text-gray-700 text-sm text-center py-8">
                No entries discovered in this category
              </div>
            )}
            {filtered.map(entry => (
              <div
                key={entry.id}
                className={`border p-2 cursor-pointer transition-colors ${
                  selectedEntry === entry.id
                    ? 'border-gray-400 bg-gray-900'
                    : 'border-gray-800 hover:border-gray-600'
                }`}
                onClick={() => setSelectedEntry(entry.id)}
              >
                <div className={`text-xs ${CATEGORY_COLORS[entry.category] || 'text-gray-400'}`}>
                  {entry.category}
                </div>
                <div className="text-gray-200 text-sm">{entry.title}</div>
              </div>
            ))}

            {/* Undiscovered entries */}
            {allEntries.filter(e => !e.discovered).length > 0 && (
              <div className="mt-4">
                <div className="text-gray-700 text-xs uppercase tracking-wider mb-1">Undiscovered</div>
                {allEntries.filter(e => !e.discovered).map(entry => (
                  <div key={entry.id} className="border border-gray-900 p-2 mb-1">
                    <div className="text-gray-800 text-xs">{entry.category}</div>
                    <div className="text-gray-800 text-sm">???</div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Entry content */}
          <div className="col-span-2">
            {selected ? (
              <div className="border border-gray-800 p-6">
                <div className={`text-xs uppercase tracking-wider mb-1 ${CATEGORY_COLORS[selected.category] || 'text-gray-400'}`}>
                  {selected.category}
                </div>
                <div className="text-white text-xl mb-4">{selected.title}</div>
                <div className="text-gray-300 text-sm leading-relaxed whitespace-pre-line">
                  {selected.text}
                </div>
                <div className="mt-4 text-gray-700 text-xs">
                  Found in: {selected.zone}
                </div>
              </div>
            ) : (
              <div className="border border-gray-800 p-6 text-center">
                <div className="text-gray-700 text-sm">Select an entry to read</div>
                {discoveredCount === 0 && (
                  <div className="text-gray-800 text-xs mt-4">
                    Explore the world to discover lore entries
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
