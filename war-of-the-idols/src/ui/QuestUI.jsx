import React, { useState } from 'react';

const QUEST_TYPE_COLORS = {
  main: 'text-yellow-400 border-yellow-800',
  side: 'text-blue-400 border-blue-800',
};

export default function QuestUI({ gameState, onClose }) {
  const [filter, setFilter] = useState('active');
  const [selectedQuest, setSelectedQuest] = useState(null);

  const quests = gameState.quests || {};
  const allQuests = Object.values(quests);

  const filtered = allQuests.filter(q => {
    if (filter === 'active') return q.active && !q.completed;
    if (filter === 'completed') return q.completed;
    if (filter === 'main') return q.type === 'main';
    if (filter === 'side') return q.type === 'side';
    return true;
  });

  const selected = selectedQuest ? quests[selectedQuest] : null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-85 flex items-center justify-center z-40" style={{ fontFamily: 'Georgia, serif' }}>
      <div className="border border-gray-800 bg-black bg-opacity-95 p-6 max-w-3xl w-full mx-4 max-h-screen overflow-y-auto">

        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-gray-300 text-xl tracking-widest uppercase">Quests</h2>
          <button onClick={onClose} className="text-gray-600 hover:text-white text-xl">✕</button>
        </div>

        {/* Filter tabs */}
        <div className="flex gap-2 mb-4">
          {['active', 'completed', 'main', 'side'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-3 py-1 text-xs uppercase tracking-wider border transition-colors ${
                filter === f
                  ? 'border-gray-400 text-white'
                  : 'border-gray-800 text-gray-600 hover:border-gray-600 hover:text-gray-400'
              }`}
            >
              {f}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-4">
          {/* Quest list */}
          <div className="space-y-2">
            {filtered.length === 0 && (
              <div className="text-gray-700 text-sm text-center py-8">No quests in this category</div>
            )}
            {filtered.map(quest => (
              <div
                key={quest.id}
                className={`border p-3 cursor-pointer transition-colors ${
                  selectedQuest === quest.id
                    ? 'border-gray-400 bg-gray-900'
                    : 'border-gray-800 hover:border-gray-600'
                }`}
                onClick={() => setSelectedQuest(quest.id)}
              >
                <div className="flex items-center gap-2">
                  <span className={`text-xs uppercase ${QUEST_TYPE_COLORS[quest.type]?.split(' ')[0] || 'text-gray-400'}`}>
                    {quest.type}
                  </span>
                  {quest.completed && <span className="text-green-500 text-xs">✓</span>}
                </div>
                <div className={`text-sm mt-0.5 ${quest.completed ? 'text-gray-600 line-through' : 'text-gray-200'}`}>
                  {quest.name}
                </div>
                {!quest.completed && quest.active && (
                  <div className="text-xs text-gray-600 mt-1">
                    {quest.objectives.filter(o => o.completed).length}/{quest.objectives.length} objectives
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Quest details */}
          <div>
            {selected ? (
              <div className="border border-gray-800 p-4">
                <div className={`text-xs uppercase mb-1 ${QUEST_TYPE_COLORS[selected.type]?.split(' ')[0] || 'text-gray-400'}`}>
                  {selected.type} Quest
                </div>
                <div className="text-white text-lg mb-2">{selected.name}</div>
                <div className="text-gray-400 text-sm mb-4 leading-relaxed">{selected.description}</div>

                <div className="mb-4">
                  <div className="text-gray-500 text-xs uppercase tracking-wider mb-2">Objectives</div>
                  <div className="space-y-1">
                    {selected.objectives.map(obj => (
                      <div key={obj.id} className="flex items-start gap-2 text-sm">
                        <span className={obj.completed ? 'text-green-500' : 'text-gray-600'}>
                          {obj.completed ? '✓' : '○'}
                        </span>
                        <span className={obj.completed ? 'text-gray-600 line-through' : 'text-gray-300'}>
                          {obj.text}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                {selected.rewards && (
                  <div>
                    <div className="text-gray-500 text-xs uppercase tracking-wider mb-2">Rewards</div>
                    <div className="space-y-1 text-xs">
                      {selected.rewards.xp && (
                        <div className="text-yellow-400">+{selected.rewards.xp} XP</div>
                      )}
                      {selected.rewards.souls && (
                        <div className="text-purple-400">+{selected.rewards.souls} Souls</div>
                      )}
                      {selected.rewards.items?.map(itemId => (
                        <div key={itemId} className="text-blue-400">Item: {itemId}</div>
                      ))}
                    </div>
                  </div>
                )}

                {selected.completed && (
                  <div className="mt-4 text-green-500 text-sm">✓ Quest Completed</div>
                )}
              </div>
            ) : (
              <div className="border border-gray-800 p-4 text-gray-700 text-sm text-center">
                Select a quest to view details
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
