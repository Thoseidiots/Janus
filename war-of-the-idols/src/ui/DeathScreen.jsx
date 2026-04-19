import React, { useEffect, useState } from 'react';

const DEATH_MESSAGES = [
  "YOU DIED",
  "HOLLOWED",
  "CONSUMED BY VOID",
  "ERASED",
  "FORGOTTEN",
];

export default function DeathScreen({ onRespawn, hollowingStacks, soulsLost }) {
  const [visible, setVisible] = useState(false);
  const [message] = useState(() => {
    if (hollowingStacks >= 5) return "FULLY HOLLOWED";
    if (hollowingStacks >= 3) return "CONSUMED BY VOID";
    return DEATH_MESSAGES[Math.floor(Math.random() * DEATH_MESSAGES.length)];
  });

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), 100);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center z-50"
      style={{
        background: `rgba(0, 0, 0, ${visible ? 0.85 : 0})`,
        transition: 'background 1s ease',
      }}
    >
      <div
        className="text-center"
        style={{
          opacity: visible ? 1 : 0,
          transition: 'opacity 1.5s ease',
        }}
      >
        <div
          className="text-7xl font-bold tracking-widest mb-4"
          style={{
            color: '#cc0000',
            textShadow: '0 0 40px rgba(200, 0, 0, 0.8), 0 0 80px rgba(150, 0, 0, 0.4)',
            fontFamily: 'Georgia, serif',
          }}
        >
          {message}
        </div>

        {soulsLost > 0 && (
          <div className="text-purple-400 text-lg mb-2">
            {soulsLost.toLocaleString()} souls lost
          </div>
        )}

        {hollowingStacks > 0 && (
          <div className="text-gray-500 text-sm mb-6">
            Hollowing: {hollowingStacks}/5 stacks
            {hollowingStacks >= 5 && (
              <span className="text-red-600 ml-2">— Max HP severely reduced</span>
            )}
          </div>
        )}

        <div className="text-gray-600 text-sm mt-8 animate-pulse">
          Returning to last shrine...
        </div>

        <button
          onClick={onRespawn}
          className="mt-6 px-8 py-3 border border-red-900 text-red-400 hover:bg-red-900 hover:bg-opacity-20 hover:text-red-300 transition-all duration-200 tracking-widest text-sm uppercase"
          style={{ fontFamily: 'Georgia, serif' }}
        >
          Rise Again
        </button>
      </div>
    </div>
  );
}
