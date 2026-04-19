import React, { useState, useEffect } from 'react';

export default function MainMenu({ onNewGame, onContinue, hasSave, saveInfo }) {
  const [showCredits, setShowCredits] = useState(false);
  const [titleGlow, setTitleGlow] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setTitleGlow(prev => (prev + 0.05) % (Math.PI * 2));
    }, 50);
    return () => clearInterval(interval);
  }, []);

  const glowIntensity = Math.sin(titleGlow) * 0.5 + 0.5;

  if (showCredits) {
    return (
      <div className="fixed inset-0 bg-black flex items-center justify-center">
        <div className="text-center max-w-lg">
          <h2 className="text-purple-400 text-2xl mb-8">The War of the Idols</h2>
          <div className="text-gray-400 space-y-2 text-sm">
            <p>A dark fantasy space RPG</p>
            <p className="mt-4 text-gray-500">Built with React + Three.js</p>
            <p className="text-gray-500">Procedural geometry • No external assets</p>
            <p className="text-gray-500">localStorage saves • Complete end-to-end</p>
          </div>
          <button
            onClick={() => setShowCredits(false)}
            className="mt-8 px-6 py-2 border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500 transition-colors"
          >
            Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center"
      style={{
        background: 'radial-gradient(ellipse at center, #0a0010 0%, #000000 100%)',
      }}
    >
      {/* Animated background particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {Array.from({ length: 30 }).map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full"
            style={{
              width: Math.random() * 3 + 1,
              height: Math.random() * 3 + 1,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: `rgba(${Math.random() > 0.5 ? '100,0,255' : '255,0,100'}, ${Math.random() * 0.5 + 0.1})`,
              animation: `pulse ${2 + Math.random() * 3}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 3}s`,
            }}
          />
        ))}
      </div>

      {/* Title */}
      <div className="text-center mb-16 relative z-10">
        <div
          className="text-6xl font-bold tracking-widest mb-2"
          style={{
            color: '#ffffff',
            textShadow: `0 0 ${20 + glowIntensity * 30}px rgba(150, 0, 255, ${0.5 + glowIntensity * 0.5}), 0 0 60px rgba(100, 0, 200, 0.3)`,
            fontFamily: 'Georgia, serif',
          }}
        >
          THE WAR
        </div>
        <div
          className="text-4xl tracking-widest"
          style={{
            color: '#cc88ff',
            textShadow: `0 0 ${15 + glowIntensity * 20}px rgba(200, 100, 255, ${0.4 + glowIntensity * 0.4})`,
            fontFamily: 'Georgia, serif',
          }}
        >
          OF THE IDOLS
        </div>
        <div className="text-gray-600 text-sm tracking-widest mt-4 uppercase">
          A Dark Fantasy Space RPG
        </div>
      </div>

      {/* Menu buttons */}
      <div className="flex flex-col items-center gap-4 relative z-10 w-64">
        {hasSave && (
          <button
            onClick={onContinue}
            className="w-full py-3 border border-purple-700 text-purple-300 hover:bg-purple-900 hover:bg-opacity-30 hover:text-white transition-all duration-200 tracking-widest text-sm uppercase"
            style={{ fontFamily: 'Georgia, serif' }}
          >
            Continue
            {saveInfo && (
              <div className="text-xs text-gray-500 mt-1 normal-case tracking-normal">
                {saveInfo.zone} • Level {saveInfo.level}
              </div>
            )}
          </button>
        )}

        <button
          onClick={onNewGame}
          className="w-full py-3 border border-gray-700 text-gray-300 hover:bg-gray-900 hover:bg-opacity-50 hover:text-white transition-all duration-200 tracking-widest text-sm uppercase"
          style={{ fontFamily: 'Georgia, serif' }}
        >
          {hasSave ? 'New Game' : 'Begin'}
        </button>

        <button
          onClick={() => setShowCredits(true)}
          className="w-full py-2 text-gray-600 hover:text-gray-400 transition-colors text-sm tracking-widest uppercase"
          style={{ fontFamily: 'Georgia, serif' }}
        >
          Credits
        </button>
      </div>

      {/* Lore quote */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 text-center max-w-md">
        <div className="text-gray-700 text-xs italic">
          "When the last anchor walks willingly into the wound, a flame will come from the ash."
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.2; transform: scale(1); }
          50% { opacity: 0.8; transform: scale(1.5); }
        }
      `}</style>
    </div>
  );
}
