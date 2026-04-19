import React, { useState, useEffect } from 'react';

const ENDINGS = {
  seal: {
    title: 'The Seal',
    subtitle: 'You sacrificed yourself',
    color: '#4488ff',
    glowColor: 'rgba(68, 136, 255, 0.6)',
    text: `You stepped into the wound and became the seal.

The Void Sovereign's power — vast, ancient, incomprehensible — was not destroyed. It was contained. Within you. Within what you chose to become.

The Weave holds. The worlds continue. The stars burn on.

No one will know your name. No one will remember what you did. The wound is closed, and the universe moves forward, unaware of the flame that burned itself out to keep it warm.

But the Navigator knows. She was watching, as she always watches.

She does not cry. She has forgotten how.

She simply says: "Thank you."

And means it.`,
  },
  become: {
    title: 'The Ascension',
    subtitle: 'You absorbed the Sovereign\'s power',
    color: '#aa00ff',
    glowColor: 'rgba(170, 0, 255, 0.6)',
    text: `You reached into the Void Sovereign and took what it had taken from the Weave.

The power is yours now. Vast. Ancient. Incomprehensible.

You are not the Sovereign. You are something new. Something that stands at the boundary between existence and void, between holding and releasing.

The wound is... not closed. But it is no longer widening. You are the wound now, and you have chosen to be still.

The Weave stabilizes. The worlds continue. The stars burn on.

Whether you are a guardian or a threat, no one can say. Not even you.

The Navigator watches you from a distance. She does not approach.

She has seen this before. She knows how it ends.

She hopes, this time, she is wrong.`,
  },
  release: {
    title: 'The Unmaking',
    subtitle: 'You spoke the True Name',
    color: '#ffdd00',
    glowColor: 'rgba(255, 220, 0, 0.6)',
    text: `You spoke the name.

VAEL'THERON.

The Void Sovereign heard its own name for the first time in three thousand years. And in hearing it, remembered what it was before the choice. Before the wound. Before the Void.

It remembered being an anchor. It remembered holding.

And in remembering, it let go of what it had become.

The wound closed. Not sealed — unmade. As if it had never been.

You ceased to exist in the same moment. The True Name required a vessel, and you were the vessel, and the vessel was consumed.

But the Weave remembers you. Every thread, every world, every life that continues because of what you did — they carry a warmth they cannot explain.

The Navigator stands in Ashfeld, in the ash that no longer burns.

She looks at the sky, which is no longer torn.

She says your name.

And the universe, for just a moment, listens.`,
  },
};

export default function VictoryScreen({ endingChoice, player, onNewGamePlus, onMainMenu }) {
  const [phase, setPhase] = useState(0); // 0: fade in, 1: title, 2: text, 3: buttons
  const [displayedText, setDisplayedText] = useState('');
  const ending = ENDINGS[endingChoice] || ENDINGS.seal;

  useEffect(() => {
    const t1 = setTimeout(() => setPhase(1), 500);
    const t2 = setTimeout(() => setPhase(2), 2000);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, []);

  useEffect(() => {
    if (phase !== 2) return;
    let i = 0;
    const text = ending.text;
    const interval = setInterval(() => {
      if (i < text.length) {
        setDisplayedText(text.slice(0, i + 1));
        i++;
      } else {
        clearInterval(interval);
        setTimeout(() => setPhase(3), 1000);
      }
    }, 20);
    return () => clearInterval(interval);
  }, [phase, ending.text]);

  return (
    <div
      className="fixed inset-0 flex flex-col items-center justify-center z-50 overflow-y-auto py-8"
      style={{
        background: 'radial-gradient(ellipse at center, #050005 0%, #000000 100%)',
        opacity: phase >= 1 ? 1 : 0,
        transition: 'opacity 1s ease',
      }}
    >
      <div className="max-w-2xl w-full mx-4 text-center" style={{ fontFamily: 'Georgia, serif' }}>

        {/* Ending title */}
        {phase >= 1 && (
          <div className="mb-8">
            <div
              className="text-5xl font-bold mb-2"
              style={{
                color: ending.color,
                textShadow: `0 0 40px ${ending.glowColor}`,
              }}
            >
              {ending.title}
            </div>
            <div className="text-gray-500 text-lg italic">{ending.subtitle}</div>
          </div>
        )}

        {/* Stats */}
        {phase >= 1 && player && (
          <div className="flex justify-center gap-8 mb-8 text-sm">
            <div className="text-center">
              <div className="text-gray-500">Level</div>
              <div className="text-white text-xl">{player.level}</div>
            </div>
            <div className="text-center">
              <div className="text-gray-500">Souls Earned</div>
              <div className="text-purple-300 text-xl">{player.souls?.toLocaleString()}</div>
            </div>
            <div className="text-center">
              <div className="text-gray-500">Hollowing</div>
              <div className="text-gray-300 text-xl">{player.hollowing || 0}/5</div>
            </div>
          </div>
        )}

        {/* Ending text */}
        {phase >= 2 && (
          <div
            className="text-gray-300 text-sm leading-relaxed text-left whitespace-pre-line mb-8 px-4"
            style={{ minHeight: '200px' }}
          >
            {displayedText}
          </div>
        )}

        {/* Buttons */}
        {phase >= 3 && (
          <div className="flex flex-col items-center gap-4 mt-8">
            <button
              onClick={onNewGamePlus}
              className="px-8 py-3 border text-sm tracking-widest uppercase transition-all duration-200"
              style={{
                borderColor: ending.color,
                color: ending.color,
              }}
            >
              New Game+ (Carry Over Progress)
            </button>
            <button
              onClick={onMainMenu}
              className="px-8 py-2 border border-gray-700 text-gray-500 hover:text-gray-300 hover:border-gray-500 transition-colors text-sm tracking-widest uppercase"
            >
              Return to Main Menu
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
