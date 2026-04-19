import React from 'react';

export default function DialogueUI({ dialogue, engine }) {
  if (!dialogue) return null;

  const { npcName, displayedText, responses, isTyping, isEnd } = dialogue;

  const handleClick = () => {
    if (isTyping) {
      engine.handleDialogueSkip();
    }
  };

  return (
    <div
      className="fixed bottom-0 left-0 right-0 z-30 pointer-events-auto"
      style={{ fontFamily: 'Georgia, serif' }}
    >
      {/* Backdrop */}
      <div className="bg-black bg-opacity-90 border-t border-gray-800 p-6 mx-4 mb-4 rounded-t">

        {/* NPC name */}
        <div className="text-purple-400 text-sm uppercase tracking-widest mb-3">
          {npcName}
        </div>

        {/* Dialogue text */}
        <div
          className="text-gray-200 text-base leading-relaxed mb-4 min-h-16 cursor-pointer"
          onClick={handleClick}
          style={{ minHeight: '4rem' }}
        >
          {displayedText}
          {isTyping && <span className="animate-pulse text-gray-500">▌</span>}
        </div>

        {/* Skip hint */}
        {isTyping && (
          <div className="text-gray-600 text-xs mb-2">Click to skip</div>
        )}

        {/* Response options */}
        {!isTyping && responses && responses.length > 0 && (
          <div className="space-y-2">
            {responses.map((response, i) => (
              <button
                key={i}
                onClick={() => engine.handleDialogueResponse(i)}
                className="w-full text-left px-4 py-2 border border-gray-700 text-gray-300 hover:bg-gray-800 hover:text-white hover:border-gray-500 transition-all duration-150 text-sm"
              >
                <span className="text-gray-600 mr-2">{i + 1}.</span>
                {response.text}
              </button>
            ))}
          </div>
        )}

        {/* End of dialogue */}
        {!isTyping && (!responses || responses.length === 0) && (
          <button
            onClick={() => engine.handleDialogueResponse(0)}
            className="px-6 py-2 border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500 transition-colors text-sm"
          >
            [Close]
          </button>
        )}
      </div>
    </div>
  );
}
