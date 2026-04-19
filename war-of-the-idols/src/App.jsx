import React, { useState, useEffect, useRef, useCallback } from 'react';
import MainMenu from './ui/MainMenu.jsx';
import HUD from './ui/HUD.jsx';
import DeathScreen from './ui/DeathScreen.jsx';
import LevelUpScreen from './ui/LevelUpScreen.jsx';
import VictoryScreen from './ui/VictoryScreen.jsx';
import InventoryUI from './ui/InventoryUI.jsx';
import QuestUI from './ui/QuestUI.jsx';
import LoreUI from './ui/LoreUI.jsx';
import DialogueUI from './ui/DialogueUI.jsx';
import { GameEngine } from './game/engine/GameEngine.js';
import { SaveSystem } from './game/systems/SaveSystem.js';

const GAME_STATES = {
  MAIN_MENU: 'MAIN_MENU',
  PLAYING: 'PLAYING',
  ENDING: 'ENDING',
};

// Shared game state object (mutable, passed by reference to engine)
const createInitialGameState = () => ({
  player: null,
  inventory: null,
  currentZone: 'ashfeld',
  currentZoneData: null,
  defeatedBosses: [],
  discoveredZones: [],
  quests: {},
  lore: {},
  trueNameKnown: false,
  allLoreCollected: false,
  endingChosen: null,
  playTime: 0,
  newGamePlus: 0,
  droppedSouls: null,
  respawnZone: 'ashfeld',
  respawnPosition: { x: 0, y: 0, z: 0 },
  activeDialogue: null,
  bossDialogue: null,
  nearShrine: false,
  shrineActivated: false,
  minimapDataURL: null,
  lastCombatMessage: null,
  lastMessage: null,
  pendingXP: 0,
  pendingEnemyRespawn: false,
  pendingInteract: null,
  showInventory: false,
  showQuests: false,
  showLore: false,
  showDeath: false,
  showLevelUp: false,
  showEnding: false,
  finalBossDefeated: false,
  arenaExpanded: false,
  // System references (set by engine)
  inventorySystem: null,
  questSystem: null,
  loreSystem: null,
  levelSystem: null,
  dialogueSystem: null,
  combatSystem: null,
  audioManager: null,
});

export default function App() {
  const [appState, setAppState] = useState(GAME_STATES.MAIN_MENU);
  const [uiState, setUiState] = useState(createInitialGameState());
  const [hasSave, setHasSave] = useState(false);
  const [saveInfo, setSaveInfo] = useState(null);
  const canvasRef = useRef(null);
  const engineRef = useRef(null);
  const gameStateRef = useRef(createInitialGameState());

  // Check for existing save on mount
  useEffect(() => {
    const tempSave = new SaveSystem(gameStateRef.current);
    const info = tempSave.getSaveInfo('main');
    if (info) {
      setHasSave(true);
      setSaveInfo(info);
    }
  }, []);

  const handleStateChange = useCallback((newState) => {
    // Update UI state (triggers re-render)
    setUiState({ ...newState });
  }, []);

  const startGame = useCallback((savedData = null) => {
    setAppState(GAME_STATES.PLAYING);

    // Give React time to render the canvas
    setTimeout(() => {
      if (!canvasRef.current) return;

      // Clean up existing engine
      if (engineRef.current) {
        engineRef.current.stop();
      }

      // Reset game state
      gameStateRef.current = createInitialGameState();

      // Create and init engine
      const engine = new GameEngine(canvasRef.current, gameStateRef.current, handleStateChange);
      engineRef.current = engine;
      engine.init(savedData);
    }, 100);
  }, [handleStateChange]);

  const handleNewGame = useCallback(() => {
    startGame(null);
  }, [startGame]);

  const handleContinue = useCallback(() => {
    const tempSave = new SaveSystem(gameStateRef.current);
    const savedData = tempSave.load('main');
    if (savedData) {
      startGame(savedData);
    } else {
      startGame(null);
    }
  }, [startGame]);

  const handleRespawn = useCallback(() => {
    if (engineRef.current) {
      engineRef.current._respawnPlayer();
      handleStateChange({ ...gameStateRef.current, showDeath: false });
    }
  }, [handleStateChange]);

  const handleLevelUpStat = useCallback((stat) => {
    if (engineRef.current) {
      engineRef.current.handleLevelUpStat(stat);
    }
  }, []);

  const handleCloseLevelUp = useCallback(() => {
    handleStateChange({ ...gameStateRef.current, showLevelUp: false });
  }, [handleStateChange]);

  const handleChooseEnding = useCallback((choice) => {
    if (engineRef.current) {
      engineRef.current.handleChooseEnding(choice);
    }
    setAppState(GAME_STATES.ENDING);
  }, []);

  const handleNewGamePlus = useCallback(() => {
    const gs = gameStateRef.current;
    gs.newGamePlus = (gs.newGamePlus || 0) + 1;
    gs.endingChosen = null;
    gs.finalBossDefeated = false;
    startGame(null);
  }, [startGame]);

  const handleMainMenu = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.stop();
      engineRef.current = null;
    }
    setAppState(GAME_STATES.MAIN_MENU);
    // Re-check save
    const tempSave = new SaveSystem(gameStateRef.current);
    const info = tempSave.getSaveInfo('main');
    setHasSave(!!info);
    setSaveInfo(info);
  }, []);

  const handleCloseInventory = useCallback(() => {
    handleStateChange({ ...gameStateRef.current, showInventory: false });
  }, [handleStateChange]);

  const handleCloseQuests = useCallback(() => {
    handleStateChange({ ...gameStateRef.current, showQuests: false });
  }, [handleStateChange]);

  const handleCloseLore = useCallback(() => {
    handleStateChange({ ...gameStateRef.current, showLore: false });
  }, [handleStateChange]);

  // Keyboard shortcut for lore (L key)
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key.toLowerCase() === 'l' && appState === GAME_STATES.PLAYING) {
        handleStateChange({ ...gameStateRef.current, showLore: !gameStateRef.current.showLore });
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [appState, handleStateChange]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (engineRef.current) {
        engineRef.current.stop();
      }
    };
  }, []);

  return (
    <div className="w-full h-full relative bg-black">

      {/* Three.js Canvas — always rendered when playing */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ display: appState === GAME_STATES.PLAYING ? 'block' : 'none' }}
      />

      {/* Main Menu */}
      {appState === GAME_STATES.MAIN_MENU && (
        <MainMenu
          onNewGame={handleNewGame}
          onContinue={handleContinue}
          hasSave={hasSave}
          saveInfo={saveInfo}
        />
      )}

      {/* Game UI overlay */}
      {appState === GAME_STATES.PLAYING && (
        <>
          {/* HUD */}
          {!uiState.showDeath && !uiState.showLevelUp && (
            <HUD gameState={uiState} engine={engineRef.current} />
          )}

          {/* Dialogue */}
          {uiState.activeDialogue && (
            <DialogueUI
              dialogue={uiState.activeDialogue}
              engine={engineRef.current}
            />
          )}

          {/* Inventory */}
          {uiState.showInventory && (
            <InventoryUI
              gameState={uiState}
              engine={engineRef.current}
              onClose={handleCloseInventory}
            />
          )}

          {/* Quests */}
          {uiState.showQuests && (
            <QuestUI
              gameState={uiState}
              onClose={handleCloseQuests}
            />
          )}

          {/* Lore */}
          {uiState.showLore && (
            <LoreUI
              gameState={uiState}
              onClose={handleCloseLore}
            />
          )}

          {/* Death screen */}
          {uiState.showDeath && (
            <DeathScreen
              onRespawn={handleRespawn}
              hollowingStacks={uiState.player?.hollowing || 0}
              soulsLost={uiState.droppedSouls?.amount || 0}
            />
          )}

          {/* Level up */}
          {uiState.showLevelUp && !uiState.showDeath && (
            <LevelUpScreen
              player={uiState.player}
              onAllocate={handleLevelUpStat}
              onClose={handleCloseLevelUp}
            />
          )}

          {/* Ending choice (when final boss defeated) */}
          {uiState.finalBossDefeated && !uiState.endingChosen && (
            <EndingChoiceScreen onChoose={handleChooseEnding} />
          )}

          {/* Lore key hint */}
          <div className="fixed bottom-4 right-4 text-gray-700 text-xs pointer-events-none">
            L — Lore Archive
          </div>
        </>
      )}

      {/* Victory/Ending screen */}
      {appState === GAME_STATES.ENDING && (
        <VictoryScreen
          endingChoice={uiState.endingChosen}
          player={uiState.player}
          onNewGamePlus={handleNewGamePlus}
          onMainMenu={handleMainMenu}
        />
      )}
    </div>
  );
}

function EndingChoiceScreen({ onChoose }) {
  const choices = [
    {
      id: 'seal',
      title: 'Seal It',
      description: 'Sacrifice yourself. Become the seal. The universe is saved, but you are gone.',
      color: '#4488ff',
    },
    {
      id: 'become',
      title: 'Become It',
      description: 'Absorb the Sovereign\'s power. Become something new. The wound closes, but what are you now?',
      color: '#aa00ff',
    },
    {
      id: 'release',
      title: 'Release It',
      description: 'Speak the True Name. Unmake the wound. You cease to exist, but the Sovereign is freed.',
      color: '#ffdd00',
    },
  ];

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-90 flex items-center justify-center z-50"
      style={{ fontFamily: 'Georgia, serif' }}
    >
      <div className="max-w-2xl w-full mx-4 text-center">
        <div className="text-gray-400 text-sm uppercase tracking-widest mb-2">The Void Sovereign is defeated</div>
        <div className="text-white text-3xl mb-2">Choose Your Ending</div>
        <div className="text-gray-600 text-sm mb-8 italic">
          "Three paths. One flame. The Weave holds its breath."
        </div>

        <div className="grid grid-cols-1 gap-4">
          {choices.map(choice => (
            <button
              key={choice.id}
              onClick={() => onChoose(choice.id)}
              className="border p-6 text-left transition-all duration-200 hover:bg-opacity-10"
              style={{
                borderColor: choice.color,
                color: choice.color,
              }}
            >
              <div className="text-xl font-bold mb-2">{choice.title}</div>
              <div className="text-gray-400 text-sm">{choice.description}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
