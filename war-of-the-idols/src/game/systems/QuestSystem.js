import { QUESTS } from '../../data/quests.js';

export class QuestSystem {
  constructor(gameState) {
    this.gameState = gameState;
  }

  getQuests() {
    return this.gameState.quests || {};
  }

  initQuests() {
    const quests = {};
    for (const [id, quest] of Object.entries(QUESTS)) {
      quests[id] = JSON.parse(JSON.stringify(quest));
    }
    // Activate first main quest
    quests.theAshenPath.active = true;
    this.gameState.quests = quests;
  }

  activateQuest(questId) {
    const quests = this.getQuests();
    if (quests[questId]) {
      quests[questId].active = true;
      this.gameState.quests = quests;
    }
  }

  completeObjective(questId, objectiveId) {
    const quests = this.getQuests();
    const quest = quests[questId];
    if (!quest || !quest.active) return false;

    const obj = quest.objectives.find(o => o.id === objectiveId);
    if (!obj || obj.completed) return false;

    obj.completed = true;

    // Check if all objectives complete
    if (quest.objectives.every(o => o.completed)) {
      this.completeQuest(questId);
    }

    this.gameState.quests = quests;
    return true;
  }

  completeQuest(questId) {
    const quests = this.getQuests();
    const quest = quests[questId];
    if (!quest) return false;

    quest.completed = true;
    quest.active = false;

    // Grant rewards
    const rewards = quest.rewards;
    if (rewards) {
      if (rewards.xp) this.gameState.pendingXP = (this.gameState.pendingXP || 0) + rewards.xp;
      if (rewards.souls) this.gameState.player.souls += rewards.souls;
      if (rewards.items) {
        rewards.items.forEach(itemId => {
          if (this.gameState.inventorySystem) {
            this.gameState.inventorySystem.addItem(itemId);
          }
        });
      }
    }

    // Unlock lore
    if (quest.loreUnlock && this.gameState.loreSystem) {
      this.gameState.loreSystem.discoverEntry(quest.loreUnlock);
    }

    // Activate follow-up quests
    this._activateFollowUpQuests(questId);

    this.gameState.quests = quests;
    this.gameState.lastQuestCompleted = questId;
    return true;
  }

  _activateFollowUpQuests(completedQuestId) {
    const followUps = {
      theAshenPath: ['theFrozenTruth', 'theChoirsSilence', 'theHollowKnight'],
      theFrozenTruth: ['theGodsLastWord'],
      theGodsLastWord: ['theSovereignsName', 'theStarsMemory'],
      theSovereignsName: ['theNamelessGrave'],
    };

    const toActivate = followUps[completedQuestId] || [];
    const quests = this.getQuests();
    toActivate.forEach(qId => {
      if (quests[qId] && !quests[qId].active && !quests[qId].completed) {
        quests[qId].active = true;
      }
    });
  }

  getActiveQuests() {
    const quests = this.getQuests();
    return Object.values(quests).filter(q => q.active && !q.completed);
  }

  getCompletedQuests() {
    const quests = this.getQuests();
    return Object.values(quests).filter(q => q.completed);
  }

  checkBossDefeat(bossId) {
    const bossToObjective = {
      korgathsEcho: { quest: 'theAshenPath', objective: 'defeat_korgaths_echo' },
      mawSerpent: { quest: 'theFrozenTruth', objective: 'defeat_maw_serpent' },
      echoOfAethon: { quest: 'theGodsLastWord', objective: 'defeat_echo_of_aethon' },
      memoryEater: { quest: 'theSovereignsName', objective: 'defeat_memory_eater' },
      voidSovereign: { quest: 'theSovereignsName', objective: 'defeat_void_sovereign' },
    };

    const mapping = bossToObjective[bossId];
    if (mapping) {
      this.completeObjective(mapping.quest, mapping.objective);
    }
  }

  checkZoneTransition(zoneId) {
    const zoneToObjective = {
      frozenMaw: { quest: 'theAshenPath', objective: 'open_frozen_maw' },
    };
    const mapping = zoneToObjective[zoneId];
    if (mapping) {
      this.completeObjective(mapping.quest, mapping.objective);
    }
  }
}
