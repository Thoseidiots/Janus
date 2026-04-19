export class DialogueSystem {
  constructor(gameState) {
    this.gameState = gameState;
    this.currentDialogue = null;
    this.currentNode = null;
    this.displayedText = '';
    this.typingIndex = 0;
    this.typingSpeed = 30; // ms per character
    this.typingTimer = null;
    this.isTyping = false;
    this.onDialogueEnd = null;
    this.onShopOpen = null;
  }

  startDialogue(npcData, onEnd = null, onShopOpen = null) {
    this.currentDialogue = npcData;
    this.onDialogueEnd = onEnd;
    this.onShopOpen = onShopOpen;
    this.gameState.activeDialogue = {
      npcId: npcData.id,
      npcName: npcData.name,
      nodeId: 'initial',
      displayedText: '',
      responses: [],
      isTyping: true,
    };
    this._loadNode('initial');
  }

  _loadNode(nodeId) {
    if (!this.currentDialogue) return;
    const tree = this.currentDialogue.dialogueTrees;
    const node = tree[nodeId];
    if (!node) return;

    this.currentNode = node;
    this.displayedText = '';
    this.typingIndex = 0;
    this.isTyping = true;

    const state = this.gameState.activeDialogue;
    state.nodeId = nodeId;
    state.displayedText = '';
    state.responses = [];
    state.isTyping = true;
    state.isEnd = node.end || false;

    this._startTyping(node.text);
  }

  _startTyping(text) {
    if (this.typingTimer) clearInterval(this.typingTimer);
    this.typingIndex = 0;
    this.typingTimer = setInterval(() => {
      if (this.typingIndex < text.length) {
        this.displayedText += text[this.typingIndex];
        this.typingIndex++;
        if (this.gameState.activeDialogue) {
          this.gameState.activeDialogue.displayedText = this.displayedText;
        }
      } else {
        clearInterval(this.typingTimer);
        this.typingTimer = null;
        this.isTyping = false;
        if (this.gameState.activeDialogue) {
          this.gameState.activeDialogue.isTyping = false;
          this.gameState.activeDialogue.responses = this.currentNode?.responses || [];
        }
      }
    }, this.typingSpeed);
  }

  skipTyping() {
    if (this.isTyping && this.currentNode) {
      if (this.typingTimer) clearInterval(this.typingTimer);
      this.typingTimer = null;
      this.displayedText = this.currentNode.text;
      this.isTyping = false;
      if (this.gameState.activeDialogue) {
        this.gameState.activeDialogue.displayedText = this.displayedText;
        this.gameState.activeDialogue.isTyping = false;
        this.gameState.activeDialogue.responses = this.currentNode.responses || [];
      }
    }
  }

  selectResponse(responseIndex) {
    if (!this.currentNode || this.isTyping) return;
    const responses = this.currentNode.responses;
    if (!responses || responseIndex >= responses.length) return;

    const response = responses[responseIndex];

    if (response.openShop && this.onShopOpen) {
      this.onShopOpen();
      return;
    }

    if (response.next) {
      this._loadNode(response.next);
    } else {
      this.endDialogue();
    }
  }

  endDialogue() {
    if (this.typingTimer) clearInterval(this.typingTimer);
    this.typingTimer = null;
    this.currentDialogue = null;
    this.currentNode = null;
    this.gameState.activeDialogue = null;
    if (this.onDialogueEnd) {
      this.onDialogueEnd();
      this.onDialogueEnd = null;
    }
  }

  isActive() {
    return this.gameState.activeDialogue !== null;
  }

  update(deltaTime) {
    // Dialogue system is event-driven, no per-frame update needed
  }
}
