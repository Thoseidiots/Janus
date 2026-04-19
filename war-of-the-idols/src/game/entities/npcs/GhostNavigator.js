import { NPC } from '../NPC.js';
import { NPC_DATA } from '../../../data/npcs.js';

export class GhostNavigator extends NPC {
  constructor() {
    super(NPC_DATA.ghostNavigator);
  }
}
