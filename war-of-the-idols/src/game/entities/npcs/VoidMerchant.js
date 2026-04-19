import { NPC } from '../NPC.js';
import { NPC_DATA } from '../../../data/npcs.js';

export class VoidMerchant extends NPC {
  constructor() {
    super(NPC_DATA.voidMerchant);
  }
}
