export const NPC_DATA = {
  ghostNavigator: {
    id: 'ghostNavigator',
    name: 'The Ghost Navigator',
    zone: 'ashfeld',
    position: { x: 3, y: 0, z: 3 },
    color: 0x88aaff,
    emissive: 0x4466ff,
    dialogueTrees: {
      initial: {
        id: 'initial',
        text: "You're still alive. Good. Most aren't, by the time they reach Ashfeld. I'm the Navigator. I've been waiting for someone like you — someone who still burns.",
        responses: [
          { text: "Who are you?", next: 'who_are_you' },
          { text: "What is this place?", next: 'what_is_this' },
          { text: "What do I need to do?", next: 'what_to_do' },
        ],
      },
      who_are_you: {
        id: 'who_are_you',
        text: "I died in the First Wound. Or near it. Time gets strange near the Void. What matters is I remember the way through. I've guided others before you. I don't tell them how many. I'm telling you: seventeen. None made it. You'll be different. You have to be.",
        responses: [
          { text: "Why do you keep trying?", next: 'why_keep_trying' },
          { text: "What happened to the others?", next: 'what_happened_others' },
          { text: "Tell me about the Void.", next: 'about_void' },
        ],
      },
      why_keep_trying: {
        id: 'why_keep_trying',
        text: "Because the alternative is watching everything end. I've seen what the Void does to a world. I've seen it happen to mine. I won't watch it happen to yours without trying.",
        responses: [
          { text: "I understand.", next: 'understood' },
          { text: "What do I need to do?", next: 'what_to_do' },
        ],
      },
      what_happened_others: {
        id: 'what_happened_others',
        text: "Some fell to the enemies. Some fell to the Hollowing. Some... chose the Void. When you stand before the Sovereign and it offers you what it offered them, remember this conversation. Remember that I'm still here, waiting.",
        responses: [
          { text: "I won't choose the Void.", next: 'understood' },
          { text: "What did it offer them?", next: 'sovereign_offer' },
        ],
      },
      sovereign_offer: {
        id: 'sovereign_offer',
        text: "Peace. An end to the struggle. The Void doesn't destroy — it simplifies. No more pain, no more loss, no more weight of existence. It's a seductive offer. Especially after everything you'll go through to reach it.",
        responses: [
          { text: "I'll remember.", next: 'understood' },
        ],
      },
      what_is_this: {
        id: 'what_is_this',
        text: "Ashfeld. Where the first Idol fell. Korgath burned so hot when he died that the land never stopped burning. The ash you're breathing? That's him. What's left of a god, scattered on the wind.",
        responses: [
          { text: "What do I need to do?", next: 'what_to_do' },
          { text: "Tell me about the Idols.", next: 'about_idols' },
        ],
      },
      about_idols: {
        id: 'about_idols',
        text: "Six beings who held reality together. Anchors in the Weave. When they fell, the Weave tore. The Void poured through. The last one — the Sovereign — didn't fall. It chose. That's the wound we're trying to close.",
        responses: [
          { text: "What do I need to do?", next: 'what_to_do' },
        ],
      },
      about_void: {
        id: 'about_void',
        text: "The Void is the absence of everything. Not darkness — darkness is something. The Void is nothing given hunger. The Sovereign is the Void given will. Don't let it touch you more than it has to. Each touch takes something you won't get back.",
        responses: [
          { text: "What do I need to do?", next: 'what_to_do' },
        ],
      },
      what_to_do: {
        id: 'what_to_do',
        text: "Find the shrine. Rest. Then find Korgath's Echo and end it. The Echo is what's left of Korgath's will — it's in agony, and it's taking that agony out on everything around it. Ending it is mercy as much as necessity. After that, I'll show you the way to the Frozen Maw.",
        responses: [
          { text: "I'm ready.", next: 'ready' },
          { text: "Tell me more first.", next: 'tell_more' },
        ],
      },
      tell_more: {
        id: 'tell_more',
        text: "There are six zones between here and the Void Core. Each one is the remnant of a fallen Idol. Each one has a boss — an Echo, a fragment, a memory. Defeat them all and you'll have what you need to face the Sovereign. Or you'll be dead. One of the two.",
        responses: [
          { text: "I'm ready.", next: 'ready' },
        ],
      },
      ready: {
        id: 'ready',
        text: "Good. The shrine is north of here. Rest there — it'll anchor your soul to this place. If you die, you'll return there. And you will die. Everyone does. The question is whether you get back up.",
        responses: [
          { text: "Thank you.", next: 'farewell' },
        ],
      },
      understood: {
        id: 'understood',
        text: "Good. Now go. The ash doesn't wait.",
        responses: [
          { text: "Farewell.", next: 'farewell' },
        ],
      },
      farewell: {
        id: 'farewell',
        text: "I'll be watching. I always watch. It's all I can do.",
        responses: [],
        end: true,
      },
    },
  },

  echoOfAethonNPC: {
    id: 'echoOfAethonNPC',
    name: 'Echo of Aethon (Remnant)',
    zone: 'ruinsOfAethon',
    position: { x: -8, y: 0, z: -8 },
    color: 0x8866ff,
    emissive: 0x6644ff,
    dialogueTrees: {
      initial: {
        id: 'initial',
        text: "...you are not... one of mine... I remember all my children... you are new... or very old... time is difficult here...",
        responses: [
          { text: "What are you?", next: 'what_are_you' },
          { text: "I need information about the Sovereign.", next: 'about_sovereign' },
          { text: "Where is the archive?", next: 'archive_location' },
        ],
      },
      what_are_you: {
        id: 'what_are_you',
        text: "I am... what remains of Aethon. The god-city. I held all knowledge. Every word ever written. Every thought ever thought. Now I hold... fragments. Echoes of echoes. It is... diminishing.",
        responses: [
          { text: "I'm sorry.", next: 'sympathy' },
          { text: "I need the archive.", next: 'archive_location' },
        ],
      },
      sympathy: {
        id: 'sympathy',
        text: "Do not be sorry. I chose to remember. The others let go. Korgath burned out. Ysolde froze. The Crystalline dissolved. I held on. Perhaps... holding on was the mistake. But I cannot stop now. The knowledge must survive.",
        responses: [
          { text: "Help me stop the Sovereign.", next: 'help_sovereign' },
        ],
      },
      about_sovereign: {
        id: 'about_sovereign',
        text: "The Sovereign... I knew it before. Before the choice. It was the most beautiful of us. The most complete. When it chose the Void, I felt it like a death. Because it was. The Sovereign that exists now is not the one I knew.",
        responses: [
          { text: "Can it be stopped?", next: 'can_be_stopped' },
          { text: "Where is the archive?", next: 'archive_location' },
        ],
      },
      can_be_stopped: {
        id: 'can_be_stopped',
        text: "It can be... redirected. Unmade. Named. The Sovereign has a true name — all Idols do. Names are anchors. Speak the true name and the anchor holds, even in the Void. Find Kael. He wrote it down. He always wrote everything down.",
        responses: [
          { text: "Where is Kael?", next: 'kael_location' },
        ],
      },
      kael_location: {
        id: 'kael_location',
        text: "The Crystalline Nebula. He fled there when Aethon fell. He thought the Sovereign wouldn't look for him in a place made of dissolved thought. He was... partially right. It hasn't found him yet. Go quickly.",
        responses: [
          { text: "Thank you.", next: 'farewell' },
        ],
      },
      archive_location: {
        id: 'archive_location',
        text: "The archive is at the heart of the ruins. The Echo that guards it — the hostile one, not me — is what remains of my will to protect the knowledge. It does not distinguish between friend and enemy anymore. You will have to fight through it.",
        responses: [
          { text: "I understand.", next: 'farewell' },
        ],
      },
      help_sovereign: {
        id: 'help_sovereign',
        text: "Then go to the archive. Take what you find. And when you reach the Sovereign... remember that it was once something worth saving. That knowledge might matter, at the end.",
        responses: [
          { text: "I will.", next: 'farewell' },
        ],
      },
      farewell: {
        id: 'farewell',
        text: "Go. I will... hold on a little longer. For you.",
        responses: [],
        end: true,
      },
    },
  },

  archivistKael: {
    id: 'archivistKael',
    name: 'Archivist Kael',
    zone: 'crystallineNebula',
    position: { x: 10, y: 0, z: 10 },
    color: 0x00ffcc,
    emissive: 0x00ddaa,
    dialogueTrees: {
      initial: {
        id: 'initial',
        text: "Oh. Oh, you're real. I've been talking to thought-constructs for so long I wasn't sure I'd recognize a real person. You're real, yes? Touch something. Does it feel solid? Good. Good. I'm Kael. I've been hiding here for approximately three thousand years.",
        responses: [
          { text: "I need the Sovereign's true name.", next: 'true_name_request' },
          { text: "Three thousand years?", next: 'three_thousand' },
          { text: "Are you alright?", next: 'are_you_alright' },
        ],
      },
      three_thousand: {
        id: 'three_thousand',
        text: "Give or take a century. Time is unreliable in the Nebula. I've been writing. Documenting. Preserving. The Sovereign wants to erase everything — I've been making sure there's a record. Someone has to.",
        responses: [
          { text: "I need the Sovereign's true name.", next: 'true_name_request' },
        ],
      },
      are_you_alright: {
        id: 'are_you_alright',
        text: "Honestly? No. I'm a three-thousand-year-old archivist hiding in a nebula made of dissolved god-thoughts, waiting for someone to come kill the thing that destroyed my civilization. But I'm functional. That's something.",
        responses: [
          { text: "I need the Sovereign's true name.", next: 'true_name_request' },
          { text: "I'm sorry about Aethon.", next: 'sorry_aethon' },
        ],
      },
      sorry_aethon: {
        id: 'sorry_aethon',
        text: "Thank you. Aethon was... it was everything. Every book ever written. Every question ever asked. When it fell, I felt like the universe had forgotten how to think. I've been trying to remember for it ever since.",
        responses: [
          { text: "I need the Sovereign's true name.", next: 'true_name_request' },
        ],
      },
      true_name_request: {
        id: 'true_name_request',
        text: "I knew you'd ask that. I've been waiting for someone to ask that. I have it — I've always had it. I wrote it down the day Aethon fell, when the Sovereign made its choice. I thought: someone will need this. Someday. Someone will come.",
        responses: [
          { text: "Tell me.", next: 'tell_name' },
          { text: "Is it dangerous to know?", next: 'dangerous_to_know' },
        ],
      },
      dangerous_to_know: {
        id: 'dangerous_to_know',
        text: "Extremely. The Sovereign will feel it the moment you speak it. It will know you know. It will come for you — or rather, you'll be going to it, which amounts to the same thing. But yes. Knowing the name changes you. You'll feel it. Like a weight you can't put down.",
        responses: [
          { text: "Tell me anyway.", next: 'tell_name' },
        ],
      },
      tell_name: {
        id: 'tell_name',
        text: "The Sovereign's true name is... VAEL'THERON. The Unanchored. The One Who Chose. Speak it in the Void Core, at the moment of confrontation, and the Sovereign's true nature will be exposed. Its defenses will crack. But only if you've collected all the lore — the name needs context to have power. The Sovereign has spent three thousand years making people forget it. Don't forget it.",
        responses: [
          { text: "VAEL'THERON. I won't forget.", next: 'wont_forget' },
        ],
      },
      wont_forget: {
        id: 'wont_forget',
        text: "Good. Now go. And if you survive — if you actually survive — come back and tell me how it ends. I've been writing this story for three thousand years. I'd like to know the ending.",
        responses: [
          { text: "I'll come back.", next: 'farewell' },
          { text: "I can't promise that.", next: 'cant_promise' },
        ],
      },
      cant_promise: {
        id: 'cant_promise',
        text: "No. I suppose you can't. That's honest, at least. The others always promised. Go. Do what you can.",
        responses: [
          { text: "Farewell, Kael.", next: 'farewell' },
        ],
      },
      farewell: {
        id: 'farewell',
        text: "Farewell. And... thank you. For coming. For trying. Whatever happens.",
        responses: [],
        end: true,
      },
    },
  },

  voidMerchant: {
    id: 'voidMerchant',
    name: 'The Void Merchant',
    zone: 'sovereignsVeil',
    position: { x: 12, y: 0, z: -12 },
    color: 0xff00ff,
    emissive: 0xcc00cc,
    isMerchant: true,
    inventory: ['voidStaff', 'sovereignsMantle', 'voidOrb', 'trueNameFragment', 'voidTear', 'hollowingCure'],
    dialogueTrees: {
      initial: {
        id: 'initial',
        text: "Ah. Another one. You have the look of someone who's been through several zones of existential horror and is still, somehow, standing. I respect that. I trade in things that matter. Interested?",
        responses: [
          { text: "What are you?", next: 'what_are_you' },
          { text: "Show me your wares.", next: 'show_wares' },
          { text: "Why are you helping me?", next: 'why_helping' },
        ],
      },
      what_are_you: {
        id: 'what_are_you',
        text: "A merchant. What else matters? I exist in the spaces between categories. Not void, not weave, not god, not mortal. I am the transaction. The exchange. The moment when something changes hands and both parties walk away changed.",
        responses: [
          { text: "Show me your wares.", next: 'show_wares' },
          { text: "Why are you helping me?", next: 'why_helping' },
        ],
      },
      why_helping: {
        id: 'why_helping',
        text: "Because your success is more profitable than your failure. If the Sovereign wins, there's no one left to trade with. I have a vested interest in the continued existence of existence. Also, I find you interesting. That's rarer than you'd think.",
        responses: [
          { text: "Show me your wares.", next: 'show_wares' },
        ],
      },
      show_wares: {
        id: 'show_wares',
        text: "I have things you won't find anywhere else. Things that matter. The price is souls — the currency of the dead, which is the only currency that holds value near the Void. Browse freely.",
        responses: [
          { text: "Let me see what you have.", next: 'open_shop', openShop: true },
          { text: "Maybe later.", next: 'farewell' },
        ],
      },
      open_shop: {
        id: 'open_shop',
        text: "Take your time. I'm not going anywhere. I never go anywhere. I simply... am, wherever commerce is needed.",
        responses: [],
        end: true,
        openShop: true,
      },
      farewell: {
        id: 'farewell',
        text: "Come back when you need something. I'll be here. I'm always here.",
        responses: [],
        end: true,
      },
    },
  },
};
