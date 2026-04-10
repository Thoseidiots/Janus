
export interface ChatMessage {
    role: 'user' | 'model';
    content: string;
    sources?: GroundingSource[];
}

export interface GroundingSource {
    title: string;
    uri: string;
}

export interface StoryboardScene {
    sceneNumber: number;
    startTime: string;
    endTime: string;
    keyframeDescription: string;
    sceneDescription: string;
    keyMoments: string[];
    keyframeImageUrl?: string;
}

export interface Storyboard {
    overallSummary: string;
    scenes: StoryboardScene[];
}

// Types for the new Unified Studio
export type UnifiedMessageContent =
    | { type: 'text'; text: string; sources?: GroundingSource[] }
    | { type: 'image'; url: string; prompt?: string }
    | { type: 'video'; url: string; prompt?: string }
    | { type: 'audio'; base64: string }
    | { type: 'storyboard'; storyboard: Storyboard }
    | { type: 'error'; message: string }
    | { type: 'video_generating'; prompt: string };

export interface UnifiedChatMessage {
    role: 'user' | 'model' | 'system';
    content: UnifiedMessageContent;
    id: string;
}


// Add window declaration for aistudio
declare global {
  // Fix: Augment the existing global AIStudio interface instead of re-declaring window.aistudio.
  // This resolves the conflict with another global declaration.
  interface AIStudio {
    hasSelectedApiKey: () => Promise<boolean>;
    openSelectKey: () => Promise<void>;
  }
}