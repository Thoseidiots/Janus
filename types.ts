export interface GroundingSource {
    title: string;
    uri: string;
}

export interface ChatMessage {
    role: 'user' | 'model';
    content: string;
    sources?: GroundingSource[];
}

export interface Scene {
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
    scenes: Scene[];
}

export interface TextContent {
    type: 'text';
    text: string;
    sources?: GroundingSource[];
}

interface ImageContent {
    type: 'image';
    url: string;
    prompt?: string;
}

interface VideoContent {
    type: 'video';
    url: string;
    prompt?: string;
}

interface AudioContent {
    type: 'audio';
    base64: string;
}

interface StoryboardContent {
    type: 'storyboard';
    storyboard: Storyboard;
}

interface VideoGeneratingContent {
    type: 'video_generating';
    prompt: string;
}

interface ErrorContent {
    type: 'error';
    message: string;
}

export type UnifiedMessageContent = TextContent | ImageContent | VideoContent | AudioContent | StoryboardContent | VideoGeneratingContent | ErrorContent;

export interface UnifiedChatMessage {
    id: string;
    role: 'user' | 'model' | 'system';
    content: UnifiedMessageContent;
}


// Types for the ThinkingBrain simulation
export type BrainRegionId = 'cortex' | 'limbic' | 'basal_ganglia' | 'cerebellum' | 'brainstem';

export interface BrainRegion {
    id: BrainRegionId;
    name: string;
    description: string;
    neuronCount: string;
    synapseCount: string;
    snnModel: string;
}

export interface CognitiveLogEntry {
    id: number;
    timestamp: string;
    system: string;
    message: string;
    region: BrainRegionId | null;
}

export interface CognitiveState {
    consciousnessLevel: number;
    arousal: number;
    attentionFocus: string | null;
    dominantOscillation: string;
    neuromodulators: {
        dopamine: number;
        serotonin: number;
        norepinephrine: number;
        acetylcholine: number;
    };
    metacognition: {
        confidence: number;
        selfAwareness: number;
    };
    emotion: {
        valence: number; // -1 to 1
        arousal: number; // 0 to 1
        type: string;
    };
    intrinsicMotivation: {
        curiosity: number; // 0 to 1
        informationHunger: number; // 0 to 1
        goal: string | null;
    };
}