import { BrainRegion, CognitiveState, BrainRegionId } from '../../../types';

// --- ThinkingBrain Simulation Constants & Types ---
export type CascadeStep = { system: string; message: string; region: BrainRegionId | null; delay: number };

export const BRAIN_REGIONS: Record<BrainRegionId, BrainRegion> = {
    cortex: { id: 'cortex', name: 'Cerebral Cortex', description: 'Center of higher-level processing, planning, and language.', neuronCount: '16 Billion', synapseCount: '60-240 Trillion', snnModel: 'Izhikevich, AdEx' },
    limbic: { id: 'limbic', name: 'Limbic System', description: 'Infrastructure for memory (Hippocampus), emotion (Amygdala), and regulation (Thalamus).', neuronCount: '~250 Million', synapseCount: '~3-8 Trillion', snnModel: 'AdEx, LIF' },
    basal_ganglia: { id: 'basal_ganglia', name: 'Basal Ganglia', description: 'Governs action selection and habit automation via "Go" and "NoGo" pathways.', neuronCount: '~150 Million', synapseCount: '~1-2 Trillion', snnModel: 'Izhikevich (Bistable)' },
    cerebellum: { id: 'cerebellum', name: 'Cerebellum', description: 'Focuses on real-time motor control, balance, and error correction.', neuronCount: '69 Billion', synapseCount: '40-150 Trillion', snnModel: 'Hodgkin-Huxley, LIF' },
    brainstem: { id: 'brainstem', name: 'Brainstem', description: 'Manages vital functions and is the source of key neuromodulators.', neuronCount: '~100 Million', synapseCount: '~0.5-1 Trillion', snnModel: 'Specialized' },
};

export const INITIAL_STATE: CognitiveState = {
    consciousnessLevel: 0.7, 
    arousal: 0.8, 
    attentionFocus: 'External environment', 
    dominantOscillation: 'Beta',
    neuromodulators: { 
        dopamine: 0.5, 
        serotonin: 0.5, 
        norepinephrine: 0.6, 
        acetylcholine: 0.7 
    },
    metacognition: { 
        confidence: 0.8, 
        selfAwareness: 0.6 
    },
    emotion: { 
        valence: 0.0, 
        arousal: 0.5, 
        type: 'Neutral' 
    },
    intrinsicMotivation: { 
        curiosity: 0.3, 
        informationHunger: 0.2, 
        goal: null 
    },
};