import React, { useRef, useEffect, FC } from 'react';
import { Card } from '../../common/Card';
import { Spinner } from '../../common/Spinner';
import { ProgressBar } from '../../common/ProgressBar';
import { BrainVisualizer } from './BrainVisualizer';
import { BrainRegion, BrainRegionId, CognitiveLogEntry, CognitiveState } from '../../../types';
import { FireIcon, HeartIcon } from '@heroicons/react/24/solid';

interface CognitiveDashboardProps {
    cognitiveLogs: CognitiveLogEntry[];
    cognitiveState: CognitiveState;
    isThinking: boolean;
    activeRegions: Set<BrainRegionId>;
    selectedRegion: BrainRegion | null;
    onRegionSelect: (regionId: BrainRegionId) => void;
}

export const CognitiveDashboard: FC<CognitiveDashboardProps> = ({
    cognitiveLogs,
    cognitiveState,
    isThinking,
    activeRegions,
    selectedRegion,
    onRegionSelect
}) => {
    const cognitiveLogRef = useRef<HTMLDivElement>(null);
    
    useEffect(() => {
        if (cognitiveLogRef.current) {
            cognitiveLogRef.current.scrollTo({ top: cognitiveLogRef.current.scrollHeight, behavior: 'smooth' });
        }
    }, [cognitiveLogs]);

    return (
         <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 h-full">
            <div className="lg:col-span-3 flex flex-col min-h-0">
                <h3 className="text-xl font-semibold mb-2 flex-shrink-0">Live Cognitive Log</h3>
                <div ref={cognitiveLogRef} className="flex-grow bg-gray-900/50 p-3 rounded-lg text-sm font-mono space-y-1 overflow-y-auto">
                    {cognitiveLogs.map(log => (<p key={log.id}><span className="text-gray-500">{log.timestamp}</span> <span className="text-cyan-400">[{log.system}]</span> {log.message}</p>))}
                    {isThinking && <div className="flex justify-center pt-2"><Spinner size="sm" /></div>}
                </div>
            </div>
            <div className="lg:col-span-2 flex flex-col gap-4 overflow-y-auto pr-2">
                <Card><h3 className="text-xl font-semibold mb-4 text-center">Brain State</h3><BrainVisualizer activeRegions={activeRegions} onRegionClick={onRegionSelect} /></Card>
                <Card><h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><FireIcon className="h-6 w-6 text-orange-400" /> Intrinsic Motivation</h3><div className="space-y-4 text-sm"><ProgressBar value={cognitiveState.intrinsicMotivation.curiosity} color="bg-green-500" label="Curiosity Drive" /><ProgressBar value={cognitiveState.intrinsicMotivation.informationHunger} color="bg-yellow-500" label="Information Hunger" /></div></Card>
                <Card><h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><HeartIcon className="h-6 w-6 text-red-400" /> Emotional State</h3><div className="space-y-4 text-sm"><ProgressBar value={(cognitiveState.emotion.valence + 1) / 2} color="bg-blue-500" label={`Valence: ${cognitiveState.emotion.valence.toFixed(2)}`} /><ProgressBar value={cognitiveState.emotion.arousal} color="bg-red-500" label={`Arousal: ${cognitiveState.emotion.arousal.toFixed(2)}`} /></div></Card>
                {selectedRegion && <Card className="flex-shrink-0 animate-fadeIn"><h3 className="text-lg font-semibold">{selectedRegion.name}</h3><p className="text-xs text-gray-400">{selectedRegion.description}</p><p className="text-xs mt-2"><strong>Neurons:</strong> {selectedRegion.neuronCount}</p><p className="text-xs"><strong>Synapses:</strong> {selectedRegion.synapseCount}</p></Card>}
            </div>
        </div>
    );
};