import React from 'react';
import { BrainRegionId } from '../../../types';

interface BrainVisualizerProps {
    activeRegions: Set<BrainRegionId>;
    onRegionClick: (regionId: BrainRegionId) => void;
}

export const BrainVisualizer: React.FC<BrainVisualizerProps> = React.memo(({ activeRegions, onRegionClick }) => {
    const getRegionClasses = (id: BrainRegionId) => {
        const base = "transition-all duration-300 rounded-lg cursor-pointer hover:stroke-indigo-400 hover:stroke-2";
        const active = activeRegions.has(id) ? "fill-indigo-500/50 stroke-indigo-300 animate-pulse-glow" : "fill-gray-700/50 stroke-gray-500";
        return `${base} ${active}`;
    };

    return (
        <div className="relative w-full max-w-sm mx-auto aspect-square">
            <svg viewBox="0 0 100 100">
                {/* Cortex */}
                <path d="M20 30 C 10 40, 10 60, 25 80 S 40 95, 60 85 S 90 70, 90 50 S 80 15, 60 20 S 30 20, 20 30 Z" 
                    className={getRegionClasses('cortex')} onClick={() => onRegionClick('cortex')} />
                {/* Limbic System */}
                <circle cx="45" cy="55" r="12" className={getRegionClasses('limbic')} onClick={() => onRegionClick('limbic')} />
                {/* Basal Ganglia */}
                <circle cx="48" cy="65" r="8" className={getRegionClasses('basal_ganglia')} onClick={() => onRegionClick('basal_ganglia')} />
                {/* Cerebellum */}
                <path d="M30 80 C 40 90, 60 90, 70 80 L 65 95 L 35 95 Z" 
                    className={getRegionClasses('cerebellum')} onClick={() => onRegionClick('cerebellum')} />
                {/* Brainstem */}
                <rect x="45" y="75" width="10" height="20" rx="3" 
                    className={getRegionClasses('brainstem')} onClick={() => onRegionClick('brainstem')} />
            </svg>
        </div>
    );
});