
import React, { useState, useCallback, useMemo } from 'react';
import { Sidebar } from './components/sidebar/Sidebar';
import { JanusHub } from './components/features/JanusHub';
import { UnbiasedAgiConsole } from './components/features/UnbiasedAgiConsole';

import { 
    CubeTransparentIcon,
    AcademicCapIcon
} from '@heroicons/react/24/outline';

export type FeatureID = 'janus_hub' | 'unbiased_agi';

export interface Feature {
    id: FeatureID;
    name: string;
    icon: React.ElementType;
}

const App: React.FC = () => {
    const [activeFeature, setActiveFeature] = useState<FeatureID>('janus_hub');

    const features: Feature[] = useMemo(() => [
        { id: 'janus_hub', name: 'Janus Control Hub', icon: CubeTransparentIcon },
        { id: 'unbiased_agi', name: 'Unbiased AGI Console', icon: AcademicCapIcon },
    ], []);

    const renderActiveFeature = useCallback(() => {
        switch (activeFeature) {
            case 'janus_hub':
                return <JanusHub />;
            case 'unbiased_agi':
                return <UnbiasedAgiConsole />;
            default:
                return <JanusHub />;
        }
    }, [activeFeature]);

    return (
        <div className="flex h-screen bg-gray-900 text-gray-100 font-sans">
            <Sidebar features={features} activeFeature={activeFeature} setActiveFeature={setActiveFeature} />
            <main className="flex-1 overflow-y-auto p-4 sm:p-6 lg:p-8">
                {renderActiveFeature()}
            </main>
        </div>
    );
};

export default App;
