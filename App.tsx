import React from 'react';
import { JanusAGIControl } from './components/features/JanusAGIControl';

const App: React.FC = () => {
    return (
        <div className="h-screen bg-gray-900 text-gray-100 font-sans">
            <main className="h-full">
                <JanusAGIControl />
            </main>
        </div>
    );
};

export default App;