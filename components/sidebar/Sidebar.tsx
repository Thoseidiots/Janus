
import React, { useState } from 'react';
import { Bars3Icon, XMarkIcon, BeakerIcon } from '@heroicons/react/24/solid';
import { Feature, FeatureID } from '../../App';

interface SidebarProps {
    features: Feature[];
    activeFeature: FeatureID;
    setActiveFeature: (feature: FeatureID) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ features, activeFeature, setActiveFeature }) => {
    const [isOpen, setIsOpen] = useState(false);

    const handleItemClick = (featureId: FeatureID) => {
        setActiveFeature(featureId);
        setIsOpen(false);
    };

    const sidebarContent = (
        <div className="flex flex-col h-full">
            <div className="flex items-center justify-between p-4 border-b border-gray-700">
                <div className="flex items-center gap-3">
                    <BeakerIcon className="h-8 w-8 text-indigo-400" />
                    <h1 className="text-xl font-bold text-white">Hyperparameter Studio</h1>
                </div>
                 <button onClick={() => setIsOpen(false)} className="lg:hidden p-1 text-gray-400 hover:text-white">
                    <XMarkIcon className="h-6 w-6" />
                </button>
            </div>
            <nav className="flex-1 p-4 space-y-2">
                {features.map((feature) => (
                    <button
                        key={feature.id}
                        onClick={() => handleItemClick(feature.id)}
                        className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg text-left transition-colors ${
                            activeFeature === feature.id
                                ? 'bg-indigo-600 text-white'
                                : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                        }`}
                    >
                        <feature.icon className="h-5 w-5" />
                        <span>{feature.name}</span>
                    </button>
                ))}
            </nav>
        </div>
    );


    return (
        <>
            {/* Mobile/Tablet Menu Button */}
            <button 
                onClick={() => setIsOpen(!isOpen)} 
                className="lg:hidden fixed top-4 left-4 z-30 p-2 bg-gray-800 rounded-md text-white"
            >
                <Bars3Icon className="h-6 w-6" />
            </button>

            {/* Mobile/Tablet Sidebar (Overlay) */}
            <div className={`fixed inset-0 z-20 bg-gray-900 bg-opacity-75 transition-opacity lg:hidden ${isOpen ? 'opacity-100' : 'opacity-0 pointer-events-none'}`} onClick={() => setIsOpen(false)}></div>
            <aside className={`fixed top-0 left-0 h-full w-64 bg-gray-800 border-r border-gray-700 transform transition-transform z-20 lg:hidden ${isOpen ? 'translate-x-0' : '-translate-x-full'}`}>
                {sidebarContent}
            </aside>
            
            {/* Desktop Sidebar */}
            <aside className="hidden lg:block w-64 bg-gray-800 border-r border-gray-700 flex-shrink-0">
                {sidebarContent}
            </aside>
        </>
    );
};
