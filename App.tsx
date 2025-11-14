
import React, { useState, FC, ElementType, useCallback } from 'react';
// Import all feature components
import { JanusAGIControl } from './components/features/JanusAGIControl';
import { HyperparameterGenerator } from './components/features/HyperparameterGenerator';
import { Chatbot } from './components/features/Chatbot';
import { ImageTools } from './components/features/ImageTools';
import { VideoTools } from './components/features/VideoTools';
import { VoiceAssistant } from './components/features/VoiceAssistant';
import { GroundedSearch } from './components/features/GroundedSearch';
import { AnalysisTools } from './components/features/AnalysisTools';
import { TextToSpeechTool } from './components/features/TextToSpeechTool';
import { Storyboarder } from './components/features/Storyboarder';
import { SecureLauncher } from './components/features/SecureLauncher';
import { AgiCoreConsole } from './components/features/AgiCoreConsole';

// Import icons for the sidebar
import {
    CpuChipIcon,
    AdjustmentsHorizontalIcon,
    ChatBubbleLeftRightIcon,
    PhotoIcon,
    VideoCameraIcon,
    MicrophoneIcon,
    GlobeAltIcon,
    DocumentChartBarIcon,
    SpeakerWaveIcon,
    FilmIcon,
    ShieldCheckIcon,
    CommandLineIcon,
    CubeTransparentIcon
} from '@heroicons/react/24/outline';

// Define the structure for a tool
interface Tool {
    name: string;
    component: FC;
    icon: ElementType;
}

// Create a list of all available tools
const tools: Tool[] = [
    { name: 'Janus AGI', component: JanusAGIControl, icon: CpuChipIcon },
    { name: 'Hyperparameter Gen', component: HyperparameterGenerator, icon: AdjustmentsHorizontalIcon },
    { name: 'Tuning Chatbot', component: Chatbot, icon: ChatBubbleLeftRightIcon },
    { name: 'Grounded Search', component: GroundedSearch, icon: GlobeAltIcon },
    { name: 'Image Studio', component: ImageTools, icon: PhotoIcon },
    { name: 'Video Studio', component: VideoTools, icon: VideoCameraIcon },
    { name: 'Storyboarder', component: Storyboarder, icon: FilmIcon },
    { name: 'Content Analyzer', component: AnalysisTools, icon: DocumentChartBarIcon },
    { name: 'Voice Assistant', component: VoiceAssistant, icon: MicrophoneIcon },
    { name: 'Text-to-Speech', component: TextToSpeechTool, icon: SpeakerWaveIcon },
    { name: 'Secure Launcher', component: SecureLauncher, icon: ShieldCheckIcon },
    { name: 'AGI Core Console', component: AgiCoreConsole, icon: CommandLineIcon },
];


const App: React.FC = () => {
    // State to track the currently selected tool
    const [activeToolName, setActiveToolName] = useState<string>(tools[0].name);
    // Keep track of which tools have been mounted to preserve their state
    const [mountedTools, setMountedTools] = useState<Set<string>>(() => new Set([tools[0].name]));

    const handleToolSelect = useCallback((name: string) => {
        setActiveToolName(name);
        // Mount the tool if it hasn't been mounted before
        if (!mountedTools.has(name)) {
            setMountedTools(prev => new Set(prev).add(name));
        }
    }, [mountedTools]);


    return (
        <div className="h-screen bg-gray-900 text-gray-100 font-sans flex overflow-hidden">
            {/* Sidebar Navigation */}
            <aside className="w-64 bg-gray-900/70 backdrop-blur-sm border-r border-gray-800 flex flex-col flex-shrink-0">
                <div className="h-16 flex items-center justify-center border-b border-gray-800 flex-shrink-0">
                    <CubeTransparentIcon className="h-8 w-8 text-indigo-400" />
                    <h1 className="text-xl font-bold ml-2">ThinkingBrain</h1>
                </div>
                <nav className="flex-grow overflow-y-auto">
                    <ul className="p-2 space-y-1">
                        {tools.map((tool) => (
                            <li key={tool.name}>
                                <button
                                    onClick={() => handleToolSelect(tool.name)}
                                    className={`w-full flex items-center gap-3 px-3 py-2 text-sm rounded-md transition-colors ${
                                        activeToolName === tool.name
                                            ? 'bg-indigo-600 text-white font-semibold'
                                            : 'text-gray-300 hover:bg-gray-800'
                                    }`}
                                >
                                    <tool.icon className="h-5 w-5" />
                                    <span>{tool.name}</span>
                                </button>
                            </li>
                        ))}
                    </ul>
                </nav>
            </aside>

            {/* Main Content Area */}
            <main className="flex-1 overflow-y-auto">
                <div className="p-4 sm:p-6 lg:p-8 h-full">
                    {/* 
                      Render all mounted tools, but only show the active one.
                      This preserves the component state when switching between tools.
                    */}
                    {tools.map(({ name, component: Component }) => {
                        if (!mountedTools.has(name)) {
                            return null;
                        }
                        return (
                            <div
                                key={name}
                                className={`${activeToolName === name ? 'block' : 'hidden'} h-full w-full`}
                                role="tabpanel"
                                aria-hidden={activeToolName !== name}
                            >
                                <Component />
                            </div>
                        );
                    })}
                </div>
            </main>
        </div>
    );
};

export default App;
