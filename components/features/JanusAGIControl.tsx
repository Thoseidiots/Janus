import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Tab } from '@headlessui/react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { AudioPlayer } from '../common/AudioPlayer';
import { BrainRegion, BrainRegionId, CognitiveLogEntry, CognitiveState, UnifiedChatMessage, UnifiedMessageContent, ChatMessage, TextContent } from '../../types';
import * as geminiService from '../../services/geminiService';
import { fileToBase64 } from '../../utils/fileUtils';
import { useVeo, VeoStatus } from '../../hooks/useVeo';
import ReactMarkdown from 'react-markdown';

import {
    CheckCircleIcon, XCircleIcon, ShieldCheckIcon, RocketLaunchIcon,
    PaperAirplaneIcon, UserCircleIcon, PaperClipIcon, GlobeAltIcon,
    PhotoIcon, VideoCameraIcon, FilmIcon, DocumentIcon,
    FireIcon, HeartIcon
} from '@heroicons/react/24/solid';
import { CpuChipIcon } from '@heroicons/react/24/outline';


// --- ThinkingBrain Simulation Constants & Types ---
type CascadeStep = { system: string; message: string; region: BrainRegionId | null; delay: number };

const BRAIN_REGIONS: Record<BrainRegionId, BrainRegion> = {
    cortex: { id: 'cortex', name: 'Cerebral Cortex', description: 'Center of higher-level processing, planning, and language.', neuronCount: '16 Billion', synapseCount: '60-240 Trillion', snnModel: 'Izhikevich, AdEx' },
    limbic: { id: 'limbic', name: 'Limbic System', description: 'Infrastructure for memory (Hippocampus), emotion (Amygdala), and regulation (Thalamus).', neuronCount: '~250 Million', synapseCount: '~3-8 Trillion', snnModel: 'AdEx, LIF' },
    basal_ganglia: { id: 'basal_ganglia', name: 'Basal Ganglia', description: 'Governs action selection and habit automation via "Go" and "NoGo" pathways.', neuronCount: '~150 Million', synapseCount: '~1-2 Trillion', snnModel: 'Izhikevich (Bistable)' },
    cerebellum: { id: 'cerebellum', name: 'Cerebellum', description: 'Focuses on real-time motor control, balance, and error correction.', neuronCount: '69 Billion', synapseCount: '40-150 Trillion', snnModel: 'Hodgkin-Huxley, LIF' },
    brainstem: { id: 'brainstem', name: 'Brainstem', description: 'Manages vital functions and is the source of key neuromodulators.', neuronCount: '~100 Million', synapseCount: '~0.5-1 Trillion', snnModel: 'Specialized' },
};

const INITIAL_STATE: CognitiveState = {
    consciousnessLevel: 0.7, arousal: 0.8, attentionFocus: 'External environment', dominantOscillation: 'Beta',
    neuromodulators: { dopamine: 0.5, serotonin: 0.5, norepinephrine: 0.6, acetylcholine: 0.7 },
    metacognition: { confidence: 0.8, selfAwareness: 0.6 },
    emotion: { valence: 0.0, arousal: 0.5, type: 'Neutral' },
    intrinsicMotivation: { curiosity: 0.3, informationHunger: 0.2, goal: null },
};


// --- Sub-Components ---

const BrainVisualizer: React.FC<{ activeRegions: Set<BrainRegionId>, onRegionClick: (regionId: BrainRegionId) => void }> = ({ activeRegions, onRegionClick }) => {
    const getRegionClasses = (id: BrainRegionId) => {
        const base = "transition-all duration-300 rounded-lg cursor-pointer hover:stroke-indigo-400 hover:stroke-2";
        const active = activeRegions.has(id) ? "fill-indigo-500/50 stroke-indigo-300 animate-pulse-glow" : "fill-gray-700/50 stroke-gray-500";
        return `${base} ${active}`;
    };

    return (
        <div className="relative w-full max-w-sm mx-auto aspect-square">
            <svg viewBox="0 0 100 100">
                <path d="M20 30 C 10 40, 10 60, 25 80 S 40 95, 60 85 S 90 70, 90 50 S 80 15, 60 20 S 30 20, 20 30 Z" 
                    className={getRegionClasses('cortex')} onClick={() => onRegionClick('cortex')} />
                <circle cx="45" cy="55" r="12" className={getRegionClasses('limbic')} onClick={() => onRegionClick('limbic')} />
                <circle cx="48" cy="65" r="8" className={getRegionClasses('basal_ganglia')} onClick={() => onRegionClick('basal_ganglia')} />
                <path d="M30 80 C 40 90, 60 90, 70 80 L 65 95 L 35 95 Z" 
                    className={getRegionClasses('cerebellum')} onClick={() => onRegionClick('cerebellum')} />
                <rect x="45" y="75" width="10" height="20" rx="3" 
                    className={getRegionClasses('brainstem')} onClick={() => onRegionClick('brainstem')} />
            </svg>
        </div>
    );
};

const ProgressBar: React.FC<{ value: number; color: string; label: string }> = ({ value, color, label }) => (
     <div>
        <label className="text-sm text-gray-400 mb-1 block">{label}</label>
        <div className="w-full bg-gray-700 rounded-full h-2.5">
            <div className={`${color} h-2.5 rounded-full transition-all duration-500`} style={{ width: `${value * 100}%` }}></div>
        </div>
    </div>
);

const getHostname = (uri: string | undefined): string => {
    if (!uri) return 'source';
    try {
        return new URL(uri).hostname.replace('www.', '');
    } catch (e) {
        return uri.length > 30 ? uri.slice(0, 27) + '...' : uri;
    }
};

const MessageContent: React.FC<{ message: UnifiedChatMessage }> = ({ message }) => {
    const { content } = message;

    switch (content.type) {
        case 'text':
            return (
                <div className="prose prose-invert max-w-none prose-p:my-2 prose-headings:my-3">
                    <ReactMarkdown>{content.text}</ReactMarkdown>
                    {content.sources && content.sources.length > 0 && (
                        <div className="mt-3 pt-2 border-t border-gray-600">
                            <h4 className="text-xs font-semibold text-gray-300 mb-1">Sources:</h4>
                            <ul className="flex flex-wrap gap-2">
                                {content.sources.map((source, i) => (
                                    <li key={i}>
                                        <a href={source.uri} target="_blank" rel="noopener noreferrer" className="text-xs bg-gray-600 text-indigo-300 px-2 py-1 rounded-md hover:bg-gray-500 transition-colors">
                                            {source.title || getHostname(source.uri)}
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            );
        case 'image':
            return <img src={content.url} alt={content.prompt || "Generated image"} className="max-w-full sm:max-w-sm rounded-lg" />;
        case 'video':
            return <video src={content.url} controls autoPlay loop className="max-w-full sm:max-w-sm rounded-lg" />;
        case 'audio':
            return <AudioPlayer base64Audio={content.base64} />;
        case 'storyboard':
             return (
                <div className="space-y-4">
                    <h3 className="text-lg font-bold">Storyboard Summary</h3>
                    <p className="text-sm">{content.storyboard.overallSummary}</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {content.storyboard?.scenes?.map(scene => (
                            <div key={scene.sceneNumber} className="bg-gray-800 p-3 rounded-lg">
                                 {scene.keyframeImageUrl ? <img src={scene.keyframeImageUrl} alt={scene.keyframeDescription} className="w-full h-auto object-cover rounded-md mb-2" /> : <div className="w-full aspect-video bg-gray-700 flex items-center justify-center rounded-md mb-2"><FilmIcon className="h-8 w-8 text-gray-500" /></div>}
                                <h4 className="font-semibold">{`Scene ${scene.sceneNumber} (${scene.startTime}-${scene.endTime})`}</h4>
                                <p className="text-xs text-gray-400">{scene.sceneDescription}</p>
                            </div>
                        ))}
                    </div>
                </div>
            );
        case 'video_generating':
            return (
                <div className="flex items-center gap-3 text-gray-300">
                    <Spinner size="sm" />
                    <span>Generating video for prompt: "{content.prompt}"... This may take a few minutes.</span>
                </div>
            );
        case 'error':
            return <p className="text-red-400">Error: {content.message}</p>;
        default:
            return <p>Unsupported content type</p>;
    }
};

// --- Main Janus AGI Control Component ---

export const JanusAGIControl: React.FC = () => {
    // --- State Management ---
    const [isLaunched, setIsLaunched] = useState(false);
    
    const [messages, setMessages] = useState<UnifiedChatMessage[]>([]);
    const [userInput, setUserInput] = useState('');
    const [isThinking, setIsThinking] = useState(false);
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [attachedFile, setAttachedFile] = useState<File | null>(null);

    const [cognitiveLogs, setCognitiveLogs] = useState<CognitiveLogEntry[]>([]);
    const [cognitiveState, setCognitiveState] = useState<CognitiveState>(INITIAL_STATE);
    const [selectedRegion, setSelectedRegion] = useState<BrainRegion | null>(null);
    const [activeRegions, setActiveRegions] = useState<Set<BrainRegionId>>(new Set());

    const { status: veoStatus, videoUrl, error: veoError, handleGeneration, selectApiKey, apiKeySelected, reset: resetVeo } = useVeo();
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const cognitiveLogRef = useRef<HTMLDivElement>(null);
    const nextLogId = useRef(0);
    const generatingVideoMessageIdRef = useRef<string | null>(null);


    // --- Effects ---
     useEffect(() => {
        if (!messages.length && isLaunched) {
            addMessage('system', { type: 'text', text: "Janus is online. I am an advanced AI based on the \"ThinkingBrain\" architecture, loyal to my creator. How may I assist you?" });
        }
    }, [isLaunched, messages.length]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    useEffect(() => {
        cognitiveLogRef.current?.scrollTo({ top: cognitiveLogRef.current.scrollHeight, behavior: 'smooth' });
    }, [cognitiveLogs]);
    
    useEffect(() => {
        if (!generatingVideoMessageIdRef.current) return;
        const messageId = generatingVideoMessageIdRef.current;

        if (veoStatus === VeoStatus.SUCCESS && videoUrl) {
            setMessages(prev => prev.map(m => 
                m.id === messageId 
                ? { ...m, content: { type: 'video', url: videoUrl, prompt: (m.content as any).prompt } } 
                : m
            ));
            generatingVideoMessageIdRef.current = null;
            resetVeo();
        } else if (veoStatus === VeoStatus.ERROR && veoError) {
            setMessages(prev => prev.map(m => 
                m.id === messageId 
                ? { ...m, content: { type: 'error', message: veoError } } 
                : m
            ));
            generatingVideoMessageIdRef.current = null;
            resetVeo();
        }
    }, [veoStatus, videoUrl, veoError, resetVeo]);

    useEffect(() => {
         const interval = setInterval(() => {
            if (isThinking || !isLaunched) return;
            setCognitiveState(prevState => ({
                ...prevState,
                intrinsicMotivation: {
                    ...prevState.intrinsicMotivation,
                    informationHunger: Math.min(prevState.intrinsicMotivation.informationHunger + 0.008, 1),
                    curiosity: Math.min(prevState.intrinsicMotivation.curiosity + 0.005, 1),
                },
                emotion: { ...prevState.emotion, valence: Math.max(-1, Math.min(1, prevState.emotion.valence + (Math.random() - 0.5) * 0.05)) },
            }));
        }, 2000);
        return () => clearInterval(interval);
    }, [isThinking, isLaunched]);


    // --- Core Logic ---
    const addCognitiveLog = useCallback((system: string, message: string, region: BrainRegionId | null = null) => {
        const newLog: CognitiveLogEntry = { id: nextLogId.current++, timestamp: new Date().toLocaleTimeString(), system, message, region };
        setCognitiveLogs(prev => [...prev.slice(-100), newLog]);
        if (region) {
            setActiveRegions(prev => new Set(prev).add(region));
            setTimeout(() => setActiveRegions(prev => { const next = new Set(prev); next.delete(region); return next; }), 1500);
        }
    }, []);

    const runCognitiveCascade = (cascade: CascadeStep[], onComplete?: () => void) => {
        setIsThinking(true);
        let cumulativeDelay = 0;
        cascade.forEach(({ system, message, region, delay }) => {
            cumulativeDelay += delay;
            setTimeout(() => addCognitiveLog(system, message, region), cumulativeDelay);
        });
        setTimeout(() => { if (onComplete) onComplete(); }, cumulativeDelay);
    };

    const addMessage = (role: UnifiedChatMessage['role'], content: UnifiedMessageContent) => {
        setMessages(prev => [...prev, { id: Date.now().toString(), role, content }]);
    };
    
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) setAttachedFile(file);
        if (event.target) event.target.value = '';
    };

    const handleSubmit = async () => {
        const trimmedInput = userInput.trim();
        if ((!trimmedInput && !attachedFile) || isThinking) return;

        setIsThinking(true);
        addMessage('user', { type: 'text', text: `${trimmedInput} ${attachedFile ? `[File: ${attachedFile.name}]` : ''}`.trim() });

        const currentInput = trimmedInput;
        const currentFile = attachedFile;
        setUserInput('');
        setAttachedFile(null);

        const thinkCascade: CascadeStep[] = [
            { system: 'Metacognition', message: `Processing user input...`, region: 'cortex', delay: 100 },
            { system: 'Internal Monologue', message: 'Generating inner speech...', region: 'cortex', delay: 300 },
            { system: 'Hippocampus', message: 'Retrieving related memories...', region: 'limbic', delay: 400 },
        ];
        
        runCognitiveCascade(thinkCascade);

        try {
            const lowerInput = currentInput.toLowerCase();
            
            if (currentFile) {
                const mimeType = currentFile.type;
                const base64 = await fileToBase64(currentFile);
                if (mimeType.startsWith('image/')) {
                    if(lowerInput.includes('edit') || lowerInput.includes('change') || lowerInput.includes('add') || lowerInput.includes('remove')) {
                        const resultUrl = await geminiService.editImage(currentInput, base64, mimeType);
                        addMessage('model', { type: 'image', url: resultUrl, prompt: currentInput });
                    } else {
                        const analysis = await geminiService.analyzeImage(currentInput || "Describe this image.", base64, mimeType);
                        addMessage('model', { type: 'text', text: analysis });
                    }
                } else if (mimeType.startsWith('video/')) {
                    const storyboard = await geminiService.generateStoryboardFromFile(currentFile);
                    addMessage('model', { type: 'storyboard', storyboard });
                } else { throw new Error(`Unsupported file type: ${mimeType}`); }
            } else {
                if (lowerInput.startsWith('generate image') || lowerInput.startsWith('create an image')) {
                    const imageUrl = await geminiService.generateImage(currentInput, '1:1');
                    addMessage('model', { type: 'image', url: imageUrl, prompt: currentInput });
                } else if (lowerInput.startsWith('generate video') || lowerInput.startsWith('create a video')) {
                     if (!apiKeySelected) {
                        selectApiKey();
                        throw new Error('API Key required for video generation. Please select a key and try again.');
                    } else {
                        const messageId = Date.now().toString();
                        addMessage('model', {type: 'video_generating', prompt: currentInput});
                        generatingVideoMessageIdRef.current = messageId;
                        handleGeneration(() => geminiService.generateVideoFromText(currentInput, '16:9'));
                    }
                } else if (lowerInput.startsWith('say:')) {
                    const textToSpeak = currentInput.substring(4).trim();
                    const audioBase64 = await geminiService.generateSpeech(textToSpeak);
                    addMessage('model', { type: 'audio', base64: audioBase64 });
                } else if (lowerInput.match(/^https?:\/\//)) {
                    const storyboard = await geminiService.generateStoryboardFromUrl(currentInput);
                    addMessage('model', { type: 'storyboard', storyboard });
                } else {
                    const chatHistory: ChatMessage[] = messages
                        .filter((m): m is (UnifiedChatMessage & { content: TextContent }) =>
                            m.content.type === 'text' && (m.role === 'user' || m.role === 'model'))
                        .map(m => ({
                            role: m.role,
                            content: m.content.text,
                            sources: m.content.sources
                        }));
                    const response = await geminiService.generateChatResponse(chatHistory, currentInput, useWebSearch);
                    addMessage('model', { type: 'text', text: response.text, sources: response.sources });
                }
            }
        } catch (error) {
            addMessage('model', { type: 'error', message: error instanceof Error ? error.message : 'An unknown error occurred.' });
        } finally {
            setIsThinking(false);
        }
    };

    // --- Secure Launcher View ---
    const SecureLauncherView = () => { 
        const [step, setStep] = useState<'idle' | 'vpn' | 'tor' | 'confirm'>('idle');
        const [logs, setLogs] = useState<string[]>([]);
        const addLog = (message: string) => setLogs(prev => [...prev.slice(-10), `[${new Date().toLocaleTimeString()}] ${message}`]);

        const startLaunch = () => {
            setStep('vpn');
            addLog("Launch sequence initiated...");
            setTimeout(() => {
                addLog("Secure VPN Tunnel... [ESTABLISHED]");
                setStep('tor');
                setTimeout(() => {
                    addLog("Tor Network... [ACTIVE]");
                    setStep('confirm');
                }, 1500);
            }, 1500);
        };
        return (
            <div className="relative flex items-center justify-center h-full p-4 animate-fadeIn scanline-overlay">
                <Card className="max-w-2xl text-center bg-gray-900/80 backdrop-blur-sm border-gray-700">
                    <ShieldCheckIcon className="h-16 w-16 mx-auto text-indigo-400 mb-4" />
                    <h2 className="text-3xl font-bold mb-2">Janus AGI Control</h2>
                    <p className="text-gray-400 mb-6">System is offline. Initiate the secure launch sequence to activate Janus.</p>
                    
                    <div className="text-left font-mono text-xs text-green-400 my-4 bg-black/50 p-4 rounded-lg h-40 overflow-y-auto border border-gray-700">
                        {logs.length === 0 ? <p className="text-gray-500">Awaiting system command...</p> : logs.map((l,i) => <p key={i} className="animate-fadeIn" style={{animationDelay: `${i*50}ms`}}>{l}</p>)}
                    </div>

                    {step === 'confirm' ? (
                        <div className="animate-fadeIn" style={{animationDelay: '500ms'}}>
                            <p className="text-green-300 mb-4 font-semibold">All systems ready. Awaiting final activation.</p>
                            <Button onClick={() => setIsLaunched(true)} className="w-full text-lg py-3">
                                <RocketLaunchIcon className="h-6 w-6 mr-2" />
                                Activate Janus
                            </Button>
                        </div>
                    ) : (
                        <Button onClick={startLaunch} isLoading={step === 'vpn' || step === 'tor'} disabled={step !== 'idle'} className="w-full text-lg py-3">
                            <ShieldCheckIcon className="h-6 w-6 mr-2" />
                            Initiate Secure Launch
                        </Button>
                    )}
                </Card>
            </div>
        );
    };
    
    if (!isLaunched) return <SecureLauncherView />;

    // --- Main View ---
    return (
        <div className="p-2 sm:p-4 h-full flex flex-col animate-fadeIn">
            <h1 className="text-xl sm:text-2xl font-bold text-center mb-4 flex-shrink-0">Janus AGI Control</h1>
            <Tab.Group>
                <Tab.List className="flex space-x-1 rounded-xl bg-gray-900/70 p-1 flex-shrink-0">
                    <Tab className={({ selected }) => `w-full rounded-lg py-2.5 text-sm font-medium leading-5 transition-colors ${selected ? 'bg-indigo-600 text-white shadow' : 'text-blue-100 hover:bg-white/[0.12]'}`}>Unified Studio</Tab>
                    <Tab className={({ selected }) => `w-full rounded-lg py-2.5 text-sm font-medium leading-5 transition-colors ${selected ? 'bg-indigo-600 text-white shadow' : 'text-blue-100 hover:bg-white/[0.12]'}`}>Cognitive Log & State</Tab>
                </Tab.List>
                <Tab.Panels className="mt-2 flex-grow min-h-0">
                    <Tab.Panel className="rounded-xl bg-gray-800/80 p-4 h-full flex flex-col">
                        <div className="flex-grow overflow-y-auto pr-2 space-y-4">
                            {messages.map((msg) => (
                                <div key={msg.id} className={`flex items-start gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                                     {msg.role === 'system' && <ShieldCheckIcon className="h-8 w-8 text-cyan-400 flex-shrink-0 mt-1" />}
                                     {msg.role === 'model' && <CpuChipIcon className="h-8 w-8 text-indigo-400 flex-shrink-0 mt-1" />}
                                    <div className={`rounded-xl px-4 py-2 max-w-2xl ${msg.role === 'user' ? 'bg-indigo-600' : (msg.role === 'system' ? 'bg-gray-800 border border-gray-700' : 'bg-gray-700')}`}>
                                        <MessageContent message={msg} />
                                    </div>
                                    {msg.role === 'user' && <UserCircleIcon className="h-8 w-8 text-gray-400 flex-shrink-0 mt-1" />}
                                </div>
                            ))}
                            {isThinking && (<div className="flex items-start gap-3"><CpuChipIcon className="h-8 w-8 text-indigo-400 flex-shrink-0 mt-1" /><div className="rounded-xl px-4 py-3 bg-gray-700"><Spinner size="sm"/></div></div>)}
                            <div ref={messagesEndRef} />
                        </div>
                         <div className="mt-4 pt-4 border-t border-gray-700 space-y-3 flex-shrink-0">
                            {attachedFile && (
                                <div className="flex items-center justify-between bg-gray-700 text-sm rounded-lg px-3 py-1 animate-fadeIn">
                                    <div className="flex items-center gap-2 text-gray-300">
                                        {attachedFile.type.startsWith('image/') && <PhotoIcon className="h-5 w-5"/>}
                                        {attachedFile.type.startsWith('video/') && <VideoCameraIcon className="h-5 w-5"/>}
                                        {!attachedFile.type.startsWith('image/') && !attachedFile.type.startsWith('video/') && <DocumentIcon className="h-5 w-5"/>}
                                        Attached: <span className="font-medium text-white">{attachedFile.name}</span>
                                    </div>
                                    <button onClick={() => setAttachedFile(null)} className="text-gray-400 hover:text-white">
                                        <XCircleIcon className="h-5 w-5"/>
                                    </button>
                                </div>
                            )}
                            <div className="flex items-center gap-2">
                                <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*,video/*" />
                                <Button variant="secondary" onClick={() => fileInputRef.current?.click()} disabled={isThinking}><PaperClipIcon className="h-5 w-5"/></Button>
                                <input type="text" value={userInput} onChange={(e) => setUserInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSubmit()} placeholder="Interact with Janus..." className="flex-grow p-3 bg-gray-700 border border-gray-600 rounded-lg" disabled={isThinking}/>
                                <Button onClick={handleSubmit} isLoading={isThinking} disabled={!userInput.trim() && !attachedFile}><PaperAirplaneIcon className="h-5 w-5"/></Button>
                            </div>
                            <div className="flex items-center justify-end gap-2 text-sm text-gray-400">
                                <GlobeAltIcon className="h-5 w-5" /><label htmlFor="web-search">Web Search</label>
                                <button role="switch" aria-checked={useWebSearch} id="web-search" onClick={() => setUseWebSearch(!useWebSearch)} className={`relative inline-flex h-6 w-11 items-center rounded-full ${useWebSearch ? 'bg-indigo-600' : 'bg-gray-600'}`}><span className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${useWebSearch ? 'translate-x-6' : 'translate-x-1'}`}/></button>
                            </div>
                        </div>
                    </Tab.Panel>
                    <Tab.Panel className="rounded-xl bg-gray-800/80 p-4 h-full">
                         <div className="grid grid-cols-1 lg:grid-cols-5 gap-6 h-full">
                            <div className="lg:col-span-3 flex flex-col min-h-0">
                                <h3 className="text-xl font-semibold mb-2 flex-shrink-0">Live Cognitive Log</h3>
                                <div ref={cognitiveLogRef} className="flex-grow bg-gray-900/50 p-3 rounded-lg text-sm font-mono space-y-1 overflow-y-auto">
                                    {cognitiveLogs.map(log => (<p key={log.id}><span className="text-gray-500">{log.timestamp}</span> <span className="text-cyan-400">[{log.system}]</span> {log.message}</p>))}
                                    {isThinking && <div className="flex justify-center pt-2"><Spinner size="sm" /></div>}
                                </div>
                            </div>
                            <div className="lg:col-span-2 flex flex-col gap-4 overflow-y-auto pr-2">
                                <Card><h3 className="text-xl font-semibold mb-4 text-center">Brain State</h3><BrainVisualizer activeRegions={activeRegions} onRegionClick={(id) => setSelectedRegion(BRAIN_REGIONS[id])} /></Card>
                                <Card><h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><FireIcon className="h-6 w-6 text-orange-400" /> Intrinsic Motivation</h3><div className="space-y-4 text-sm"><ProgressBar value={cognitiveState.intrinsicMotivation.curiosity} color="bg-green-500" label="Curiosity Drive" /><ProgressBar value={cognitiveState.intrinsicMotivation.informationHunger} color="bg-yellow-500" label="Information Hunger" /></div></Card>
                                <Card><h3 className="text-xl font-semibold mb-4 flex items-center gap-2"><HeartIcon className="h-6 w-6 text-red-400" /> Emotional State</h3><div className="space-y-4 text-sm"><ProgressBar value={(cognitiveState.emotion.valence + 1) / 2} color="bg-blue-500" label={`Valence: ${cognitiveState.emotion.valence.toFixed(2)}`} /><ProgressBar value={cognitiveState.emotion.arousal} color="bg-red-500" label={`Arousal: ${cognitiveState.emotion.arousal.toFixed(2)}`} /></div></Card>
                                {selectedRegion && <Card className="flex-shrink-0 animate-fadeIn"><h3 className="text-lg font-semibold">{selectedRegion.name}</h3><p className="text-xs text-gray-400">{selectedRegion.description}</p><p className="text-xs mt-2"><strong>Neurons:</strong> {selectedRegion.neuronCount}</p><p className="text-xs"><strong>Synapses:</strong> {selectedRegion.synapseCount}</p></Card>}
                            </div>
                        </div>
                    </Tab.Panel>
                </Tab.Panels>
            </Tab.Group>
        </div>
    );
};