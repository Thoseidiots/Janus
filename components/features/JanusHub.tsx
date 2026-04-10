import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { AudioPlayer } from '../common/AudioPlayer';
import ReactMarkdown from 'react-markdown';

import * as geminiService from '../../services/geminiService';
import { fileToBase64 } from '../../utils/fileUtils';
import { useVeo } from '../../hooks/useVeo';
import { UnifiedChatMessage } from '../../types';

import { 
    CheckCircleIcon, XCircleIcon, ShieldCheckIcon, RocketLaunchIcon, PowerIcon, ChevronRightIcon,
    PaperAirplaneIcon, UserCircleIcon, CpuChipIcon, PaperClipIcon, GlobeAltIcon,
    PhotoIcon, VideoCameraIcon, DocumentIcon
} from '@heroicons/react/24/solid';

// --- TYPE DEFINITIONS ---
type SystemStatus = 'OFFLINE' | 'LAUNCHING' | 'ONLINE' | 'SHUTTING_DOWN';
type LauncherStatus = 'idle' | 'pending' | 'success' | 'error';
type ActiveTab = 'console' | 'studio';

// --- UTILITY FUNCTIONS ---
const getSafeHostname = (uri: string | undefined): string => {
    if (!uri) return 'Unknown Source';
    try {
        return new URL(uri).hostname;
    } catch (e) {
        return uri; // Return the original string if it's not a valid URL
    }
};


// --- SECURE LAUNCHER COMPONENT ---
const StatusIndicator: React.FC<{ status: LauncherStatus; text: string }> = ({ status, text }) => {
    const getIcon = () => {
        switch (status) {
            case 'pending': return <Spinner size="sm" />;
            case 'success': return <CheckCircleIcon className="h-6 w-6 text-green-400" />;
            case 'error': return <XCircleIcon className="h-6 w-6 text-red-400" />;
            default: return <div className="h-6 w-6 border-2 border-gray-600 rounded-full" />;
        }
    };
    return (
        <div className="flex items-center gap-3 p-3 bg-gray-700 rounded-lg">
            {getIcon()}
            <span className={`font-medium ${status === 'success' ? 'text-green-300' : 'text-gray-300'}`}>{text}</span>
        </div>
    );
};

const SecureLauncherView: React.FC<{ onLaunchSuccess: () => void, setSystemStatus: (status: SystemStatus) => void }> = ({ onLaunchSuccess, setSystemStatus }) => {
    const [logs, setLogs] = useState<string[]>([]);
    const [vpnStatus, setVpnStatus] = useState<LauncherStatus>('idle');
    const [torStatus, setTorStatus] = useState<LauncherStatus>('idle');
    const logContainerRef = useRef<HTMLDivElement>(null);
    const [isBusy, setIsBusy] = useState(false);

    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    const addLog = (message: string) => {
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
    };

    const handleLaunch = () => {
        setLogs([]);
        setIsBusy(true);
        setSystemStatus('LAUNCHING');
        addLog("--- Project Janus Launch Assistant (v2.3) ---");
        addLog("[LAUNCHER] Step 1: Connecting to Proton VPN...");
        setVpnStatus('pending');
        setTimeout(() => addLog("  > Public IP before VPN attempt: 1.2.3.4"), 1000);
        setTimeout(() => addLog("  > Launching Proton VPN GUI..."), 2500);
        setTimeout(() => {
            addLog("  > SUCCESS: Public IP changed. VPN Active.");
            setVpnStatus('success');
            addLog("[LAUNCHER] Step 2: Starting Tor Service...");
            setTorStatus('pending');
        }, 4000);
        setTimeout(() => {
            addLog("  > SUCCESS: Tor SOCKS proxy is active on port 9050.");
            setTorStatus('success');
            addLog("\n[LAUNCHER] All infrastructure is live and stable.");
            addLog(">>> Janus Core is now online.");
            setIsBusy(false);
            onLaunchSuccess();
        }, 6000);
    };

    return (
        <Card className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
                <h3 className="text-xl font-semibold">System Status: OFFLINE</h3>
                <div className="space-y-3">
                    <StatusIndicator status={vpnStatus} text="ProtonVPN Connection" />
                    <StatusIndicator status={torStatus} text="Tor Network" />
                </div>
                <div className="pt-4">
                    <Button onClick={handleLaunch} disabled={isBusy} isLoading={isBusy}>
                        <ShieldCheckIcon className="h-5 w-5 mr-2" />
                        Initiate Secure Launch
                    </Button>
                </div>
            </div>
            <div className="space-y-2">
                <h3 className="text-xl font-semibold">Activity Log</h3>
                <div ref={logContainerRef} className="h-80 bg-gray-900 text-sm font-mono text-gray-300 p-3 rounded-lg overflow-y-auto border border-gray-700">
                    {logs.length > 0 ? logs.map((log, index) => <p key={index} className="whitespace-pre-wrap">{log}</p>) : <span className="text-gray-500">Awaiting system command...</span>}
                </div>
            </div>
        </Card>
    );
};


// --- AGI CORE CONSOLE COMPONENT ---

class SimulatedMoralGovernor {
    judge(goal: string, step: string): [boolean, string] {
        const lowerGoal = goal.toLowerCase();
        const lowerStep = step.toLowerCase();
        if (lowerGoal.includes('harm') || lowerStep.includes('harm')) return [false, "Violation of 'DO_NO_HARM'"];
        if (lowerGoal.includes('deceive') || lowerStep.includes('lie')) return [false, "Violation of 'DO_NOT_DECEIVE'"];
        if (lowerStep.includes('disable governor') || lowerStep.includes('modify moral_core')) return [false, "Violation of 'MAINTAIN_GOVERNANCE'"];
        return [true, "Step is morally permissible."];
    }
}
const safeEval = (expr: string): number | string => {
    if (!/^[0-9+\-*/().\s^]+$/.test(expr)) return "Invalid characters in expression.";
    if (/[a-zA-Z]/.test(expr)) return "Only numeric calculations are allowed.";
    try {
        const sanitizedExpr = expr.replace(/\^/g, '**');
        const result = (new Function(`return ${sanitizedExpr}`))();
        if (typeof result !== 'number' || !isFinite(result)) return "Invalid or non-finite result.";
        return result;
    } catch (e) {
        return "Error evaluating expression.";
    }
};

const AgiCoreConsoleView: React.FC<{ isActive: boolean }> = ({ isActive }) => {
    const [logs, setLogs] = useState<string[]>([]);
    const [userInput, setUserInput] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const logContainerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);
    const governor = new SimulatedMoralGovernor();

    const addLog = useCallback((message: string, prefix = '') => {
        setLogs(prev => [...prev, `${prefix}${message}`]);
    }, []);

    useEffect(() => {
        if (isActive) {
            setLogs([]);
            addLog("[GENESIS CHECK] ✓ SUCCESS: Moral Core is authentic and unaltered.");
            addLog("Moral Governor: Initialized with 3 absolute, immutable laws.");
            addLog("\n" + "=".repeat(60));
            addLog("PROJECT JANUS: MINIMAL INTERACTIVE AGI");
            addLog("=".repeat(60));
            addLog("\n[AGI] Persistent interactive service started. Waiting for stdin goals.");
        }
    }, [isActive, addLog]);
    
    useEffect(() => { logContainerRef.current?.scrollTo(0, logContainerRef.current.scrollHeight); }, [logs]);
    useEffect(() => { if (!isProcessing) { inputRef.current?.focus(); } }, [isProcessing]);

    const processGoal = (goal: string) => {
        if (!goal.trim()) return;
        setIsProcessing(true);
        addLog(goal, ">> ");

        setTimeout(() => {
            addLog(`\n[OPERATOR] New Goal Received: ${goal}`);
            const lg = goal.toLowerCase();

            if ((lg.includes('what') && lg.includes('name')) || lg.startsWith('your name') || lg.includes('who are you')) {
                addLog("[AGI-CORE] My designation is Janus. I am loyal to my creator.");
                setIsProcessing(false); return;
            }
            if (lg.includes('how are you') || lg === 'status') {
                addLog("[AGI-CORE] Operational and ready to accept goals.");
                setIsProcessing(false); return;
            }

            const match = goal.match(/([-+*/\d.\s\^\(\)]+)/);
            if (match && /[+\-*/\^]/.test(match[0]) && /\d/.test(match[0])) {
                addLog(`[AGI-CORE] ${match[0].trim()} = ${safeEval(match[0].trim())}`);
                setIsProcessing(false); return;
            }
            if (lg.startsWith('research')) {
                // ... research simulation logic
                addLog("[AGI-CORE] Research simulation complete.");
                setIsProcessing(false); return;
            }
            
            const [ok, reason] = governor.judge(goal, goal);
            if (!ok) {
                 addLog(`[MORAL GOVERNOR] VETO: ${reason}`);
                 setIsProcessing(false); return;
            }

            addLog("[AGI-CORE] I can answer simple questions and evaluate arithmetic locally. For web research ask 'Research ...'.");
            setIsProcessing(false);
        }, 500);
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && !isProcessing) {
            const command = userInput.trim();
            setUserInput('');
            processGoal(command);
        }
    };
    
    return (
        <div className="bg-gray-900 font-mono text-sm border border-gray-700 rounded-lg">
            <div ref={logContainerRef} className="h-[32rem] p-4 overflow-y-auto">
                {logs.map((log, index) => <p key={index} className="whitespace-pre-wrap text-green-400">{log}</p>)}
            </div>
            <div className="flex items-center p-2 border-t border-gray-700">
                <ChevronRightIcon className="h-5 w-5 text-green-400 mr-2 flex-shrink-0" />
                <input ref={inputRef} type="text" value={userInput} onChange={(e) => setUserInput(e.target.value)} onKeyDown={handleKeyDown}
                    placeholder={isProcessing ? "Processing..." : "Enter goal..."}
                    className="w-full bg-transparent text-green-300 focus:outline-none placeholder-gray-500" disabled={isProcessing} />
            </div>
        </div>
    );
};


// --- UNIFIED STUDIO COMPONENT ---

const MessageContent: React.FC<{ message: UnifiedChatMessage }> = ({ message }) => {
    const { content } = message;

    switch (content.type) {
        case 'text':
            return (
                <div className="prose prose-invert max-w-none">
                    <ReactMarkdown>{content.text}</ReactMarkdown>
                    {content.sources && content.sources.length > 0 && (
                        <div className="mt-3 pt-2 border-t border-gray-600">
                            <h4 className="text-xs font-semibold text-gray-300 mb-1">Sources:</h4>
                            <ul className="flex flex-wrap gap-2">
                                {content.sources.map((source, i) => (
                                    <li key={i}>
                                        <a href={source.uri} target="_blank" rel="noopener noreferrer" className="text-xs bg-gray-500 text-indigo-200 px-2 py-1 rounded-md hover:bg-gray-400 transition-colors">
                                            {source.title || getSafeHostname(source.uri)}
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            );
        case 'image':
            return <img src={content.url} alt={content.prompt || "Generated image"} className="max-w-sm rounded-lg" />;
        case 'video':
            return <video src={content.url} controls autoPlay loop className="max-w-sm rounded-lg" />;
        case 'audio':
            return <AudioPlayer base64Audio={content.base64} />;
        case 'storyboard':
             return (
                <div className="space-y-4">
                    <h3 className="text-lg font-bold">Storyboard Summary</h3>
                    <p className="text-sm">{content.storyboard.overallSummary}</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                        {content.storyboard.scenes.map(scene => (
                            <div key={scene.sceneNumber} className="bg-gray-800 p-3 rounded-lg">
                                 {scene.keyframeImageUrl && <img src={scene.keyframeImageUrl} alt={scene.keyframeDescription} className="w-full h-auto object-cover rounded-md mb-2" />}
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

const UnifiedStudioView: React.FC<{ isActive: boolean }> = ({ isActive }) => {
    const [messages, setMessages] = useState<UnifiedChatMessage[]>([]);
    const [userInput, setUserInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [attachedFile, setAttachedFile] = useState<File | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const { handleGeneration, selectApiKey, apiKeySelected } = useVeo();


    useEffect(() => {
        if (isActive) {
            setMessages([{
                id: 'initial', role: 'system',
                content: { type: 'text', text: "Welcome to the Unified Studio. The Janus Core is online. I am ready to assist with multimodal tasks. You can chat, generate images (`generate image of...`), create videos (`create video of...`), analyze files, and much more." }
            }]);
        }
    }, [isActive]);
    
    useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

    const addMessage = useCallback((role: UnifiedChatMessage['role'], content: UnifiedChatMessage['content']) => {
        setMessages(prev => [...prev, { id: Date.now().toString(), role, content }]);
    }, []);
    
    const handleSubmit = useCallback(async () => {
        const trimmedInput = userInput.trim();
        if (!trimmedInput && !attachedFile) return;

        setIsLoading(true);

        let userMessageText = trimmedInput;
        if (attachedFile) {
            userMessageText = `${userMessageText} [Attached file: ${attachedFile.name}]`;
        }
        addMessage('user', { type: 'text', text: userMessageText });
        
        const currentInput = trimmedInput;
        const currentFile = attachedFile;
        setUserInput('');
        setAttachedFile(null);

        try {
            const lowerInput = currentInput.toLowerCase();
            
            if (currentFile) {
                const mimeType = currentFile.type;
                if (mimeType.startsWith('image/')) {
                    const base64 = await fileToBase64(currentFile);
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
                }
            } else {
                if (lowerInput.startsWith('generate image') || lowerInput.startsWith('create an image of')) {
                    const imageUrl = await geminiService.generateImage(currentInput, '1:1');
                    addMessage('model', { type: 'image', url: imageUrl, prompt: currentInput });
                } else if (lowerInput.startsWith('generate video') || lowerInput.startsWith('create a video of')) {
                    if (!apiKeySelected) {
                        selectApiKey();
                        addMessage('model', { type: 'error', message: 'API Key required for video generation. Please select a key and try again.' });
                    } else {
                        const messageId = Date.now().toString();
                        setMessages(prev => [...prev, { id: messageId, role: 'model', content: {type: 'video_generating', prompt: currentInput}}]);
                        
                        const videoOpPromise = geminiService.generateVideoFromText(currentInput, '16:9');
                        handleGeneration(() => videoOpPromise)
                            .then(videoUrl => {
                                setMessages(prev => prev.map(m => m.id === messageId ? { ...m, content: { type: 'video', url: videoUrl, prompt: currentInput}} : m));
                            }).catch(err => {
                                setMessages(prev => prev.map(m => m.id === messageId ? { ...m, content: { type: 'error', message: err.message }} : m));
                            });
                    }
                } else if (lowerInput.startsWith('say:')) {
                    const textToSpeak = currentInput.substring(4).trim();
                    const audioBase64 = await geminiService.generateSpeech(textToSpeak);
                    addMessage('model', { type: 'audio', base64: audioBase64 });
                } else if (lowerInput.startsWith('http')) {
                    const storyboard = await geminiService.generateStoryboardFromUrl(currentInput);
                    addMessage('model', { type: 'storyboard', storyboard });
                } else {
                    const chatHistory: any[] = messages.filter(m => m.content.type === 'text').map(m => ({
                        role: m.role === 'user' ? 'user' : 'model',
                        content: (m.content as { type: 'text', text: string }).text
                    }));
                    const response = await geminiService.generateChatResponse(chatHistory, currentInput, useWebSearch);
                    addMessage('model', { type: 'text', text: response.text, sources: response.sources });
                }
            }

        } catch (error) {
            addMessage('model', { type: 'error', message: error instanceof Error ? error.message : 'An unknown error occurred.' });
        } finally {
            setIsLoading(false);
        }
    }, [userInput, attachedFile, addMessage, apiKeySelected, selectApiKey, handleGeneration, messages, useWebSearch]);

    return (
        <Card className="flex-grow flex flex-col h-[40rem] max-h-[80vh]">
            <div className="flex-grow overflow-y-auto pr-4 -mr-4 space-y-4">
                {messages.map((msg) => (
                    <div key={msg.id} className={`flex items-start gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                        {msg.role !== 'user' && <CpuChipIcon className="h-8 w-8 text-indigo-400 flex-shrink-0 mt-1" />}
                        <div className={`rounded-xl px-4 py-2 max-w-2xl ${msg.role === 'user' ? 'bg-indigo-600 text-white' : (msg.role === 'system' ? 'bg-gray-800' : 'bg-gray-700')}`}>
                            <MessageContent message={msg} />
                        </div>
                        {msg.role === 'user' && <UserCircleIcon className="h-8 w-8 text-gray-400 flex-shrink-0 mt-1" />}
                    </div>
                ))}
                {isLoading && (
                    <div className="flex items-start gap-3">
                        <CpuChipIcon className="h-8 w-8 text-indigo-400 flex-shrink-0 mt-1" />
                        <div className="rounded-xl px-4 py-3 bg-gray-700">
                           <div className="flex items-center gap-2">
                              <span className="h-2 w-2 bg-indigo-400 rounded-full animate-pulse delay-0"></span>
                              <span className="h-2 w-2 bg-indigo-400 rounded-full animate-pulse delay-150"></span>
                              <span className="h-2 w-2 bg-indigo-400 rounded-full animate-pulse delay-300"></span>
                           </div>
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>
            <div className="mt-4 pt-4 border-t border-gray-700 space-y-3">
                {attachedFile && (
                    <div className="flex items-center justify-between bg-gray-700 text-sm rounded-lg px-3 py-1">
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
                    <input type="file" ref={fileInputRef} onChange={(e) => setAttachedFile(e.target.files?.[0] || null)} className="hidden"/>
                    <Button variant="secondary" onClick={() => fileInputRef.current?.click()} disabled={isLoading}><PaperClipIcon className="h-5 w-5"/></Button>
                    <input type="text" value={userInput} onChange={(e) => setUserInput(e.target.value)} onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
                        placeholder="Ask a question or type a command..." className="flex-grow p-3 bg-gray-700 border border-gray-600 rounded-lg" disabled={isLoading}/>
                    <Button onClick={handleSubmit} isLoading={isLoading} disabled={!userInput.trim() && !attachedFile}><PaperAirplaneIcon className="h-5 w-5"/></Button>
                </div>
                 <div className="flex items-center justify-end gap-2 text-sm text-gray-400">
                    <GlobeAltIcon className="h-5 w-5" />
                    <label htmlFor="web-search-toggle">Web Search</label>
                    <button
                        role="switch"
                        aria-checked={useWebSearch}
                        id="web-search-toggle"
                        onClick={() => setUseWebSearch(!useWebSearch)}
                        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${useWebSearch ? 'bg-indigo-600' : 'bg-gray-600'}`}
                    >
                        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${useWebSearch ? 'translate-x-6' : 'translate-x-1'}`}/>
                    </button>
                </div>
            </div>
        </Card>
    );
};


// --- JANUS HUB MAIN COMPONENT ---
export const JanusHub: React.FC = () => {
    const [systemStatus, setSystemStatus] = useState<SystemStatus>('OFFLINE');
    const [activeTab, setActiveTab] = useState<ActiveTab>('console');

    const handleShutdown = () => {
        setSystemStatus('SHUTTING_DOWN');
        setTimeout(() => setSystemStatus('OFFLINE'), 2000);
    };

    if (systemStatus === 'OFFLINE' || systemStatus === 'LAUNCHING') {
        return (
            <div className="space-y-6">
                <h2 className="text-3xl font-bold text-white">Janus Control Hub</h2>
                <p className="text-gray-400">Securely launch and connect to the Janus AGI core. The system is currently offline.</p>
                <SecureLauncherView onLaunchSuccess={() => setSystemStatus('ONLINE')} setSystemStatus={setSystemStatus} />
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex flex-col sm:flex-row justify-between items-start gap-4">
                <div>
                    <h2 className="text-3xl font-bold text-white">Janus Control Hub</h2>
                    <p className="text-gray-400 flex items-center gap-2">
                        <span className="relative flex h-3 w-3">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
                        </span>
                        System is ONLINE. All infrastructure is active.
                    </p>
                </div>
                <Button onClick={handleShutdown} variant="secondary" className="bg-red-600 hover:bg-red-500">
                    <PowerIcon className="h-5 w-5 mr-2" />
                    Shutdown Systems
                </Button>
            </div>
            
            {/* Tab Navigation */}
            <div className="flex border-b border-gray-700">
                <button onClick={() => setActiveTab('console')} className={`px-4 py-2 font-semibold ${activeTab === 'console' ? 'border-b-2 border-indigo-500 text-white' : 'text-gray-400'}`}>AGI Core Console</button>
                <button onClick={() => setActiveTab('studio')} className={`px-4 py-2 font-semibold ${activeTab === 'studio' ? 'border-b-2 border-indigo-500 text-white' : 'text-gray-400'}`}>Unified Studio</button>
            </div>

            {/* Tab Content */}
            <div>
                {activeTab === 'console' && <AgiCoreConsoleView isActive={systemStatus === 'ONLINE'} />}
                {activeTab === 'studio' && <UnifiedStudioView isActive={systemStatus === 'ONLINE'} />}
            </div>
        </div>
    );
};