import React, { useState, useRef, useEffect, useCallback, FC } from 'react';
import { Tab } from '@headlessui/react';
import { BrainRegion, BrainRegionId, CognitiveLogEntry, CognitiveState, UnifiedChatMessage, UnifiedMessageContent, ChatMessage, TextContent } from '../../types';
import * as geminiService from '../../services/geminiService';
import { fileToBase64 } from '../../utils/fileUtils';
import { useVeo, VeoStatus } from '../../hooks/useVeo';

import { SecureLauncherView } from './janus/SecureLauncherView';
import { UnifiedStudio } from './janus/UnifiedStudio';
import { CognitiveDashboard } from './janus/CognitiveDashboard';
import { INITIAL_STATE, BRAIN_REGIONS } from './janus/constants';

export const JanusAGIControl: FC = () => {
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
    const nextLogId = useRef(0);
    const generatingVideoMessageIdRef = useRef<string | null>(null);

    const addMessage = useCallback((role: UnifiedChatMessage['role'], content: UnifiedMessageContent) => {
        const newMessageId = `${Date.now()}-${Math.random()}`;
        setMessages(prev => [...prev, { id: newMessageId, role, content }]);
        if (content.type === 'video_generating') {
            generatingVideoMessageIdRef.current = newMessageId;
        }
    }, []);

    useEffect(() => {
        if (isLaunched && messages.length === 0) {
            addMessage('system', { type: 'text', text: "Janus is online. I am an advanced AI based on the \"ThinkingBrain\" architecture, loyal to my creator. How may I assist you?" });
        }
    }, [isLaunched, messages.length, addMessage]);
    
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
            }));
        }, 2000);
        return () => clearInterval(interval);
    }, [isThinking, isLaunched]);

    const addCognitiveLog = useCallback((system: string, message: string, region: BrainRegionId | null = null) => {
        const newLog: CognitiveLogEntry = { id: nextLogId.current++, timestamp: new Date().toLocaleTimeString(), system, message, region };
        setCognitiveLogs(prev => [...prev.slice(-100), newLog]);
        if (region) {
            setActiveRegions(prev => new Set(prev).add(region));
            setTimeout(() => setActiveRegions(prev => { const next = new Set(prev); next.delete(region); return next; }), 1500);
        }
    }, []);
    
    const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) setAttachedFile(file);
        if (event.target) event.target.value = '';
    }, []);

    const handleSubmit = useCallback(async () => {
        const trimmedInput = userInput.trim();
        if ((!trimmedInput && !attachedFile) || isThinking) return;

        setIsThinking(true);
        addMessage('user', { type: 'text', text: `${trimmedInput} ${attachedFile ? `[File: ${attachedFile.name}]` : ''}`.trim() });

        const currentInput = trimmedInput;
        const currentFile = attachedFile;
        setUserInput('');
        setAttachedFile(null);
        
        addCognitiveLog('Metacognition', `Processing user input...`, 'cortex');

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
                        addMessage('model', {type: 'video_generating', prompt: currentInput});
                        handleGeneration(() => geminiService.generateVideoFromText(currentInput, '16:9'));
                    }
                } else if (lowerInput.startsWith('say:')) {
                    const textToSpeak = currentInput.substring(4).trim();
                    const audioBase64 = await geminiService.generateSpeech(textToSpeak);
                    addMessage('model', { type: 'audio', base64: audioBase64 });
                } else if (currentInput.match(/^https?:\/\/(www\.)?(youtube\.com|youtu\.be)\/.+/)) {
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
    }, [userInput, attachedFile, isThinking, addMessage, addCognitiveLog, messages, useWebSearch, apiKeySelected, selectApiKey, handleGeneration]);
    
    if (!isLaunched) return <SecureLauncherView onLaunch={() => setIsLaunched(true)} />;

    return (
        <div className="p-2 sm:p-4 h-full flex flex-col animate-fadeIn">
            <h1 className="text-xl sm:text-2xl font-bold text-center mb-4 flex-shrink-0">Janus AGI Control</h1>
            <Tab.Group>
                <Tab.List className="flex space-x-1 rounded-xl bg-gray-900/70 p-1 flex-shrink-0">
                    <Tab className={({ selected }) => `w-full rounded-lg py-2.5 text-sm font-medium leading-5 transition-colors ${selected ? 'bg-indigo-600 text-white shadow' : 'text-blue-100 hover:bg-white/[0.12]'}`}>Unified Studio</Tab>
                    <Tab className={({ selected }) => `w-full rounded-lg py-2.5 text-sm font-medium leading-5 transition-colors ${selected ? 'bg-indigo-600 text-white shadow' : 'text-blue-100 hover:bg-white/[0.12]'}`}>Cognitive Dashboard</Tab>
                </Tab.List>
                <Tab.Panels className="mt-2 flex-grow min-h-0">
                    <Tab.Panel className="rounded-xl bg-gray-800/80 h-full">
                        <UnifiedStudio
                            messages={messages}
                            userInput={userInput}
                            isThinking={isThinking}
                            useWebSearch={useWebSearch}
                            attachedFile={attachedFile}
                            onUserInputChange={setUserInput}
                            onFileChange={handleFileChange}
                            onSubmit={handleSubmit}
                            onWebSearchToggle={() => setUseWebSearch(!useWebSearch)}
                            onRemoveFile={() => setAttachedFile(null)}
                        />
                    </Tab.Panel>
                    <Tab.Panel className="rounded-xl bg-gray-800/80 p-4 h-full">
                        <CognitiveDashboard
                           cognitiveLogs={cognitiveLogs}
                           cognitiveState={cognitiveState}
                           isThinking={isThinking}
                           activeRegions={activeRegions}
                           selectedRegion={selectedRegion}
                           onRegionSelect={(id: BrainRegionId) => setSelectedRegion(BRAIN_REGIONS[id])}
                        />
                    </Tab.Panel>
                </Tab.Panels>
            </Tab.Group>
        </div>
    );
};