
import React, { useState, useRef, useEffect, useCallback } from 'react';
import { UnifiedChatMessage, GroundingSource } from '../../types';
import * as geminiService from '../../services/geminiService';
import { fileToBase64 } from '../../utils/fileUtils';
import { useVeo, VeoStatus } from '../../hooks/useVeo';

import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { Spinner } from '../common/Spinner';
import { AudioPlayer } from '../common/AudioPlayer';
import ReactMarkdown from 'react-markdown';
import { 
    PaperAirplaneIcon, 
    UserCircleIcon, 
    CpuChipIcon, 
    PaperClipIcon, 
    XCircleIcon, 
    GlobeAltIcon,
    PhotoIcon,
    VideoCameraIcon,
    DocumentIcon,
    FilmIcon
} from '@heroicons/react/24/solid';

// Renderer for different message types
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
                                            {source.title || new URL(source.uri).hostname}
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
                                 <img src={scene.keyframeImageUrl} alt={scene.keyframeDescription} className="w-full h-auto object-cover rounded-md mb-2" />
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


export const UnifiedStudio: React.FC = () => {
    const [messages, setMessages] = useState<UnifiedChatMessage[]>([]);
    const [userInput, setUserInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [attachedFile, setAttachedFile] = useState<File | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const { status, handleGeneration, selectApiKey, apiKeySelected } = useVeo();

    useEffect(() => {
        setMessages([{
            id: 'initial',
            role: 'system',
            content: { type: 'text', text: "Welcome to the Unified Studio! I'm a comprehensive AI assistant. You can chat, generate images (`generate image of...`), create videos (`create video of...`), analyze files, and much more, all in one conversation." }
        }]);
    }, []);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(scrollToBottom, [messages]);

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setAttachedFile(file);
        }
        if (event.target) {
            event.target.value = '';
        }
    };

    const addMessage = (role: UnifiedChatMessage['role'], content: UnifiedChatMessage['content']) => {
        setMessages(prev => [...prev, { id: Date.now().toString(), role, content }]);
    };
    
    const handleSubmit = async () => {
        const trimmedInput = userInput.trim();
        if (!trimmedInput && !attachedFile) return;

        setIsLoading(true);

        // Add user message to history
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
            
            // --- File-based Actions ---
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
            } 
            // --- Text-based Actions ---
            else {
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
                        handleGeneration(() => geminiService.generateVideoFromText(currentInput, '16:9'))
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
                } else if (lowerInput.startsWith('http')) { // Basic URL check
                    const storyboard = await geminiService.generateStoryboardFromUrl(currentInput);
                    addMessage('model', { type: 'storyboard', storyboard });
                }
                else {
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
    };

    return (
         <div className="h-full flex flex-col">
             <h2 className="text-3xl font-bold text-white mb-4">Unified Studio</h2>
             <p className="text-gray-400 mb-6">A single, multimodal AI assistant to help with all your hyperparameter tuning tasks.</p>
            <Card className="flex-grow flex flex-col">
                <div className="flex-grow overflow-y-auto pr-4 -mr-4 space-y-4">
                    {messages.map((msg, index) => (
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
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept="image/*,video/*" />
                        <Button variant="secondary" onClick={() => fileInputRef.current?.click()} disabled={isLoading} aria-label="Attach file">
                            <PaperClipIcon className="h-5 w-5"/>
                        </Button>
                        <input
                            type="text"
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
                            placeholder="Ask a question or type a command..."
                            className="flex-grow p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                            disabled={isLoading}
                        />
                        <Button onClick={handleSubmit} isLoading={isLoading} disabled={!userInput.trim() && !attachedFile}>
                            <PaperAirplaneIcon className="h-5 w-5"/>
                        </Button>
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
        </div>
    );
};
