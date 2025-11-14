import React, { useState, useEffect, useCallback, FC, useRef } from 'react';
import { generateWithGoogleSearch, generateWithGoogleMaps } from '../../services/geminiService';
import { ChatMessage } from '../../types';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { PaperAirplaneIcon, GlobeAltIcon, MapPinIcon } from '@heroicons/react/24/solid';
import ReactMarkdown from 'react-markdown';
import { Spinner } from '../common/Spinner';

type SearchMode = 'web' | 'maps';

export const GroundedSearch: FC = () => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [userInput, setUserInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [mode, setMode] = useState<SearchMode>('web');
    const [location, setLocation] = useState<{ lat: number, lon: number } | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (mode === 'maps' && !location) {
            navigator.geolocation.getCurrentPosition(
                (position) => {
                    setLocation({
                        lat: position.coords.latitude,
                        lon: position.coords.longitude
                    });
                },
                (error) => {
                    console.warn("Could not get geolocation:", error.message);
                }
            );
        }
    }, [mode, location]);
    
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSendMessage = useCallback(async () => {
        if (!userInput.trim() || isLoading) return;

        const newUserMessage: ChatMessage = { role: 'user', content: userInput };
        setMessages(prev => [...prev, newUserMessage]);
        const currentInput = userInput;
        setUserInput('');
        setIsLoading(true);

        try {
            let response;
            if (mode === 'web') {
                response = await generateWithGoogleSearch(currentInput);
            } else {
                response = await generateWithGoogleMaps(currentInput, location?.lat, location?.lon);
            }
            const modelMessage: ChatMessage = { role: 'model', content: response.text, sources: response.sources };
            setMessages(prev => [...prev, modelMessage]);
        } catch (error) {
            const errorMessage: ChatMessage = { role: 'model', content: `Sorry, an error occurred: ${error instanceof Error ? error.message : 'Unknown error'}` };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    }, [userInput, isLoading, mode, location]);
    
    return (
        <div className="h-full flex flex-col">
            <h2 className="text-3xl font-bold text-white mb-4">Grounded Search</h2>
            <p className="text-gray-400 mb-6">Get up-to-date information grounded in Google Search and Maps. Find the latest ML techniques or discover AI research hubs near you.</p>

            <div className="flex items-center space-x-2 bg-gray-800 rounded-lg p-1 w-min mb-6">
                <button onClick={() => setMode('web')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors flex items-center gap-2 ${mode === 'web' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}><GlobeAltIcon className="h-5 w-5"/>Web</button>
                <button onClick={() => setMode('maps')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors flex items-center gap-2 ${mode === 'maps' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}><MapPinIcon className="h-5 w-5"/>Maps</button>
            </div>

            <Card className="flex-grow flex flex-col">
                <div className="flex-grow overflow-y-auto pr-4 -mr-4 space-y-6">
                    {messages.map((msg, index) => (
                        <div key={index}>
                            <div className={`flex items-start gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                                <div className={`rounded-xl px-4 py-2 max-w-2xl ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-gray-700'}`}>
                                    <div className="prose prose-invert max-w-none">
                                        <ReactMarkdown>{msg.content}</ReactMarkdown>
                                    </div>
                                </div>
                            </div>
                            {msg.sources && msg.sources.length > 0 && (
                                <div className={`mt-2 ${msg.role === 'user' ? 'mr-10 text-right' : 'ml-10'}`}>
                                    <h4 className="text-sm font-semibold text-gray-400 mb-1">Sources:</h4>
                                    <ul className="flex flex-wrap gap-2">
                                        {msg.sources.map((source, i) => (
                                            <li key={i}>
                                                <a href={source.uri} target="_blank" rel="noopener noreferrer" className="text-xs bg-gray-600 text-indigo-300 px-2 py-1 rounded-md hover:bg-gray-500 transition-colors">
                                                    {source.title || new URL(source.uri).hostname}
                                                </a>
                                            </li>
                                        ))}
                                    </ul>
                                </div>
                            )}
                        </div>
                    ))}
                    {isLoading && <div className="flex justify-center"><Spinner /></div>}
                    <div ref={messagesEndRef} />
                </div>
                <div className="mt-4 pt-4 border-t border-gray-700 flex items-center gap-2">
                    <input
                        type="text"
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                        placeholder={mode === 'web' ? 'Ask about recent events...' : 'Find places near you...'}
                        className="flex-grow p-3 bg-gray-700 border border-gray-600 rounded-lg"
                        disabled={isLoading}
                    />
                    <Button onClick={handleSendMessage} isLoading={isLoading} disabled={!userInput.trim()}>
                        <PaperAirplaneIcon className="h-5 w-5"/>
                    </Button>
                </div>
            </Card>
        </div>
    );
};