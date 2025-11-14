import React, { useState, useRef, useEffect, useCallback } from 'react';
import { generateChatResponse } from '../../services/geminiService';
import { ChatMessage } from '../../types';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { PaperAirplaneIcon, UserCircleIcon, CpuChipIcon, PaperClipIcon, XCircleIcon, GlobeAltIcon } from '@heroicons/react/24/solid';
import ReactMarkdown from 'react-markdown';

const promptStarters = [
    "Compare Grid Search vs. Random Search",
    "Generate Python code for a Random Search",
    "Suggest metrics for a classification problem"
];

export const Chatbot: React.FC = () => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [userInput, setUserInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [useWebSearch, setUseWebSearch] = useState(false);
    const [attachedFile, setAttachedFile] = useState<{ name: string; content: string } | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        setMessages([{
            role: 'model',
            content: `Hello! I'm your hyperparameter tuning assistant. While I can't run experiments for you, I can provide expert guidance to accelerate your work. I can:
- **Explain complex strategies** and their trade-offs.
- **Generate Python code snippets** for popular tuning libraries.
- **Discuss best practices** for model evaluation.

To get the best advice, enable **Web Search** for the latest research or **attach your code/logs** for more tailored feedback. How can I help you start?`
        }]);
    }, []);

    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, []);

    useEffect(scrollToBottom, [messages]);

    const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const content = e.target?.result as string;
                setAttachedFile({ name: file.name, content });
            };
            reader.readAsText(file);
        }
        if (event.target) {
            event.target.value = '';
        }
    }, []);

    const handleSendMessage = useCallback(async () => {
        if (!userInput.trim() || isLoading) return;

        let promptWithContext = userInput;
        if (attachedFile) {
            promptWithContext = `CONTEXT FROM FILE (${attachedFile.name}):\n${attachedFile.content}\n\n---\n\nUSER PROMPT:\n${userInput}`;
        }
        
        const newUserMessage: ChatMessage = { role: 'user', content: userInput };
        const historyBeforeThisTurn = [...messages];

        setMessages(prev => [...prev, newUserMessage]);
        setUserInput('');
        setAttachedFile(null);
        setIsLoading(true);

        try {
            const response = await generateChatResponse(historyBeforeThisTurn, promptWithContext, useWebSearch);
            const modelMessage: ChatMessage = { role: 'model', content: response.text, sources: response.sources };
            setMessages(prev => [...prev, modelMessage]);
        } catch (error) {
            console.error(error);
            const errorMessage: ChatMessage = { role: 'model', content: "Sorry, I encountered an error. Please try again." };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    }, [userInput, isLoading, attachedFile, messages, useWebSearch]);

    const handlePromptStarter = useCallback((prompt: string) => {
        setUserInput(prompt);
    }, []);

    return (
        <div className="h-full flex flex-col">
            <h2 className="text-3xl font-bold text-white mb-4">Tuning Chatbot</h2>
            <p className="text-gray-400 mb-6">Have a conversation about your model. Ask for clarifications, explore alternatives, and dive deeper into hyperparameter tuning strategies.</p>
            <Card className="flex-grow flex flex-col">
                <div className="flex-grow overflow-y-auto pr-4 -mr-4 space-y-4">
                    {messages.map((msg, index) => (
                        <div key={index} className={`flex items-start gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                            {msg.role === 'model' && <CpuChipIcon className="h-8 w-8 text-indigo-400 flex-shrink-0 mt-1" />}
                            <div className={`rounded-xl px-4 py-2 max-w-lg ${msg.role === 'user' ? 'bg-indigo-600 text-white' : 'bg-gray-700'}`}>
                               <div className="prose prose-invert max-w-none">
                                 <ReactMarkdown>{msg.content}</ReactMarkdown>
                               </div>
                               {msg.sources && msg.sources.length > 0 && (
                                    <div className="mt-3 pt-2 border-t border-gray-600">
                                        <h4 className="text-xs font-semibold text-gray-300 mb-1">Sources:</h4>
                                        <ul className="flex flex-wrap gap-2">
                                            {msg.sources.map((source, i) => (
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
                            <span className="text-gray-300">Attached: <span className="font-medium text-white">{attachedFile.name}</span></span>
                            <button onClick={() => setAttachedFile(null)} className="text-gray-400 hover:text-white">
                                <XCircleIcon className="h-5 w-5"/>
                            </button>
                        </div>
                    )}

                    {!isLoading && messages.length <= 1 && (
                        <div className="flex flex-wrap gap-2">
                             {promptStarters.map((prompt, i) => (
                                <button 
                                    key={i} 
                                    onClick={() => handlePromptStarter(prompt)}
                                    className="text-sm bg-gray-700 hover:bg-gray-600 text-gray-200 px-3 py-1 rounded-full transition-colors"
                                >
                                    {prompt}
                                </button>
                            ))}
                        </div>
                    )}

                    <div className="flex items-center gap-2">
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept=".txt,.py,.md,.log,.json,.csv" />
                        <Button variant="secondary" onClick={() => fileInputRef.current?.click()} disabled={isLoading} aria-label="Attach file">
                            <PaperClipIcon className="h-5 w-5"/>
                        </Button>
                        <input
                            type="text"
                            value={userInput}
                            onChange={(e) => setUserInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                            placeholder="Ask a follow-up question..."
                            className="flex-grow p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                            disabled={isLoading}
                        />
                        <Button onClick={handleSendMessage} isLoading={isLoading} disabled={!userInput.trim()}>
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