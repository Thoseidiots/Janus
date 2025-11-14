import React, { useRef, useEffect } from 'react';
import { Button } from '../../common/Button';
import { Spinner } from '../../common/Spinner';
import { MessageContent } from './MessageContent';
import { UnifiedChatMessage } from '../../../types';
import {
    PaperAirplaneIcon, UserCircleIcon, PaperClipIcon, GlobeAltIcon,
    PhotoIcon, VideoCameraIcon, DocumentIcon, XCircleIcon
} from '@heroicons/react/24/solid';
import { CpuChipIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';

interface UnifiedStudioProps {
    messages: UnifiedChatMessage[];
    userInput: string;
    isThinking: boolean;
    useWebSearch: boolean;
    attachedFile: File | null;
    onUserInputChange: (value: string) => void;
    onFileChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
    onSubmit: () => void;
    onWebSearchToggle: () => void;
    onRemoveFile: () => void;
}

export const UnifiedStudio: React.FC<UnifiedStudioProps> = ({
    messages,
    userInput,
    isThinking,
    useWebSearch,
    attachedFile,
    onUserInputChange,
    onFileChange,
    onSubmit,
    onWebSearchToggle,
    onRemoveFile
}) => {
    const messagesEndRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const textareaRef = useRef<HTMLTextAreaElement>(null);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Auto-resize textarea based on content
    useEffect(() => {
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto'; // Reset height to shrink
            const scrollHeight = textareaRef.current.scrollHeight;
            textareaRef.current.style.height = `${scrollHeight}px`; // Set to new height
        }
    }, [userInput]);

    return (
        <div className="h-full flex flex-col p-1">
            <div className="flex-grow overflow-y-auto p-4 space-y-4">
                {messages.map((msg) => (
                    <div key={msg.id} className={`flex items-start gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>
                         {msg.role === 'system' && <ShieldCheckIcon className="h-8 w-8 text-cyan-400 flex-shrink-0 mt-1" />}
                         {msg.role === 'model' && <CpuChipIcon className="h-8 w-8 text-indigo-400 flex-shrink-0 mt-1" />}
                        <div className={`rounded-xl px-4 py-2 max-w-2xl shadow-md ${msg.role === 'user' ? 'bg-indigo-600 text-white' : (msg.role === 'system' ? 'bg-gradient-to-br from-gray-800 to-gray-900 border border-cyan-500/30' : 'bg-gray-700')}`}>
                            <MessageContent message={msg} />
                        </div>
                        {msg.role === 'user' && <UserCircleIcon className="h-8 w-8 text-gray-400 flex-shrink-0 mt-1" />}
                    </div>
                ))}
                {isThinking && (
                    <div className="flex items-start gap-4">
                        <CpuChipIcon className="h-8 w-8 text-indigo-400 flex-shrink-0 mt-1" />
                        <div className="rounded-xl px-4 py-3 bg-gray-700 shadow-md">
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
             <div className="mt-2 pt-2 flex-shrink-0 px-2">
                {attachedFile && (
                    <div className="flex items-center justify-between bg-gray-600/50 text-sm rounded-lg px-3 py-2 mb-2 animate-fadeIn">
                        <div className="flex items-center gap-2 text-gray-300 overflow-hidden">
                            {attachedFile.type.startsWith('image/') && <PhotoIcon className="h-5 w-5 flex-shrink-0"/>}
                            {attachedFile.type.startsWith('video/') && <VideoCameraIcon className="h-5 w-5 flex-shrink-0"/>}
                            {!attachedFile.type.startsWith('image/') && !attachedFile.type.startsWith('video/') && <DocumentIcon className="h-5 w-5 flex-shrink-0"/>}
                            <span className="font-medium text-white truncate">{attachedFile.name}</span>
                        </div>
                        <button onClick={onRemoveFile} className="text-gray-400 hover:text-white flex-shrink-0 ml-2">
                            <XCircleIcon className="h-5 w-5"/>
                        </button>
                    </div>
                )}
                 <div className="relative flex items-end gap-2 p-2 bg-gray-900/50 rounded-xl border border-gray-700">
                    <input type="file" ref={fileInputRef} onChange={onFileChange} className="hidden" accept="image/*,video/*" />
                    <button
                        type="button"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isThinking}
                        className="p-2 text-gray-400 hover:text-white transition-colors rounded-full focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500"
                        aria-label="Attach file"
                    >
                        <PaperClipIcon className="h-6 w-6"/>
                    </button>
                    
                    <textarea
                        ref={textareaRef}
                        value={userInput}
                        onChange={(e) => onUserInputChange(e.target.value)}
                        onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSubmit(); } }}
                        placeholder="Interact with Janus..."
                        className="flex-grow bg-transparent text-gray-200 resize-none focus:outline-none placeholder-gray-500 max-h-40 overflow-y-auto"
                        rows={1}
                        disabled={isThinking}
                    />

                    <button
                        type="button"
                        onClick={onWebSearchToggle}
                        aria-checked={useWebSearch}
                        className={`p-2 rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-800 focus:ring-indigo-500 ${useWebSearch ? 'text-indigo-400' : 'text-gray-400 hover:text-white'}`}
                        aria-label="Toggle web search"
                        title="Toggle Web Search"
                    >
                        <GlobeAltIcon className="h-6 w-6"/>
                    </button>

                    <Button
                        onClick={onSubmit}
                        isLoading={isThinking}
                        disabled={!userInput.trim() && !attachedFile}
                        className="p-2.5 rounded-lg"
                        aria-label="Send message"
                    >
                        <PaperAirplaneIcon className="h-6 w-6"/>
                    </Button>
                </div>
            </div>
        </div>
    );
};