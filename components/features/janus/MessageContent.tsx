import React from 'react';
import ReactMarkdown from 'react-markdown';
import { UnifiedChatMessage } from '../../../types';
import { AudioPlayer } from '../../common/AudioPlayer';
import { Spinner } from '../../common/Spinner';
import { FilmIcon } from '@heroicons/react/24/solid';


const getHostname = (uri: string | undefined): string => {
    if (!uri) return 'source';
    try {
        return new URL(uri).hostname.replace('www.', '');
    } catch (e) {
        return uri.length > 30 ? uri.slice(0, 27) + '...' : uri;
    }
};

export const MessageContent: React.FC<{ message: UnifiedChatMessage }> = ({ message }) => {
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