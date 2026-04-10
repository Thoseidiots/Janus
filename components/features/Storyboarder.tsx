import React, { useState, useCallback } from 'react';
import { generateStoryboardFromFile, generateStoryboardFromUrl } from '../../services/geminiService';
import { Storyboard } from '../../types';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { FileUpload } from '../common/FileUpload';
import { Spinner } from '../common/Spinner';
import { FilmIcon } from '@heroicons/react/24/solid';

type InputType = 'upload' | 'url';

export const Storyboarder: React.FC = () => {
    const [inputType, setInputType] = useState<InputType>('upload');
    const [sourceFile, setSourceFile] = useState<File | null>(null);
    const [videoUrl, setVideoUrl] = useState('');
    const [storyboard, setStoryboard] = useState<Storyboard | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const resetState = () => {
        setStoryboard(null);
        setError(null);
    };

    const handleFileSelect = useCallback((file: File) => {
        setSourceFile(file);
        setVideoUrl('');
        setInputType('upload');
        resetState();
    }, []);

    const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setVideoUrl(e.target.value);
        setSourceFile(null);
        setInputType('url');
        resetState();
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        const isReady = (inputType === 'upload' && sourceFile) || (inputType === 'url' && videoUrl);
        if (!isReady || isLoading) return;
        
        setIsLoading(true);
        resetState();

        try {
            let result;
            if (inputType === 'upload' && sourceFile) {
                result = await generateStoryboardFromFile(sourceFile);
            } else if (inputType === 'url' && videoUrl) {
                result = await generateStoryboardFromUrl(videoUrl);
            } else {
                throw new Error("No valid input provided.");
            }
            setStoryboard(result);
        } catch (err) {
            const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred during analysis.';
            setError(`Failed to generate storyboard. ${errorMessage}`);
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    };

    const isSubmitDisabled = isLoading || (inputType === 'upload' && !sourceFile) || (inputType === 'url' && !videoUrl.trim());

    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">AI Storyboarder</h2>
            <p className="text-gray-400">Upload a video or provide a YouTube URL to let Gemini analyze its content, identify key scenes, and generate a narrative storyboard with keyframes.</p>
            
            <Card>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="flex items-center space-x-2 bg-gray-800 rounded-lg p-1 w-min">
                        <button type="button" onClick={() => setInputType('upload')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${inputType === 'upload' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}>Upload File</button>
                        <button type="button" onClick={() => setInputType('url')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${inputType === 'url' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}>YouTube URL</button>
                    </div>

                    {inputType === 'upload' ? (
                         <FileUpload 
                            onFileSelect={handleFileSelect} 
                            accept="video/*" 
                            fileType="video"
                        />
                    ) : (
                        <div>
                            <label htmlFor="youtube_url" className="block text-sm font-medium text-gray-300 mb-1">YouTube Video URL</label>
                            <input
                                id="youtube_url"
                                type="url"
                                value={videoUrl}
                                onChange={handleUrlChange}
                                placeholder="https://www.youtube.com/watch?v=..."
                                className="w-full p-3 bg-gray-700 border border-gray-600 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none"
                                disabled={isLoading}
                            />
                        </div>
                    )}
                    
                    <div className="text-right">
                        <Button type="submit" isLoading={isLoading} disabled={isSubmitDisabled} className="w-full sm:w-auto">
                            Generate Storyboard
                        </Button>
                    </div>
                </form>
            </Card>

            {isLoading && (
                 <Card className="flex flex-col items-center justify-center p-8">
                    <Spinner text="Analyzing video and generating keyframes... This may take a minute." />
                </Card>
            )}

            {error && (
                <Card>
                    <p className="text-red-400 text-center">Error: {error}</p>
                </Card>
            )}

            {storyboard && !isLoading && (
                <div className="space-y-6">
                    <Card>
                        <h3 className="text-2xl font-semibold mb-2">Overall Summary</h3>
                        <p className="text-gray-300">{storyboard.overallSummary}</p>
                    </Card>

                    <h3 className="text-2xl font-semibold">Detected Scenes</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {storyboard.scenes.map((scene) => (
                            <Card key={scene.sceneNumber} className="flex flex-col p-0">
                                <div className="relative aspect-video bg-gray-700 rounded-t-xl overflow-hidden">
                                    {scene.keyframeImageUrl ? (
                                        <img src={scene.keyframeImageUrl} alt={scene.keyframeDescription} className="w-full h-full object-cover" />
                                    ) : (
                                        <div className="w-full h-full flex items-center justify-center">
                                            <FilmIcon className="h-12 w-12 text-gray-500" />
                                        </div>
                                    )}
                                    <div className="absolute bottom-0 left-0 bg-black bg-opacity-60 text-white px-2 py-1 text-xs font-mono rounded-tr-lg">
                                        {scene.startTime} - {scene.endTime}
                                    </div>
                                </div>
                                <div className="p-4 flex-grow flex flex-col">
                                    <h4 className="text-lg font-bold mb-2 text-white">{`Scene ${scene.sceneNumber}`}</h4>
                                    <p className="text-sm text-gray-300 mb-4 flex-grow">{scene.sceneDescription}</p>
                                    <div>
                                        <h5 className="text-sm font-semibold mb-2 text-gray-200">Key Moments:</h5>
                                        <ul className="list-disc list-inside text-xs text-gray-400 space-y-1">
                                            {scene.keyMoments.map((moment, i) => (
                                                <li key={i}>{moment}</li>
                                            ))}
                                        </ul>
                                    </div>
                                </div>
                            </Card>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};