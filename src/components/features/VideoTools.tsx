import React, { useState, useCallback, FC } from 'react';
import { generateVideoFromText, generateVideoFromImage } from '../../services/geminiService';
import { fileToBase64 } from '../../utils/fileUtils';
import { useVeo, VeoStatus } from '../../hooks/useVeo';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { FileUpload } from '../common/FileUpload';
import { Spinner } from '../common/Spinner';
import { VideoCameraIcon } from '@heroicons/react/24/outline';

type VideoMode = 'text-to-video' | 'image-to-video';
type AspectRatio = '16:9' | '9:16';

export const VideoTools: FC = () => {
    const [mode, setMode] = useState<VideoMode>('text-to-video');
    const [prompt, setPrompt] = useState('');
    const [aspectRatio, setAspectRatio] = useState<AspectRatio>('16:9');
    const [sourceImage, setSourceImage] = useState<{ file: File; base64: string; mimeType: string } | null>(null);
    const { status, videoUrl, error, apiKeySelected, handleGeneration, selectApiKey, reset } = useVeo();

    const handleFileSelect = useCallback(async (file: File) => {
        try {
            const base64 = await fileToBase64(file);
            setSourceImage({ file, base64, mimeType: file.type });
        } catch (err) {
            console.error('Failed to read file for video generation.', err);
        }
    }, []);

    const handleSubmit = useCallback((e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt || (mode === 'image-to-video' && !sourceImage)) return;

        const startGeneration = () => {
            if (mode === 'text-to-video') {
                return generateVideoFromText(prompt, aspectRatio);
            } else if (sourceImage) {
                return generateVideoFromImage(prompt, sourceImage.base64, sourceImage.mimeType, aspectRatio);
            }
            return Promise.reject("Invalid mode or missing source image.");
        };

        handleGeneration(startGeneration);
    }, [prompt, mode, sourceImage, aspectRatio, handleGeneration]);

    const handleModeChange = useCallback((newMode: VideoMode) => {
        setMode(newMode);
        setPrompt('');
        setSourceImage(null);
        reset();
    }, [reset]);

    const renderForm = () => (
        <form onSubmit={handleSubmit} className="space-y-4">
            <h3 className="text-xl font-semibold">{mode === 'text-to-video' ? 'Generate Video from Text' : 'Animate Image'}</h3>
            <div>
                <label htmlFor="prompt" className="block text-sm font-medium text-gray-300 mb-1">Prompt</label>
                <textarea
                    id="prompt"
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    placeholder={mode === 'text-to-video' ? "e.g., A robot holding a red skateboard" : "e.g., Animate this to show the forward pass"}
                    className="w-full h-24 p-3 bg-gray-700 border border-gray-600 rounded-lg"
                    disabled={status === VeoStatus.GENERATING}
                />
            </div>

            {mode === 'image-to-video' && (
                <div>
                     <label className="block text-sm font-medium text-gray-300 mb-1">Upload Image</label>
                    <FileUpload onFileSelect={handleFileSelect} accept="image/png, image/jpeg" fileType="image" />
                    {sourceImage && (
                        <div className="mt-2 text-center">
                            <img src={URL.createObjectURL(sourceImage.file)} alt="For animation" className="max-h-24 rounded-lg inline-block" />
                        </div>
                    )}
                </div>
            )}
            
            <div>
                 <label className="block text-sm font-medium text-gray-300 mb-1">Aspect Ratio</label>
                <div className="flex gap-2">
                    <button type="button" onClick={() => setAspectRatio('16:9')} className={`py-2 px-4 rounded-lg ${aspectRatio === '16:9' ? 'bg-indigo-600' : 'bg-gray-700'}`}>16:9 (Landscape)</button>
                    <button type="button" onClick={() => setAspectRatio('9:16')} className={`py-2 px-4 rounded-lg ${aspectRatio === '9:16' ? 'bg-indigo-600' : 'bg-gray-700'}`}>9:16 (Portrait)</button>
                </div>
            </div>

            <Button type="submit" isLoading={status === VeoStatus.GENERATING} disabled={status === VeoStatus.GENERATING || !prompt || (mode === 'image-to-video' && !sourceImage)} className="w-full">
                Generate Video
            </Button>
        </form>
    );

    if (!apiKeySelected && status !== VeoStatus.ERROR) {
        return (
            <div className="space-y-6">
                <h2 className="text-3xl font-bold text-white">Video Studio</h2>
                <Card className="text-center">
                    <h3 className="text-xl font-semibold mb-2">API Key Required for Video Generation</h3>
                    <p className="text-gray-400 mb-4">The Veo model requires you to select an API key. This helps manage resource allocation for this powerful feature.</p>
                    <p className="text-sm text-gray-500 mb-4">For more information on billing, visit <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer" className="text-indigo-400 hover:underline">ai.google.dev/gemini-api/docs/billing</a>.</p>
                    <Button onClick={selectApiKey} isLoading={status === VeoStatus.CHECKING_KEY}>
                        Select API Key
                    </Button>
                    {error && <p className="mt-4 text-red-400">Error: {error}</p>}
                </Card>
            </div>
        );
    }
    
    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Video Studio</h2>
            <p className="text-gray-400">Generate high-quality videos from text prompts, or bring your static images to life with animation.</p>

            <div className="flex items-center space-x-2 bg-gray-800 rounded-lg p-1 w-min">
                <button onClick={() => handleModeChange('text-to-video')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${mode === 'text-to-video' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}>Text-to-Video</button>
                <button onClick={() => handleModeChange('image-to-video')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${mode === 'image-to-video' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}>Image-to-Video</button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    {renderForm()}
                </Card>
                <Card className="flex items-center justify-center min-h-[300px]">
                    {status === VeoStatus.GENERATING && <Spinner text="Generating video... This may take a few minutes." />}
                    {error && (
                        <div className="text-center text-red-400">
                            <p>Error: {error}</p>
                            {status === VeoStatus.NEEDS_KEY && <Button onClick={selectApiKey} className="mt-4">Select New API Key</Button>}
                        </div>
                    )}
                    {videoUrl && status === VeoStatus.SUCCESS && (
                        <div className="w-full">
                            <h3 className="text-xl font-semibold text-center mb-4">Generated Video</h3>
                            <video src={videoUrl} controls autoPlay loop className="w-full rounded-lg" />
                        </div>
                    )}
                    {status === VeoStatus.IDLE && !videoUrl && (
                        <div className="text-center text-gray-500">
                            <VideoCameraIcon className="h-16 w-16 mx-auto mb-2"/>
                            Your generated video will appear here.
                        </div>
                    )}
                </Card>
            </div>
        </div>
    );
};