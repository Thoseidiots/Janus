import React, { useState, useCallback, FC } from 'react';
import { analyzeImage, analyzeVideo } from '../../services/geminiService';
import { fileToBase64 } from '../../utils/fileUtils';
import { Card } from '../common/Card';
import { Button } from '../common/Button';
import { FileUpload } from '../common/FileUpload';
import { Spinner } from '../common/Spinner';
import ReactMarkdown from 'react-markdown';
import { DocumentChartBarIcon } from '@heroicons/react/24/solid';

type AnalysisMode = 'image' | 'video';

export const AnalysisTools: FC = () => {
    const [mode, setMode] = useState<AnalysisMode>('image');
    const [prompt, setPrompt] = useState('');
    const [sourceFile, setSourceFile] = useState<File | null>(null);
    const [analysisResult, setAnalysisResult] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleFileSelect = useCallback((file: File) => {
        setSourceFile(file);
        setAnalysisResult(null);
        setError(null);
    }, []);
    
    const handleModeChange = useCallback((newMode: AnalysisMode) => {
        setMode(newMode);
        setSourceFile(null);
        setAnalysisResult(null);
        setError(null);
        setPrompt('');
    }, []);

    const handleSubmit = useCallback(async (e: React.FormEvent) => {
        e.preventDefault();
        if (!prompt || !sourceFile || isLoading) return;
        
        setIsLoading(true);
        setError(null);
        setAnalysisResult(null);

        try {
            let result: string;
            if (mode === 'image') {
                const base64 = await fileToBase64(sourceFile);
                result = await analyzeImage(prompt, base64, sourceFile.type);
            } else {
                result = await analyzeVideo(prompt, sourceFile);
            }
            setAnalysisResult(result);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An unknown error occurred.');
        } finally {
            setIsLoading(false);
        }
    }, [prompt, sourceFile, isLoading, mode]);

    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-white">Content Analyzer</h2>
            <p className="text-gray-400">Upload an image of a loss curve or a video of a training session to get AI-powered insights.</p>
            
            <div className="flex items-center space-x-2 bg-gray-800 rounded-lg p-1 w-min">
                <button onClick={() => handleModeChange('image')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${mode === 'image' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}>Analyze Image</button>
                <button onClick={() => handleModeChange('video')} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${mode === 'video' ? 'bg-indigo-600 text-white' : 'text-gray-300 hover:bg-gray-700'}`}>Analyze Video</button>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    <form onSubmit={handleSubmit} className="space-y-4">
                        <h3 className="text-xl font-semibold">Analysis Input</h3>
                        <FileUpload 
                            onFileSelect={handleFileSelect} 
                            accept={mode === 'image' ? 'image/*' : 'video/*'} 
                            fileType={mode}
                        />
                        <div>
                            <label htmlFor="prompt" className="block text-sm font-medium text-gray-300 mb-1">Analysis Prompt</label>
                            <textarea
                                id="prompt"
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder="e.g., Is this model overfitting based on the loss curve?"
                                className="w-full h-24 p-3 bg-gray-700 border border-gray-600 rounded-lg"
                                disabled={isLoading}
                            />
                        </div>
                        <Button type="submit" isLoading={isLoading} disabled={!prompt || !sourceFile} className="w-full">
                           Analyze
                        </Button>
                    </form>
                </Card>
                <Card className="flex items-center justify-center min-h-[300px]">
                    {isLoading && <Spinner text="Analyzing content..." />}
                    {error && <p className="text-red-400 text-center">Error: {error}</p>}
                    {analysisResult && !isLoading && (
                        <div className="space-y-4 w-full">
                            <h3 className="text-xl font-semibold">Analysis Result</h3>
                            <div className="prose prose-invert max-w-none bg-gray-900 p-4 rounded-lg">
                                <ReactMarkdown>{analysisResult}</ReactMarkdown>
                            </div>
                        </div>
                    )}
                     {!isLoading && !analysisResult && !error && (
                        <div className="text-center text-gray-500">
                             <DocumentChartBarIcon className="h-16 w-16 mx-auto mb-2"/>
                             Your analysis will appear here.
                        </div>
                    )}
                </Card>
            </div>
        </div>
    );
};